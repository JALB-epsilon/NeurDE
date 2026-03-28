import argparse
import os

import h5py
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from architectures import NeurDE, RESIDUAL_MODES
from burgers_solver import (
    BurgersSolver,
    default_config_path,
    resolve_config_path,
    resolve_artifact_path,
    get_model_config,
    resolve_stabilizer_kwargs,
    resolve_torch_dtype,
)


def compute_split_index(total_steps, train_fraction):
    split = int(total_steps * train_fraction)
    return max(1, min(split, total_steps))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=default_config_path())
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--train_fraction", type=float, default=0.5)
    parser.add_argument("--train_count", type=int, default=None)
    parser.add_argument("--epochs_override", type=int, default=None)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    args = parser.parse_args()
    dtype = resolve_torch_dtype(args.dtype)

    config_path = resolve_config_path(args.config)
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)
    artifact_root = config.get("artifact_root")
    data_path = resolve_artifact_path(config["data_dir"], artifact_root=artifact_root)
    results_dir = resolve_artifact_path(config["results_dir"], artifact_root=artifact_root)
    model_config = get_model_config(config)
    supervision_mode = str(config.get("train", {}).get("supervision", "feq")).lower()
    if supervision_mode not in {"feq", "macro"}:
        raise ValueError(f"Unsupported Burgers train.supervision: {supervision_mode}")

    device = args.device
    with h5py.File(data_path, "r") as handle:
        total_steps = handle["u"].shape[0]
        limit = total_steps if args.num_samples is None else min(args.num_samples, total_steps)
        if args.train_count is not None:
            split_idx = max(1, min(int(args.train_count), limit))
        else:
            split_idx = compute_split_index(limit, args.train_fraction)
        u = torch.as_tensor(handle["u"][:split_idx], dtype=dtype)
        feq = None
        if supervision_mode == "feq":
            if "Feq" not in handle:
                raise ValueError("Burgers train.supervision='feq' requires Feq in the dataset.")
            feq = torch.as_tensor(handle["Feq"][:split_idx], dtype=dtype)

    dataset = TensorDataset(u) if feq is None else TensorDataset(u, feq)
    dataloader = DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
    )

    solver = BurgersSolver(
        X=config["X"],
        lam=config["lam"],
        omega=config["omega"],
        lattice=config.get("lattice", "D1Q2"),
        equilibrium_mode=config.get("equilibrium_mode", "default"),
        device=device,
        newton_steps=config.get("newton_steps", 50),
        newton_tol=config.get("newton_tol", 1e-10),
        domain_length=config.get("domain_length", 1.0),
        alpha=config.get("alpha", 0.5),
        s2=config.get("s2", 1.7),
        s3=config.get("s3", 1.7),
        boundary=config.get("boundary", "outflow"),
        u_left_bc=config.get("u_left"),
        u_right_bc=config.get("u_right"),
        dtype=dtype,
        **resolve_stabilizer_kwargs(config),
    )

    model = NeurDE(
        alpha_layer=[1] + [config["hidden_dim"]] * config["num_layers"],
        phi_layer=[solver.basis().shape[-1]] + [config["hidden_dim"]] * config["num_layers"],
        activation="relu",
        learn_feq=True,
        learn_geq=False,
        feq_mode=model_config["feq_mode"],
        logit_clip=model_config["logit_clip"],
    ).to(device=device, dtype=dtype)

    basis = solver.basis().to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"])

    os.makedirs(results_dir, exist_ok=True)
    split_label = f"train_count={split_idx}" if args.train_count is not None else f"train_fraction={args.train_fraction:.3f}"
    print(
        f"Training Burgers on {split_idx}/{limit} snapshots ({split_label}, "
        f"feq_mode={model_config['feq_mode']}, logit_clip={model_config['logit_clip']}, "
        f"supervision={supervision_mode}, data_path={data_path}, results_dir={results_dir})"
    )
    epochs = args.epochs_override or int(config["train"]["epochs"])
    best_loss = float("inf")
    best_state_dict = None
    best_epoch = None
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            u_batch = batch[0].to(device=device, dtype=dtype)
            optimizer.zero_grad()
            inputs = u_batch.unsqueeze(1).unsqueeze(2)
            model_kwargs = {}
            if supervision_mode == "feq" and model_config["feq_mode"] in RESIDUAL_MODES:
                with torch.no_grad():
                    model_kwargs["feq_base"] = solver.equilibrium(u_batch)
            feq_pred = model(inputs, basis, **model_kwargs)
            if supervision_mode == "feq":
                feq_batch = batch[1].to(device=device, dtype=dtype)
                targets = feq_batch.permute(0, 2, 1).reshape(-1, solver.Qn)
                loss = torch.norm(feq_pred - targets) / (torch.norm(targets) + 1e-7)
            else:
                feq_pred_pop = feq_pred.reshape(u_batch.shape[0], solver.X, solver.Qn).permute(0, 2, 1)
                macro_pred = solver.macro(feq_pred_pop)
                loss = torch.norm(macro_pred - u_batch) / (torch.norm(u_batch) + 1e-7)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / max(len(dataloader), 1)
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_epoch = epoch
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        if epoch % 25 == 0:
            print(f"Epoch {epoch}: loss={avg_epoch_loss:.6f}")

    output_path = os.path.join(results_dir, "burgers_stage1.pt")
    best_path = os.path.join(results_dir, "burgers_stage1_best.pt")
    last_path = os.path.join(results_dir, "burgers_stage1_last.pt")

    if best_state_dict is None:
        raise RuntimeError("Stage-1 training did not produce a valid checkpoint.")

    torch.save(best_state_dict, output_path)
    torch.save(best_state_dict, best_path)
    torch.save(model.state_dict(), last_path)
    print(f"Saved best Burgers model to {output_path} (epoch={best_epoch}, loss={best_loss:.6f})")
    print(f"Saved explicit best checkpoint to {best_path}")
    print(f"Saved last checkpoint to {last_path}")


if __name__ == "__main__":
    main()
