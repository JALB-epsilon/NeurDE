import argparse
import os

import h5py
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from architectures import NeurDE
from buckley_leverett_solver import BuckleyLeverettSolver, default_config_path, resolve_config_path, resolve_module_path, resolve_torch_dtype


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=default_config_path())
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    args = parser.parse_args()
    dtype = resolve_torch_dtype(args.dtype)

    config_path = resolve_config_path(args.config)
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)
    data_path = resolve_module_path(config["data_dir"])
    results_dir = resolve_module_path(config["results_dir"])
    conservative_output = config.get("conservative_output", config.get("match_mass", True))
    supervision_mode = str(config.get("train", {}).get("supervision", "feq")).lower()
    if supervision_mode not in {"feq", "macro"}:
        raise ValueError(f"Unsupported Buckley-Leverett train.supervision: {supervision_mode}")

    with h5py.File(data_path, "r") as handle:
        u = torch.as_tensor(handle["u"][: args.num_samples], dtype=dtype)
        feq = None
        if supervision_mode == "feq":
            if "Feq" not in handle:
                raise ValueError("Buckley-Leverett train.supervision='feq' requires Feq in the dataset.")
            feq = torch.as_tensor(handle["Feq"][: args.num_samples], dtype=dtype)

    dataset = TensorDataset(u) if feq is None else TensorDataset(u, feq)
    dataloader = DataLoader(dataset, batch_size=config["train"]["batch_size"], shuffle=True)

    solver = BuckleyLeverettSolver(
        X=config["X"],
        lam=config["lam"],
        omega=config["omega"],
        lattice=config.get("lattice", "D2Q9"),
        device=args.device,
        newton_steps=config.get("newton_steps", 50),
        newton_tol=config.get("newton_tol", 1e-10),
        domain_length=config.get("domain_length", 1.0),
        boundary=config.get("boundary", "riemann"),
        u_left_bc=config.get("u_left"),
        u_right_bc=config.get("u_right"),
        mobility_ratio=config.get("mobility_ratio", 0.5),
        dtype=dtype,
    )

    model = NeurDE(
        alpha_layer=[1] + [config["hidden_dim"]] * config["num_layers"],
        phi_layer=[1] + [config["hidden_dim"]] * config["num_layers"],
        activation="relu",
        learn_feq=True,
        learn_geq=False,
        logit_clip=config.get("logit_clip", 15.0),
        conservative_output=conservative_output,
    ).to(device=args.device, dtype=dtype)

    basis = solver.basis().to(device=args.device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"])
    os.makedirs(results_dir, exist_ok=True)
    print(
        f"Training Buckley-Leverett on {u.shape[0]} snapshots "
        f"(conservative_output={conservative_output}, logit_clip={config.get('logit_clip', 15.0)}, "
        f"supervision={supervision_mode})"
    )

    for epoch in range(config["train"]["epochs"]):
        epoch_loss = 0.0
        for batch in dataloader:
            u_batch = batch[0].to(device=args.device, dtype=dtype)
            optimizer.zero_grad()
            inputs = u_batch.unsqueeze(1).unsqueeze(2)
            feq_pred = model(inputs, basis)
            if supervision_mode == "feq":
                feq_batch = batch[1].to(device=args.device, dtype=dtype)
                targets = feq_batch.permute(0, 2, 1).reshape(-1, solver.Qn)
                loss = torch.norm(feq_pred - targets) / (torch.norm(targets) + 1e-7)
            else:
                feq_pred_pop = feq_pred.reshape(u_batch.shape[0], solver.X, solver.Qn).permute(0, 2, 1)
                macro_pred = solver.macro(feq_pred_pop)
                loss = torch.norm(macro_pred - u_batch) / (torch.norm(u_batch) + 1e-7)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if epoch % 25 == 0:
            print(f"Epoch {epoch}: loss={epoch_loss / max(len(dataloader), 1):.6f}")

    output_path = os.path.join(results_dir, "buckley_leverett_stage1.pt")
    torch.save(model.state_dict(), output_path)
    print(f"Saved Buckley-Leverett model to {output_path}")


if __name__ == "__main__":
    main()
