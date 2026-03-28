import argparse
import os

import h5py
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from architectures import NeurDE
from burgers_solver import (
    BurgersSolver,
    default_config_path,
    resolve_config_path,
    resolve_module_path,
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
    data_path = resolve_module_path(config["data_dir"])
    results_dir = resolve_module_path(config["results_dir"])
    conservative_output = config.get("conservative_output", config.get("match_mass", True))

    device = args.device
    with h5py.File(data_path, "r") as handle:
        total_steps = handle["u"].shape[0]
        limit = total_steps if args.num_samples is None else min(args.num_samples, total_steps)
        if args.train_count is not None:
            split_idx = max(1, min(int(args.train_count), limit))
        else:
            split_idx = compute_split_index(limit, args.train_fraction)
        u = torch.as_tensor(handle["u"][:split_idx], dtype=dtype)
        feq = torch.as_tensor(handle["Feq"][:split_idx], dtype=dtype)

    dataset = TensorDataset(u, feq)
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
        phi_layer=[1] + [config["hidden_dim"]] * config["num_layers"],
        activation="relu",
        learn_feq=True,
        learn_geq=False,
        logit_clip=config.get("logit_clip", 15.0),
        conservative_output=conservative_output,
    ).to(device=device, dtype=dtype)

    basis = solver.basis().to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"])

    os.makedirs(results_dir, exist_ok=True)
    split_label = f"train_count={split_idx}" if args.train_count is not None else f"train_fraction={args.train_fraction:.3f}"
    print(
        f"Training Burgers on {split_idx}/{limit} snapshots ({split_label}, "
        f"conservative_output={conservative_output}, logit_clip={config.get('logit_clip', 15.0)})"
    )
    epochs = args.epochs_override or int(config["train"]["epochs"])
    for epoch in range(epochs):
        epoch_loss = 0.0
        for u_batch, feq_batch in dataloader:
            inputs = u_batch.unsqueeze(1).unsqueeze(2).to(device)
            targets = feq_batch.permute(0, 2, 1).reshape(-1, solver.Qn).to(device)
            optimizer.zero_grad()
            feq_pred = model(inputs, basis)
            loss = torch.norm(feq_pred - targets) / (torch.norm(targets) + 1e-7)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 25 == 0:
            print(f"Epoch {epoch}: loss={epoch_loss / max(len(dataloader), 1):.6f}")

    output_path = os.path.join(results_dir, "burgers_stage1.pt")
    torch.save(model.state_dict(), output_path)
    print(f"Saved Burgers model to {output_path}")


if __name__ == "__main__":
    main()
