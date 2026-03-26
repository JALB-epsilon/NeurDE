import argparse
import os
import random

import h5py
import torch
import yaml

from architectures import NeurDE
from burgers_solver import BurgersSolver, default_config_path, resolve_config_path, resolve_module_path


def reshape_prediction(prediction, x_points, qn):
    return prediction.reshape(1, x_points, qn).permute(0, 2, 1).squeeze(0)


def relative_error(prediction, target, eps=1.0e-7):
    return torch.norm(prediction - target) / (torch.norm(target) + eps)


def resolve_rollout(rollout_schedule, epoch):
    if not rollout_schedule:
        return 1
    remaining = int(epoch)
    last_rollout = int(rollout_schedule[-1]["rollout"])
    for stage in rollout_schedule:
        rollout = int(stage["rollout"])
        stage_epochs = int(stage.get("epochs", 0))
        last_rollout = rollout
        if stage_epochs <= 0:
            return rollout
        if remaining < stage_epochs:
            return rollout
        remaining -= stage_epochs
    return last_rollout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=default_config_path())
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs_override", type=int, default=None)
    parser.add_argument("--train_count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    config_path = resolve_config_path(args.config)
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)
    if "stage2" not in config:
        raise ValueError("Stage-2 config section is required.")
    stage2 = config["stage2"]

    data_path = resolve_module_path(config["data_dir"])
    results_dir = resolve_module_path(stage2["results_dir"])
    pretrained_path = resolve_module_path(stage2["pretrained_path"])
    os.makedirs(results_dir, exist_ok=True)

    with h5py.File(data_path, "r") as handle:
        total_steps = handle["u"].shape[0]
        limit = min(int(stage2.get("num_samples", total_steps)), total_steps)
        train_count = args.train_count if args.train_count is not None else int(stage2.get("train_count", limit))
        train_count = max(2, min(int(train_count), limit))
        all_u = torch.tensor(handle["u"][:train_count], dtype=torch.float32, device=args.device)
        all_feq = torch.tensor(handle["Feq"][:train_count], dtype=torch.float32, device=args.device)

    solver = BurgersSolver(
        X=config["X"],
        lam=config["lam"],
        omega=config["omega"],
        lattice=config.get("lattice", "D1Q2"),
        equilibrium_mode=config.get("equilibrium_mode", "default"),
        device=args.device,
        newton_steps=config.get("newton_steps", 50),
        newton_tol=config.get("newton_tol", 1e-10),
        domain_length=config.get("domain_length", 1.0),
        alpha=config.get("alpha", 0.5),
        s2=config.get("s2", 1.7),
        s3=config.get("s3", 1.7),
        boundary=config.get("boundary", "outflow"),
        u_left_bc=config.get("u_left"),
        u_right_bc=config.get("u_right"),
    )
    model = NeurDE(
        alpha_layer=[1] + [config["hidden_dim"]] * config["num_layers"],
        phi_layer=[1] + [config["hidden_dim"]] * config["num_layers"],
        activation="relu",
        learn_feq=True,
        learn_geq=False,
        logit_clip=config.get("logit_clip", 15.0),
        conservative_output=config.get("conservative_output", config.get("match_mass", True)),
    ).to(args.device)
    model.load_state_dict(torch.load(pretrained_path, map_location=args.device))
    basis = solver.basis().to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=stage2["lr"])
    epochs = args.epochs_override or int(stage2["epochs"])
    rollout_schedule = stage2.get("rollout_schedule", [{"rollout": 1, "epochs": 0}])
    grad_clip_norm = float(stage2.get("grad_clip_norm", 1.0))

    print(
        f"Stage-2 Burgers fine-tune on {args.device}. train_count={train_count}, "
        f"epochs={epochs}, pretrained={pretrained_path}"
    )
    print(f"Rollout schedule: {rollout_schedule}")

    best_loss = float("inf")
    best_path = os.path.join(results_dir, "burgers_stage2_best.pt")

    x_points = solver.X
    for epoch in range(epochs):
        current_rollout = resolve_rollout(rollout_schedule, epoch)
        max_start = train_count - current_rollout - 1
        if max_start < 0:
            raise ValueError(
                f"Not enough training snapshots ({train_count}) for rollout={current_rollout}. "
                f"Need at least rollout + 1 snapshots."
            )

        starts = list(range(max_start + 1))
        random.shuffle(starts)

        epoch_loss = 0.0
        model.train()
        for start in starts:
            F = all_feq[start].clone()
            loss = torch.zeros((), device=args.device)
            optimizer.zero_grad()
            for step in range(current_rollout):
                u_current = solver.macro(F)
                inputs = u_current.unsqueeze(0).unsqueeze(0).unsqueeze(1)
                feq_pred = model(inputs, basis)
                feq_pred = reshape_prediction(feq_pred, x_points, solver.Qn)
                F, _, _ = solver.step(F, feq_pred)
                u_next = solver.macro(F)
                target = all_u[start + step + 1]
                loss = loss + relative_error(u_next, target)
            loss = loss / float(current_rollout)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            epoch_loss += float(loss.item())

        epoch_loss /= max(len(starts), 1)
        print(f"Epoch {epoch}: rollout={current_rollout}, loss={epoch_loss:.6f}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), best_path)

    last_path = os.path.join(results_dir, "burgers_stage2_last.pt")
    torch.save(model.state_dict(), last_path)
    print(f"Saved best stage-2 model to {best_path}")
    print(f"Saved last stage-2 model to {last_path}")


if __name__ == "__main__":
    main()
