import argparse
import os

import h5py
import matplotlib.pyplot as plt
import torch
import yaml

from architectures import NeurDE
from burgers_solver import BurgersSolver, default_config_path, resolve_config_path, resolve_module_path


def compute_split_index(total_steps, train_fraction):
    split = int(total_steps * train_fraction)
    return max(1, min(split, total_steps))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=default_config_path())
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--train_fraction", type=float, default=0.5)
    parser.add_argument("--train_count", type=int, default=None)
    parser.add_argument("--start_step", type=int, default=None)
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)
    data_path = resolve_module_path(config["data_dir"])
    model_path = resolve_module_path(args.model_path) if args.model_path is not None else resolve_module_path(
        os.path.join(config["results_dir"], "burgers_stage1.pt")
    )
    conservative_output = config.get("conservative_output", config.get("match_mass", True))

    with h5py.File(data_path, "r") as handle:
        total_steps = handle["u"].shape[0]
        limit = total_steps if args.num_samples is None else min(args.num_samples, total_steps)
        if args.train_count is not None:
            default_start = max(1, min(int(args.train_count), limit))
        else:
            default_start = compute_split_index(limit, args.train_fraction)
        start_step = default_start if args.start_step is None else max(0, min(args.start_step, limit - 1))
        end_step = limit if args.steps is None else min(limit, start_step + args.steps)
        u_ref = torch.tensor(handle["u"][start_step:end_step], dtype=torch.float32)
        if u_ref.shape[0] == 0:
            raise ValueError("Evaluation slice is empty; adjust train_fraction/start_step/steps.")

    device = args.device
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
    )
    model = NeurDE(
        alpha_layer=[1] + [config["hidden_dim"]] * config["num_layers"],
        phi_layer=[1] + [config["hidden_dim"]] * config["num_layers"],
        activation="relu",
        learn_feq=True,
        learn_geq=False,
        logit_clip=config.get("logit_clip", 15.0),
        conservative_output=conservative_output,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    basis = solver.basis().to(device)
    F = solver.equilibrium(u_ref[0].to(device))

    x = solver.x_grid()
    model_stem = os.path.splitext(os.path.basename(model_path))[0]
    out_dir = resolve_module_path(os.path.join("images", "eval", model_stem))
    os.makedirs(out_dir, exist_ok=True)

    rel_error = 0.0
    rel_errors = []
    plotted_u = None
    plotted_target = None
    plotted_step = None
    failure_step = None
    max_abs_u = 0.0
    with torch.no_grad():
        for step in range(u_ref.shape[0]):
            u = solver.macro(F)
            if not torch.isfinite(u).all():
                failure_step = step
                break
            max_abs_u = max(max_abs_u, float(u.detach().abs().max().item()))
            inputs = u.unsqueeze(0).unsqueeze(0).unsqueeze(1)
            Feq_pred = model(inputs, basis).permute(1, 0)
            if not torch.isfinite(Feq_pred).all():
                failure_step = step
                break
            F, _, _ = solver.step(F, Feq_pred)
            target = u_ref[step].to(device)
            step_rel = (torch.norm(u - target) / (torch.norm(target) + 1e-7)).item()
            rel_error += step_rel
            rel_errors.append(step_rel)
            plotted_u = u.detach().cpu()
            plotted_target = target.detach().cpu()
            plotted_step = start_step + step

    if plotted_u is not None and plotted_target is not None:
        final_plot_path = os.path.join(out_dir, f"burgers_eval_step_{plotted_step:04d}.png")
        plt.figure(figsize=(10, 4))
        plt.plot(x, plotted_target.numpy(), label="analytic", linewidth=2)
        plt.plot(x, plotted_u.numpy(), label="nn", linewidth=2)
        plt.xlabel("x")
        plt.ylabel("u")
        title = f"Burgers Holdout Rollout at step {plotted_step}"
        if failure_step is not None:
            title += f" (failed at local step {failure_step})"
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(final_plot_path)
        plt.close()
    else:
        final_plot_path = None

    if rel_errors:
        error_plot_path = os.path.join(out_dir, "burgers_error_accumulation.png")
        plt.figure(figsize=(8, 4))
        plt.plot(range(start_step, start_step + len(rel_errors)), rel_errors, linewidth=2)
        plt.xlabel("step")
        plt.ylabel("relative error")
        plt.title("Burgers Holdout Rollout Error")
        plt.tight_layout()
        plt.savefig(error_plot_path)
        plt.close()
    else:
        error_plot_path = None

    avg_error = rel_error / max(len(rel_errors), 1)
    message = (
        f"Average rollout relative error over {len(rel_errors)} Burgers steps "
        f"(start_step={start_step}, end_step={start_step + len(rel_errors) - 1}): "
        f"{avg_error:.6f}; max|u|={max_abs_u:.6f}; conservative_output={conservative_output}"
    )
    if failure_step is not None:
        message += f"; rollout became non-finite at local step {failure_step} (global step {start_step + failure_step})"
    print(message)
    if final_plot_path is not None:
        print(f"Saved final-step plot to {final_plot_path}")
    if error_plot_path is not None:
        print(f"Saved error plot to {error_plot_path}")


if __name__ == "__main__":
    main()
