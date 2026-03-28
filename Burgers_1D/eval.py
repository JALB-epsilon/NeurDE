import argparse
import os

import h5py
import matplotlib.pyplot as plt
import math
import torch
import yaml

from architectures import NeurDE, RESIDUAL_MODES
from burgers_solver import (
    BurgersSolver,
    default_config_path,
    resolve_config_path,
    resolve_artifact_path,
    get_model_config,
    resolve_module_path,
    resolve_stabilizer_kwargs,
    resolve_torch_dtype,
)


def compute_split_index(total_steps, train_fraction):
    split = int(total_steps * train_fraction)
    return max(1, min(split, total_steps))


def parse_plot_steps(raw_value):
    if not raw_value:
        return set()
    result = set()
    for item in raw_value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        result.add(int(stripped))
    return result


def model_output_slug(model_path):
    model_stem = os.path.splitext(os.path.normpath(model_path))[0]
    parts = [part for part in model_stem.split(os.sep) if part]
    if not parts:
        return "burgers_model"
    return "__".join(parts[-4:])


def plot_rollout_state(x, prediction, target, global_step, time_value, output_path, failure_step=None):
    plt.figure(figsize=(10, 4))
    plt.plot(x, target.numpy(), label="dataset", linewidth=2)
    plt.plot(x, prediction.numpy(), label="nn", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("u")
    title = f"Burgers Holdout Rollout at step {global_step} (t={time_value:.4f})"
    if failure_step is not None:
        title += f" (failed at local step {failure_step})"
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_snapshot_grid(x, snapshots, output_path):
    if not snapshots:
        return None

    columns = min(3, len(snapshots))
    rows = math.ceil(len(snapshots) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 3.5 * rows), squeeze=False)

    for ax, snapshot in zip(axes.flat, snapshots):
        ax.plot(x, snapshot["target"].numpy(), label="dataset", linewidth=2)
        ax.plot(x, snapshot["prediction"].numpy(), label="nn", linewidth=2)
        ax.set_title(f"step {snapshot['global_step']} | t={snapshot['time_value']:.4f} | rel={snapshot['rel_error']:.4f}")
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.grid(True, alpha=0.25)

    for ax in axes.flat[len(snapshots):]:
        ax.axis("off")

    axes.flat[0].legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


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
    parser.add_argument("--plot_every", type=int, default=0, help="Save rollout overlay plots every N steps (0 disables periodic plots).")
    parser.add_argument("--plot_steps", type=str, default="", help="Comma-separated global rollout steps to save explicitly, e.g. 610,620.")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    args = parser.parse_args()
    dtype = resolve_torch_dtype(args.dtype)

    config_path = resolve_config_path(args.config)
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)
    artifact_root = config.get("artifact_root")
    model_config = get_model_config(config)
    data_path = resolve_artifact_path(config["data_dir"], artifact_root=artifact_root)
    model_path = (
        resolve_artifact_path(args.model_path, artifact_root=artifact_root)
        if args.model_path is not None
        else resolve_artifact_path(os.path.join(config["results_dir"], "burgers_stage1.pt"), artifact_root=artifact_root)
    )
    explicit_plot_steps = parse_plot_steps(args.plot_steps)

    with h5py.File(data_path, "r") as handle:
        if "F" not in handle:
            raise ValueError(
                "Burgers evaluation requires population states saved as dataset 'F'. "
                "Regenerate the dataset with Burgers_1D/burgers_solver.py."
            )
        total_steps = handle["u"].shape[0]
        limit = total_steps if args.num_samples is None else min(args.num_samples, total_steps)
        if args.train_count is not None:
            default_start = max(1, min(int(args.train_count), limit))
        else:
            default_start = compute_split_index(limit, args.train_fraction)
        start_step = default_start if args.start_step is None else max(0, min(args.start_step, limit - 1))
        end_step = limit if args.steps is None else min(limit, start_step + args.steps)
        initial_F = torch.as_tensor(handle["F"][start_step], dtype=dtype)
        u_ref = torch.as_tensor(handle["u"][start_step:end_step], dtype=dtype)
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
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    basis = solver.basis().to(device=device, dtype=dtype)
    F = initial_F.to(device).unsqueeze(0)

    x = solver.x_grid()
    out_dir = resolve_module_path(os.path.join("images", "eval", model_output_slug(model_path)))
    os.makedirs(out_dir, exist_ok=True)

    rel_error = 0.0
    rel_errors = []
    plotted_u = None
    plotted_target = None
    plotted_step = None
    snapshots = []
    saved_snapshot_steps = set()
    failure_step = None
    max_abs_u = 0.0
    with torch.no_grad():
        for step in range(u_ref.shape[0]):
            global_step = start_step + step
            u = solver.macro(F)
            if not torch.isfinite(u).all():
                failure_step = step
                break
            max_abs_u = max(max_abs_u, float(u.detach().abs().max().item()))
            inputs = u.unsqueeze(1).unsqueeze(2)
            model_kwargs = {}
            if model_config["feq_mode"] in RESIDUAL_MODES:
                model_kwargs["feq_base"] = solver.equilibrium(u)
            Feq_pred = model(inputs, basis, **model_kwargs).reshape(1, solver.X, solver.Qn).permute(0, 2, 1)
            if not torch.isfinite(Feq_pred).all():
                failure_step = step
                break
            F, _, _ = solver.step(F, Feq_pred)
            target = u_ref[step].to(device).unsqueeze(0)
            step_rel = (torch.norm(u - target) / (torch.norm(target) + 1e-7)).item()
            rel_error += step_rel
            rel_errors.append(step_rel)
            plotted_u = u[0].detach().cpu()
            plotted_target = target[0].detach().cpu()
            plotted_step = global_step
            plotted_time = global_step * solver.dt
            should_plot = (
                global_step in explicit_plot_steps
                or (
                    args.plot_every > 0
                    and (
                    step == 0
                    or ((step % args.plot_every) == 0)
                    or (step == u_ref.shape[0] - 1)
                )
                )
            )
            if should_plot and global_step not in saved_snapshot_steps:
                snapshot_path = os.path.join(out_dir, f"burgers_eval_step_{plotted_step:04d}.png")
                plot_rollout_state(
                    x=x,
                    prediction=plotted_u,
                    target=plotted_target,
                    global_step=plotted_step,
                    time_value=plotted_time,
                    output_path=snapshot_path,
                )
                snapshots.append(
                    {
                        "global_step": plotted_step,
                        "time_value": plotted_time,
                        "prediction": plotted_u,
                        "target": plotted_target,
                        "rel_error": step_rel,
                    }
                )
                saved_snapshot_steps.add(global_step)

    if plotted_u is not None and plotted_target is not None:
        final_plot_path = os.path.join(out_dir, f"burgers_eval_step_{plotted_step:04d}.png")
        plot_rollout_state(
            x=x,
            prediction=plotted_u,
            target=plotted_target,
            global_step=plotted_step,
            time_value=plotted_step * solver.dt,
            output_path=final_plot_path,
            failure_step=failure_step,
        )
    else:
        final_plot_path = None

    snapshot_grid_path = None
    if snapshots:
        snapshot_grid_path = os.path.join(out_dir, "burgers_eval_snapshots.png")
        plot_snapshot_grid(x, snapshots, snapshot_grid_path)

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
        f"{avg_error:.6f}; max|u|={max_abs_u:.6f}; feq_mode={model_config['feq_mode']}; "
        f"data_path={data_path}; model_path={model_path}"
    )
    if failure_step is not None:
        message += f"; rollout became non-finite at local step {failure_step} (global step {start_step + failure_step})"
    print(message)
    if final_plot_path is not None:
        print(f"Saved final-step plot to {final_plot_path}")
    if snapshot_grid_path is not None:
        print(f"Saved snapshot grid to {snapshot_grid_path}")
    if error_plot_path is not None:
        print(f"Saved error plot to {error_plot_path}")


if __name__ == "__main__":
    main()
