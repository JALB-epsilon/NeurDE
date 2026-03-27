import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from burgers_solver import (
    BurgersSolver,
    burgers_sinusoidal_initial,
    burgers_sinusoidal_shock_time,
    default_config_path,
    exact_burgers_riemann,
    resolve_config_path,
    resolve_module_path,
    resolve_stabilizer_kwargs,
)


def parse_snapshot_times(snapshot_times, final_time):
    if snapshot_times is None:
        if final_time is None:
            return None
        return np.linspace(0.0, final_time, num=6).tolist()
    if isinstance(snapshot_times, (list, tuple)):
        return [float(value) for value in snapshot_times]
    if isinstance(snapshot_times, str):
        values = [value.strip() for value in snapshot_times.split(",") if value.strip()]
        return [float(value) for value in values]
    return [float(snapshot_times)]


def discrete_max_gradient(u_values, dx):
    return float(np.max(np.abs(np.diff(u_values) / max(dx, 1.0e-12))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=default_config_path())
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--final_time", type=float, default=None)
    parser.add_argument("--snapshot_times", type=str, default=None)
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)

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
        **resolve_stabilizer_kwargs(config),
    )

    steps = args.steps
    if args.final_time is not None:
        steps = max(int(round(args.final_time / solver.dt)) + 1, 1)

    out_dir = resolve_module_path("images")
    os.makedirs(out_dir, exist_ok=True)
    mode_suffix = ""
    if solver.lattice in {"D1Q3", "D2Q9"}:
        mode_suffix = f"_{solver.equilibrium_mode}"
    omega_suffix = f"_omega{str(solver.omega).replace('.', 'p')}"
    x = solver.x_grid()
    initial_condition = str(config.get("initial_condition", "riemann")).lower()

    if initial_condition == "sinusoidal":
        u0 = burgers_sinusoidal_initial(
            x,
            mean=config.get("sine_mean", 0.6),
            amplitude=config.get("sine_amplitude", 0.4),
            wavenumber=config.get("sine_wavenumber", 1.0),
            phase=config.get("sine_phase", 0.0),
            domain_length=config.get("domain_length", 1.0),
        )
        shock_time = burgers_sinusoidal_shock_time(
            amplitude=config.get("sine_amplitude", 0.4),
            wavenumber=config.get("sine_wavenumber", 1.0),
            domain_length=config.get("domain_length", 1.0),
        )
        F = solver.equilibrium(torch.tensor(u0, dtype=torch.float32, device=solver.device))
        final_time = float(steps - 1) * solver.dt
        snapshot_times = parse_snapshot_times(args.snapshot_times or config.get("benchmark_snapshot_times"), final_time)
        snapshot_indices = sorted({max(0, min(int(round(t / solver.dt)), steps - 1)) for t in snapshot_times})
        snapshots = {}
        gradients = {}

        for step in range(steps):
            u = solver.macro(F)
            if step in snapshot_indices:
                u_np = u.detach().cpu().numpy()
                snapshots[step] = u_np
                gradients[step] = discrete_max_gradient(u_np, solver.dx)
            if step < steps - 1:
                F, _, _ = solver.step(F)

        ncols = 3
        nrows = math.ceil(len(snapshot_indices) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.8 * nrows), sharex=True, sharey=True)
        axes = np.atleast_1d(axes).reshape(-1)
        for ax, step in zip(axes, snapshot_indices):
            time_value = float(step) * solver.dt
            ax.plot(x, u0, color="0.75", linewidth=1.6, label="initial" if step == snapshot_indices[0] else None)
            ax.plot(x, snapshots[step], color="#d62728", linewidth=2.2, label="lbm" if step == snapshot_indices[0] else None)
            ax.set_title(f"t={time_value:.3f} | max|du/dx|={gradients[step]:.2f}")
            ax.set_xlabel("x")
            ax.set_ylabel("u")
            ax.grid(True, alpha=0.2)
        for ax in axes[len(snapshot_indices):]:
            ax.axis("off")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
        title = f"Burgers {solver.lattice} {solver.equilibrium_mode} sinusoidal rollout"
        if np.isfinite(shock_time):
            title += f" | estimated shock time {shock_time:.3f}"
        fig.suptitle(title, y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.965])
        time_suffix = f"_t{final_time:.4f}".replace(".", "p")
        plot_path = os.path.join(out_dir, f"{solver.lattice.lower()}{mode_suffix}{omega_suffix}{time_suffix}_sinusoidal_rollout.png")
        fig.savefig(plot_path)
        plt.close(fig)

        print(
            {
                "lattice": solver.lattice,
                "steps": steps,
                "dt": solver.dt,
                "dx": solver.dx,
                "initial_condition": initial_condition,
                "final_time": final_time,
                "estimated_shock_time": None if not np.isfinite(shock_time) else float(shock_time),
                "snapshot_times": [float(step) * solver.dt for step in snapshot_indices],
                "max_gradients": {f"{float(step) * solver.dt:.4f}": gradients[step] for step in snapshot_indices},
                "plot_path": plot_path,
            }
        )
        return

    x0 = solver.physical_x0(config["x0"])
    u0 = exact_burgers_riemann(x, 0.0, config["u_left"], config["u_right"], x0)
    F = solver.equilibrium(torch.tensor(u0, dtype=torch.float32, device=solver.device))

    rel_errors = []
    final_u = None
    for step in range(steps):
        u = solver.macro(F)
        u_exact = torch.tensor(
            exact_burgers_riemann(x, float(step) * solver.dt, config["u_left"], config["u_right"], x0),
            dtype=torch.float32,
            device=solver.device,
        )
        rel = torch.norm(u - u_exact) / (torch.norm(u_exact) + 1e-7)
        rel_errors.append(float(rel))
        F, _, _ = solver.step(F)
        final_u = u

    final_exact = torch.tensor(
        exact_burgers_riemann(x, float(steps - 1) * solver.dt, config["u_left"], config["u_right"], x0),
        dtype=torch.float32,
        device=solver.device,
    )
    final_rel = torch.norm(final_u - final_exact) / (torch.norm(final_exact) + 1e-7)
    final_time = float(steps - 1) * solver.dt
    time_suffix = f"_t{final_time:.4f}".replace(".", "p")
    plot_path = os.path.join(out_dir, f"{solver.lattice.lower()}{mode_suffix}{omega_suffix}{time_suffix}_lbm_vs_exact.png")

    plt.figure(figsize=(10, 4))
    plt.plot(x, final_exact.cpu().numpy(), label="exact", linewidth=2)
    plt.plot(x, final_u.cpu().numpy(), label=f"{solver.lattice} lbm", linewidth=2)
    plt.title(f"Burgers {solver.lattice} {solver.equilibrium_mode} omega={solver.omega:g} vs Exact at t={final_time:.4f}")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print(
        {
            "lattice": solver.lattice,
            "steps": steps,
            "dt": solver.dt,
            "dx": solver.dx,
            "final_time": final_time,
            "avg_rel_error": sum(rel_errors) / len(rel_errors),
            "final_rel_error": float(final_rel),
            "max_rel_error": max(rel_errors),
            "plot_path": plot_path,
        }
    )


if __name__ == "__main__":
    main()
