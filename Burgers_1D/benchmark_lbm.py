import argparse
import os

import matplotlib.pyplot as plt
import torch
import yaml

from burgers_solver import BurgersSolver, default_config_path, exact_burgers_riemann, resolve_config_path, resolve_module_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=default_config_path())
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--final_time", type=float, default=None)
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
    )

    steps = args.steps
    if args.final_time is not None:
        steps = max(int(round(args.final_time / solver.dt)) + 1, 1)

    x = solver.x_grid()
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

    out_dir = resolve_module_path("images")
    os.makedirs(out_dir, exist_ok=True)
    mode_suffix = ""
    if solver.lattice == "D1Q3":
        mode_suffix = f"_{solver.equilibrium_mode}"
    omega_suffix = f"_omega{str(solver.omega).replace('.', 'p')}"
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
