import argparse
import os

import matplotlib.pyplot as plt
import torch
import yaml

from buckley_leverett_solver import BuckleyLeverettSolver, default_config_path, reference_rollout, resolve_config_path, resolve_module_path


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
    )

    steps = args.steps
    if args.final_time is not None:
        steps = max(int(round(args.final_time / solver.dt)) + 1, 1)

    x = solver.x_grid()
    x0 = solver.physical_x0(config["x0"])
    reference = reference_rollout(
        x=x,
        steps=steps,
        dt=solver.dt,
        u_left=float(config["u_left"]),
        u_right=float(config["u_right"]),
        x0=x0,
        mobility_ratio=config.get("mobility_ratio", 0.5),
        reference_factor=config.get("reference_factor", 8),
        cfl=config.get("reference_cfl", 0.45),
    )
    F = solver.equilibrium(torch.tensor(reference[0], dtype=torch.float32, device=solver.device))

    rel_errors = []
    final_u = None
    for step in range(steps):
        u = solver.macro(F)
        u_ref = torch.tensor(reference[step], dtype=torch.float32, device=solver.device)
        rel = torch.norm(u - u_ref) / (torch.norm(u_ref) + 1e-7)
        rel_errors.append(float(rel))
        F, _, _ = solver.step(F)
        final_u = u

    final_ref = torch.tensor(reference[steps - 1], dtype=torch.float32, device=solver.device)
    final_rel = torch.norm(final_u - final_ref) / (torch.norm(final_ref) + 1e-7)
    final_time = float(steps - 1) * solver.dt

    out_dir = resolve_module_path("images")
    os.makedirs(out_dir, exist_ok=True)
    omega_suffix = f"_omega{str(solver.omega).replace('.', 'p')}"
    time_suffix = f"_t{final_time:.4f}".replace(".", "p")
    plot_path = os.path.join(out_dir, f"{solver.lattice.lower()}{omega_suffix}{time_suffix}_lbm_vs_reference.png")

    plt.figure(figsize=(10, 4))
    plt.plot(x, final_ref.cpu().numpy(), label="reference", linewidth=2)
    plt.plot(x, final_u.cpu().numpy(), label=f"{solver.lattice} lbm", linewidth=2)
    plt.title(f"Buckley-Leverett {solver.lattice} omega={solver.omega:g} vs reference at t={final_time:.4f}")
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
