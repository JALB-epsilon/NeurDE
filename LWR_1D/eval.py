import argparse

import h5py
import torch
import yaml

from architectures import NeurDE
from lwr_solver import LWRSolver, default_config_path, resolve_config_path, resolve_module_path, resolve_torch_dtype


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=default_config_path())
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model_path", type=str, default="results_d2q9/lwr_stage1.pt")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    args = parser.parse_args()
    dtype = resolve_torch_dtype(args.dtype)

    config_path = resolve_config_path(args.config)
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)
    data_path = resolve_module_path(config["data_dir"])
    model_path = resolve_module_path(args.model_path)
    conservative_output = config.get("conservative_output", config.get("match_mass", True))

    with h5py.File(data_path, "r") as handle:
        u_ref = torch.as_tensor(handle["u"][: args.steps], dtype=dtype)

    solver = LWRSolver(
        X=config["X"],
        lam=config["lam"],
        omega=config["omega"],
        lattice=config.get("lattice", "D2Q9"),
        device=args.device,
        newton_steps=config.get("newton_steps", 50),
        newton_tol=config.get("newton_tol", 1e-10),
        domain_length=config.get("domain_length", 1.0),
        boundary=config.get("boundary", "riemann"),
        rho_left_bc=config.get("rho_left"),
        rho_right_bc=config.get("rho_right"),
        rho_max=config.get("rho_max", 1.0),
        v_free=config.get("v_free", 1.0),
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
    model.load_state_dict(torch.load(model_path, map_location=args.device))
    model.eval()

    basis = solver.basis().to(device=args.device, dtype=dtype)
    F = solver.equilibrium(u_ref[0].to(args.device).unsqueeze(0))
    rel_error = 0.0

    with torch.no_grad():
        for step in range(args.steps):
            u = solver.macro(F)
            inputs = u.unsqueeze(1).unsqueeze(2)
            feq_pred = model(inputs, basis).reshape(1, solver.X, solver.Qn).permute(0, 2, 1)
            F, _, _ = solver.step(F, feq_pred)
            target = u_ref[step].to(args.device).unsqueeze(0)
            rel_error += (torch.norm(u - target) / (torch.norm(target) + 1e-7)).item()

    print(
        f"Average rollout relative error over {args.steps} LWR steps: "
        f"{rel_error / max(args.steps, 1):.6f}; conservative_output={conservative_output}"
    )


if __name__ == "__main__":
    main()
