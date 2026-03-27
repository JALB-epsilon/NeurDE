import argparse

import h5py
import torch
import yaml

from architectures import NeurDE
from buckley_leverett_solver import BuckleyLeverettSolver, default_config_path, resolve_config_path, resolve_module_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=default_config_path())
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model_path", type=str, default="results_d2q9/buckley_leverett_stage1.pt")
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)
    data_path = resolve_module_path(config["data_dir"])
    model_path = resolve_module_path(args.model_path)
    conservative_output = config.get("conservative_output", config.get("match_mass", True))

    with h5py.File(data_path, "r") as handle:
        u_ref = torch.tensor(handle["u"][: args.steps], dtype=torch.float32)

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
    model = NeurDE(
        alpha_layer=[1] + [config["hidden_dim"]] * config["num_layers"],
        phi_layer=[1] + [config["hidden_dim"]] * config["num_layers"],
        activation="relu",
        learn_feq=True,
        learn_geq=False,
        logit_clip=config.get("logit_clip", 15.0),
        conservative_output=conservative_output,
    ).to(args.device)
    model.load_state_dict(torch.load(model_path, map_location=args.device))
    model.eval()

    basis = solver.basis().to(args.device)
    F = solver.equilibrium(u_ref[0].to(args.device))
    rel_error = 0.0

    with torch.no_grad():
        for step in range(args.steps):
            u = solver.macro(F)
            inputs = u.unsqueeze(0).unsqueeze(0).unsqueeze(1)
            feq_pred = model(inputs, basis).permute(1, 0)
            F, _, _ = solver.step(F, feq_pred)
            target = u_ref[step].to(args.device)
            rel_error += (torch.norm(u - target) / (torch.norm(target) + 1e-7)).item()

    print(
        f"Average rollout relative error over {args.steps} Buckley-Leverett steps: "
        f"{rel_error / max(args.steps, 1):.6f}; conservative_output={conservative_output}"
    )


if __name__ == "__main__":
    main()
