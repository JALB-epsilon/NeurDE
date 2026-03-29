"""
eval_rollout_comparison.py

Multi-model auto-regressive rollout evaluation for SOD shock tube stage-2.
Starts from dataset step --start_index, runs --num_steps steps auto-regressively,
and compares each model to the exact Riemann solution.

Each --model argument: <checkpoint_path>:<arch>:<geq_mode>:<label>[:<logit_clip>]
  arch     = neurde | constrained
  geq_mode = positive | hybrid | hybrid_energy | energy_projected | none

Example:
  python eval_rollout_comparison.py \\
    --model results/positive/best.pt:neurde:positive:Positive \\
    --model results/hybrid_energy/best.pt:neurde:hybrid_energy:HybridEnergy \\
    --model results/constrained/best.pt:constrained:none:Constrained8D \\
    --start_index 500 --num_steps 500 --case 2 --output_dir results/plots/rollout
"""

import argparse
import os

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from architectures.model import NeurDE
from architectures.model_constrained_geq import NeurDEConstrainedGeq
from exact_solution import build_exact_macro_rollout
from SOD_solver_batch import SODBatchSolver
from train_stage_1 import create_basis
from utilities import get_device, set_seed


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def _strip_orig_mod(state_dict):
    new = {}
    for k, v in state_dict.items():
        new[k.replace("_orig_mod.", "")] = v
    return new


def build_model(arch, geq_mode, param_training, case_params, device, logit_clip=None):
    alpha_layer = [4] + [param_training["hidden_dim"]] * param_training["num_layers"]
    phi_layer   = [2] + [param_training["hidden_dim"]] * param_training["num_layers"]
    cv = 1.0 / (case_params["vuy"] - 1.0)
    if arch == "constrained":
        return NeurDEConstrainedGeq(
            alpha_layer=alpha_layer, phi_layer=phi_layer,
            activation="relu", cv=cv,
        ).to(device)
    return NeurDE(
        alpha_layer=alpha_layer, phi_layer=phi_layer,
        activation="relu", geq_mode=geq_mode, cv=cv,
        logit_clip=(15.0 if logit_clip is None else float(logit_clip)),
    ).to(device)


def load_model(path, arch, geq_mode, param_training, case_params, device, logit_clip=None):
    if arch == "levermore":
        return None  # no NN; Newton solver is used directly in predict_geq_step
    model = build_model(arch, geq_mode, param_training, case_params, device, logit_clip=logit_clip)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(_strip_orig_mod(ckpt))
    model.eval()
    return model


def predict_geq_step(model, arch, geq_mode, sod_solver, basis, rho, ux, uy, T, khi, zetax, zetay):
    """Returns (geq [B,Qn,Y,X], khi, zetax, zetay).
    arch='levermore' uses pure Newton solver with no NN correction."""
    if arch == "levermore":
        geq, khi, zetax, zetay = sod_solver.get_Geq_Newton_solver_batch(
            rho, ux, uy, T, khi, zetax, zetay
        )
        return geq, khi, zetax, zetay
    inputs = torch.stack([rho, ux, uy, T], dim=1)
    if arch == "constrained" or geq_mode in {"hybrid", "hybrid_energy"}:
        geq_base, khi, zetax, zetay = sod_solver.get_Geq_Newton_solver_batch(
            rho, ux, uy, T, khi, zetax, zetay
        )
        geq_flat = model(inputs, basis, geq_base=geq_base)
    else:
        geq_flat = model(inputs, basis)
    B, Qn, Y, X = rho.shape[0], sod_solver.Qn, sod_solver.Y, sod_solver.X
    geq = geq_flat.reshape(B, Y, X, Qn).permute(0, 3, 1, 2).contiguous()
    return geq, khi, zetax, zetay


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_rollout(model, arch, geq_mode, sod_solver, basis, fi0, gi0, num_steps, device):
    """Returns dicts of lists: rho, ux, T, P, energy_err at each step."""
    fi = fi0.clone()
    gi = gi0.clone()
    khi = zetax = zetay = None
    cv = sod_solver.Cv

    records = {"rho": [], "ux": [], "T": [], "P": [], "energy_err": []}

    for _ in range(num_steps):
        rho, ux, uy, E = sod_solver.get_macroscopic(fi, gi)
        T = sod_solver.get_temp_from_energy(ux, uy, E)
        feq = sod_solver.get_Feq(rho, ux, uy, T)
        geq, khi, zetax, zetay = predict_geq_step(
            model, arch, geq_mode, sod_solver, basis, rho, ux, uy, T, khi, zetax, zetay
        )
        fi, gi = sod_solver.collision(fi, gi, feq, geq, rho, ux, uy, T)
        fi, gi = sod_solver.streaming(fi, gi)

        rho_n, ux_n, uy_n, E_n = sod_solver.get_macroscopic(fi, gi)
        T_n = sod_solver.get_temp_from_energy(ux_n, uy_n, E_n)
        P_n = rho_n * T_n

        # Energy conservation: |sum(G) - 2*rho*E| / |2*rho*E|, max over domain
        expected_energy = 2.0 * rho_n * (cv * T_n + 0.5 * (ux_n ** 2 + uy_n ** 2))
        actual_energy = gi.sum(dim=1)
        eps = 1.0e-10
        energy_err = ((actual_energy - expected_energy).abs() / (expected_energy.abs() + eps)).amax().item()

        cy = sod_solver.Y // 2
        records["rho"].append(rho_n[0, cy].cpu())
        records["ux"].append(ux_n[0, cy].cpu())
        records["T"].append(T_n[0, cy].cpu())
        records["P"].append(P_n[0, cy].cpu())
        records["energy_err"].append(energy_err)

    for key in ("rho", "ux", "T", "P"):
        records[key] = torch.stack(records[key], dim=0)  # [num_steps, X]
    records["energy_err"] = np.array(records["energy_err"])
    return records


def relative_l2_error(pred, ref):
    """pred, ref: [num_steps, X]. Returns [num_steps]."""
    eps = 1.0e-8
    diff = (pred - ref).norm(dim=1)
    denom = ref.norm(dim=1) + eps
    return (diff / denom).numpy()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_spatial_profiles(all_records, exact_records, x_coords, snapshot_steps,
                           model_labels, output_path, case):
    quantities = [("rho", "Density ρ"), ("ux", "Velocity ux"), ("T", "Temperature T"), ("P", "Pressure P")]
    n_snap = len(snapshot_steps)
    fig, axes = plt.subplots(
        len(quantities), n_snap,
        figsize=(4.0 * n_snap, 3.2 * len(quantities)),
        squeeze=False,
    )
    for col, step in enumerate(snapshot_steps):
        idx = step - 1  # 0-indexed
        for row, (key, label) in enumerate(quantities):
            ax = axes[row][col]
            ax.plot(x_coords, exact_records[key][idx].numpy(), "k--", linewidth=2.0, label="Exact", zorder=10)
            for m_idx, (records, mlabel) in enumerate(zip(all_records, model_labels)):
                c = COLORS[m_idx % len(COLORS)]
                ax.plot(x_coords, records[key][idx].numpy(), color=c, linewidth=1.5, label=mlabel)
            ax.set_title(f"{label} @ step {step}", fontsize=9)
            ax.grid(alpha=0.25)
            if row == len(quantities) - 1:
                ax.set_xlabel("x", fontsize=8)
            if col == 0:
                ax.set_ylabel(label, fontsize=8)
            ax.tick_params(labelsize=7)
            if row == 0 and col == n_snap - 1:
                ax.legend(fontsize=7, loc="best")

    fig.suptitle(f"SOD Case {case}: Spatial profiles (auto-regressive rollout)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_error_curves(all_records, exact_records, model_labels, num_steps, output_path, case):
    quantities = [("rho", "Density ρ"), ("ux", "Velocity ux"), ("T", "Temperature T")]
    fig, axes = plt.subplots(1, len(quantities), figsize=(6.5 * len(quantities), 5), squeeze=False)
    steps = np.arange(1, num_steps + 1)
    for col, (key, label) in enumerate(quantities):
        ax = axes[0][col]
        for m_idx, (records, mlabel) in enumerate(zip(all_records, model_labels)):
            err = relative_l2_error(records[key], exact_records[key])
            c = COLORS[m_idx % len(COLORS)]
            ax.semilogy(steps, err, color=c, linewidth=1.8, label=mlabel)
        ax.set_xlabel("Rollout step", fontsize=10)
        ax.set_ylabel(f"Relative L2 error ({label})", fontsize=10)
        ax.set_title(f"{label} error vs exact", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle(f"SOD Case {case}: Error over 500-step rollout", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_energy_conservation(all_records, model_labels, num_steps, output_path, case):
    fig, ax = plt.subplots(figsize=(9, 5))
    steps = np.arange(1, num_steps + 1)
    for m_idx, (records, mlabel) in enumerate(zip(all_records, model_labels)):
        c = COLORS[m_idx % len(COLORS)]
        ax.semilogy(steps, records["energy_err"] + 1e-16, color=c, linewidth=1.8, label=mlabel)
    ax.set_xlabel("Rollout step", fontsize=11)
    ax.set_ylabel("Max relative energy conservation error", fontsize=11)
    ax.set_title(f"SOD Case {case}: Energy conservation over rollout\n"
                 r"$\max_x |{\sum_q G_q - 2\rho E}| \;/\; |2\rho E|$", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_model_arg(s):
    """Parse 'path:arch:geq_mode:label[:logit_clip]'."""
    parts = s.split(":")
    if len(parts) not in {4, 5}:
        raise argparse.ArgumentTypeError(
            f"--model must be path:arch:geq_mode:label[:logit_clip], got: {s!r}"
        )
    path, arch, geq_mode, label = parts[:4]
    logit_clip = float(parts[4]) if len(parts) == 5 else None
    if arch not in {"neurde", "constrained", "levermore"}:
        raise argparse.ArgumentTypeError(f"arch must be neurde, constrained, or levermore; got {arch!r}")
    return path, arch, geq_mode, label, logit_clip


if __name__ == "__main__":
    set_seed(0)

    parser = argparse.ArgumentParser(description="Multi-model rollout comparison for SOD stage-2")
    parser.add_argument("--model", type=parse_model_arg, action="append", dest="models", required=True,
                        help="path:arch:geq_mode:label  (repeat for multiple models)")
    parser.add_argument("--case", type=int, choices=[1, 2], default=2)
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--start_index", type=int, default=500,
                        help="Dataset index to begin rollout from")
    parser.add_argument("--num_steps", type=int, default=500,
                        help="Number of auto-regressive steps")
    parser.add_argument("--n_snapshots", type=int, default=5,
                        help="Number of uniformly spaced spatial-profile snapshots")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(SCRIPT_DIR, "images"))
    args = parser.parse_args()

    device = get_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(SCRIPT_DIR, "Sod_cases_param.yml")) as f:
        case_params = yaml.safe_load(f)[args.case]
    case_params["device"] = device

    with open(os.path.join(SCRIPT_DIR, "Sod_cases_param_training.yml")) as f:
        param_training = yaml.safe_load(f)[args.case]

    sod_solver = SODBatchSolver(
        X=case_params["X"], Y=case_params["Y"], Qn=case_params["Qn"],
        alpha1=case_params["alpha1"], alpha01=case_params["alpha01"],
        vuy=case_params["vuy"], Pr=case_params["Pr"], muy=case_params["muy"],
        Uax=case_params["Uax"], Uay=case_params["Uay"], device=device,
    )
    basis = create_basis(case_params["Uax"], case_params["Uay"], device)

    # Load initial state from dataset
    data_path = os.path.join(SCRIPT_DIR, param_training["data_dir"])
    with h5py.File(data_path, "r") as f:
        fi0 = torch.as_tensor(f["Fi0"][args.start_index : args.start_index + 1], dtype=torch.float32, device=device)
        gi0 = torch.as_tensor(f["Gi0"][args.start_index : args.start_index + 1], dtype=torch.float32, device=device)

    # Build exact Riemann solution for the needed range
    exact_steps_needed = args.start_index + args.num_steps + 2
    print(f"Building exact Riemann solution for {exact_steps_needed} steps...")
    exact_rho, exact_ux, exact_uy, exact_T = build_exact_macro_rollout(
        args.case, case_params["X"], case_params["Y"], exact_steps_needed, device
    )
    cy = case_params["Y"] // 2
    exact_records = {
        "rho": exact_rho[args.start_index + 1 : args.start_index + args.num_steps + 1, cy].cpu(),
        "ux":  exact_ux [args.start_index + 1 : args.start_index + args.num_steps + 1, cy].cpu(),
        "T":   exact_T  [args.start_index + 1 : args.start_index + args.num_steps + 1, cy].cpu(),
        "P":   (exact_rho * exact_T)[args.start_index + 1 : args.start_index + args.num_steps + 1, cy].cpu(),
    }

    x_coords = np.arange(case_params["X"])

    # Snapshot steps: uniformly spaced from 1 to num_steps inclusive
    snapshot_steps = [
        max(1, int(round(args.num_steps * k / args.n_snapshots)))
        for k in range(1, args.n_snapshots + 1)
    ]

    # Run rollout for each model
    all_records = []
    model_labels = []
    for ckpt_path, arch, geq_mode, label, logit_clip in args.models:
        clip_desc = "" if logit_clip is None else f", clip={logit_clip:g}"
        print(f"Running rollout: {label}  [{arch}/{geq_mode}{clip_desc}]  {ckpt_path}")
        model = load_model(
            ckpt_path, arch, geq_mode, param_training, case_params, device, logit_clip=logit_clip
        )
        records = run_rollout(model, arch, geq_mode, sod_solver, basis,
                              fi0, gi0, args.num_steps, device)
        all_records.append(records)
        model_labels.append(label)

        # Per-model final errors
        for key in ("rho", "ux", "T"):
            err = relative_l2_error(records[key], exact_records[key])
            print(f"  {label} final {key} error: {err[-1]:.4f}  |  mean: {err.mean():.4f}  |  max: {err.max():.4f}")

    # --- Plots ---
    plot_spatial_profiles(
        all_records, exact_records, x_coords, snapshot_steps, model_labels,
        os.path.join(args.output_dir, f"case{args.case}_profiles.png"),
        args.case,
    )
    plot_error_curves(
        all_records, exact_records, model_labels, args.num_steps,
        os.path.join(args.output_dir, f"case{args.case}_errors.png"),
        args.case,
    )
    plot_energy_conservation(
        all_records, model_labels, args.num_steps,
        os.path.join(args.output_dir, f"case{args.case}_energy_conservation.png"),
        args.case,
    )
    print("Done.")
