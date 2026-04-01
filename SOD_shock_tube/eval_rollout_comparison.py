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
import matplotlib.patheffects as patheffects
import numpy as np
import torch
import yaml

from architectures.model import NeurDE
from architectures.model_v3 import NeurDE_v3
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
    clip = 15.0 if logit_clip is None else float(logit_clip)
    if arch == "constrained":
        return NeurDEConstrainedGeq(
            alpha_layer=alpha_layer, phi_layer=phi_layer,
            activation="relu", cv=cv,
        ).to(device)
    if arch == "v3":
        fine_dim = 48
        nlayers = param_training["num_layers"]
        return NeurDE_v3(
            alpha_layer=alpha_layer, phi_layer=phi_layer,
            activation="relu", geq_mode=geq_mode, cv=cv,
            logit_clip=clip,
            fine_alpha_layer=[6] + [fine_dim] * nlayers,
            fine_phi_layer=[2] + [fine_dim] * nlayers,
        ).to(device)
    return NeurDE(
        alpha_layer=alpha_layer, phi_layer=phi_layer,
        activation="relu", geq_mode=geq_mode, cv=cv,
        logit_clip=clip,
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

def format_mu(muy):
    return f"{float(muy):.1e}"


def _format_float_like(value):
    return f"{float(value):.3g}"


def get_reference_style(ref_label):
    if str(ref_label).lower() == "dataset":
        return {
            "color": "#d81b60",
            "linestyle": "-.",
            "linewidth": 2.4,
            "marker": "o",
            "marker_size": 3.8,
            "marker_facecolor": "white",
            "marker_edgewidth": 1.0,
            "zorder": 20,
            "outline_width": 4.2,
        }
    return {
        "color": "black",
        "linestyle": "--",
        "linewidth": 2.0,
        "marker": None,
        "marker_size": None,
        "marker_facecolor": None,
        "marker_edgewidth": None,
        "zorder": 15,
        "outline_width": 3.6,
    }


def build_plot_context_lines(case, muy=None, ref_label="Reference", start_index=None,
                             dataset_delta_lines=None, basis_text=None):
    lines = [f"SOD Case {case}: {ref_label} target"]
    details = []
    if muy is not None:
        details.append(f"mu={format_mu(muy)}")
    if start_index is not None:
        details.append(f"start_index={start_index}")
    if details:
        lines.append(" | ".join(details))
    if dataset_delta_lines:
        lines.extend(dataset_delta_lines)
    if basis_text:
        lines.append(basis_text)
    return lines


def build_train_to_eval_lines(case, dataset_attrs):
    if not dataset_attrs:
        return None

    if case == 2:
        base_values = {
            "rho_left": 1.0,
            "rho_right": 0.125,
            "pressure_left": 0.2,
            "pressure_right": 0.02,
        }
        eval_values = {
            "rho_left": base_values["rho_left"] * float(dataset_attrs.get("rho_left_factor", 1.0)),
            "rho_right": base_values["rho_right"] * float(dataset_attrs.get("rho_right_factor", 1.0)),
            "pressure_left": base_values["pressure_left"] * float(dataset_attrs.get("pressure_left_factor", 1.0)),
            "pressure_right": base_values["pressure_right"] * float(dataset_attrs.get("pressure_right_factor", 1.0)),
        }
        line1 = (
            "Train -> Eval: "
            f"rho_L {_format_float_like(base_values['rho_left'])} -> {_format_float_like(eval_values['rho_left'])} | "
            f"rho_R {_format_float_like(base_values['rho_right'])} -> {_format_float_like(eval_values['rho_right'])}"
        )
        line2 = (
            "Train -> Eval: "
            f"p_L {_format_float_like(base_values['pressure_left'])} -> {_format_float_like(eval_values['pressure_left'])} | "
            f"p_R {_format_float_like(base_values['pressure_right'])} -> {_format_float_like(eval_values['pressure_right'])}"
        )
        extras = []
        if "muy" in dataset_attrs:
            extras.append(f"mu {format_mu(1.0e-4)} -> {format_mu(dataset_attrs['muy'])}")
        if "Uax" in dataset_attrs:
            extras.append(f"Uax 0.4 -> {_format_float_like(dataset_attrs['Uax'])}")
        if "Uay" in dataset_attrs:
            extras.append(f"Uay 0 -> {_format_float_like(dataset_attrs['Uay'])}")
        lines = [line1, line2]
        if extras:
            lines.append("Train -> Eval: " + " | ".join(extras))
        return lines

    return None


def plot_spatial_profiles(all_records, ref_records, x_coords, snapshot_steps,
                           model_labels, output_path, case, ref_label="Reference", muy=None,
                           context_lines=None):
    quantities = [("rho", "Density ρ"), ("ux", "Velocity ux"), ("T", "Temperature T"), ("P", "Pressure P")]
    n_snap = len(snapshot_steps)
    ref_style = get_reference_style(ref_label)
    marker_every = max(1, len(x_coords) // 24)
    fig, axes = plt.subplots(
        len(quantities), n_snap,
        figsize=(4.0 * n_snap, 3.2 * len(quantities)),
        squeeze=False,
    )
    for col, step in enumerate(snapshot_steps):
        idx = step - 1  # 0-indexed
        for row, (key, label) in enumerate(quantities):
            ax = axes[row][col]
            ref_line, = ax.plot(
                x_coords,
                ref_records[key][idx].numpy(),
                color=ref_style["color"],
                linestyle=ref_style["linestyle"],
                linewidth=ref_style["linewidth"],
                marker=ref_style["marker"],
                markersize=ref_style["marker_size"],
                markerfacecolor=ref_style["marker_facecolor"],
                markeredgewidth=ref_style["marker_edgewidth"],
                markevery=marker_every if ref_style["marker"] is not None else None,
                label=ref_label,
                zorder=ref_style["zorder"],
            )
            ref_line.set_path_effects(
                [
                    patheffects.Stroke(
                        linewidth=ref_style["outline_width"],
                        foreground="white",
                    ),
                    patheffects.Normal(),
                ]
            )
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

    title_lines = context_lines or build_plot_context_lines(case, muy=muy, ref_label=ref_label)
    fig.suptitle(
        "Spatial profiles (auto-regressive rollout)\n" + "\n".join(title_lines),
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_error_curves(all_records, ref_records, model_labels, num_steps, output_path, case,
                      ref_label="Reference", muy=None, context_lines=None):
    quantities = [("rho", "Density ρ"), ("ux", "Velocity ux"), ("T", "Temperature T")]
    fig, axes = plt.subplots(1, len(quantities), figsize=(6.5 * len(quantities), 5), squeeze=False)
    steps = np.arange(1, num_steps + 1)
    for col, (key, label) in enumerate(quantities):
        ax = axes[0][col]
        for m_idx, (records, mlabel) in enumerate(zip(all_records, model_labels)):
            err = relative_l2_error(records[key], ref_records[key])
            c = COLORS[m_idx % len(COLORS)]
            ax.semilogy(steps, err, color=c, linewidth=1.8, label=mlabel)
        ax.set_xlabel("Rollout step", fontsize=10)
        ax.set_ylabel(f"Relative L2 error ({label})", fontsize=10)
        ax.set_title(f"{label} error vs {ref_label.lower()}", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    title_lines = context_lines or build_plot_context_lines(case, muy=muy, ref_label=ref_label)
    fig.suptitle(
        f"Error over {num_steps}-step rollout\n" + "\n".join(title_lines),
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_energy_conservation(all_records, model_labels, num_steps, output_path, case, muy=None,
                             context_lines=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    steps = np.arange(1, num_steps + 1)
    for m_idx, (records, mlabel) in enumerate(zip(all_records, model_labels)):
        c = COLORS[m_idx % len(COLORS)]
        ax.semilogy(steps, records["energy_err"] + 1e-16, color=c, linewidth=1.8, label=mlabel)
    ax.set_xlabel("Rollout step", fontsize=11)
    ax.set_ylabel("Max relative energy conservation error", fontsize=11)
    title_lines = context_lines or build_plot_context_lines(case, muy=muy)
    ax.set_title(
        "Energy conservation over rollout\n"
        + "\n".join(title_lines)
        + "\n"
        + r"$\max_x |{\sum_q G_q - 2\rho E}| \;/\; |2\rho E|$",
        fontsize=12,
    )
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
    if arch not in {"neurde", "v3", "constrained", "levermore"}:
        raise argparse.ArgumentTypeError(f"arch must be neurde, v3, constrained, or levermore; got {arch!r}")
    return path, arch, geq_mode, label, logit_clip


def parse_snapshot_steps_arg(s):
    steps = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            step = int(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"--snapshot_steps entries must be integers, got {part!r}"
            ) from exc
        if step < 1:
            raise argparse.ArgumentTypeError(
                f"--snapshot_steps entries must be >= 1, got {step}"
            )
        steps.append(step)
    if not steps:
        raise argparse.ArgumentTypeError("--snapshot_steps cannot be empty")
    return steps


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
                        help="Number of auto-regressive steps. In dataset target mode, -1 uses the full available horizon.")
    parser.add_argument("--n_snapshots", type=int, default=5,
                        help="Number of uniformly spaced spatial-profile snapshots")
    parser.add_argument("--snapshot_steps", type=parse_snapshot_steps_arg, default=None,
                        help="Optional comma-separated explicit snapshot steps, e.g. 5,10,20,40,80")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(SCRIPT_DIR, "images"))
    parser.add_argument("--data_path", type=str, default=None,
                        help="Optional alternate dataset .h5 path for OOD evaluation")
    parser.add_argument("--muy_override", type=float, default=None,
                        help="Optional viscosity override for the rollout solver")
    parser.add_argument("--basis_uax_override", type=float, default=None,
                        help="Optional override for the NeurDE basis x-shift")
    parser.add_argument("--basis_uay_override", type=float, default=None,
                        help="Optional override for the NeurDE basis y-shift")
    parser.add_argument("--target_mode", choices=["exact", "dataset"], default="exact",
                        help="Reference for error curves and profiles")
    args = parser.parse_args()

    device = get_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(SCRIPT_DIR, "Sod_cases_param.yml")) as f:
        case_params = yaml.safe_load(f)[args.case]
    case_params["device"] = device
    if args.muy_override is not None:
        case_params["muy"] = float(args.muy_override)

    with open(os.path.join(SCRIPT_DIR, "Sod_cases_param_training.yml")) as f:
        param_training = yaml.safe_load(f)[args.case]

    sod_solver = SODBatchSolver(
        X=case_params["X"], Y=case_params["Y"], Qn=case_params["Qn"],
        alpha1=case_params["alpha1"], alpha01=case_params["alpha01"],
        vuy=case_params["vuy"], Pr=case_params["Pr"], muy=case_params["muy"],
        Uax=case_params["Uax"], Uay=case_params["Uay"], device=device,
    )
    basis_uax = case_params["Uax"] if args.basis_uax_override is None else float(args.basis_uax_override)
    basis_uay = case_params["Uay"] if args.basis_uay_override is None else float(args.basis_uay_override)
    basis = create_basis(basis_uax, basis_uay, device)
    print(
        f"Solver shift: Uax={case_params['Uax']}, Uay={case_params['Uay']} | "
        f"Basis shift: Uax={basis_uax}, Uay={basis_uay}"
    )

    # Load initial state from dataset; dataset-mode also uses stored macro rollout as reference
    data_path = args.data_path if args.data_path is not None else os.path.join(SCRIPT_DIR, param_training["data_dir"])
    dataset_attrs = {}
    with h5py.File(data_path, "r") as f:
        dataset_attrs = {key: f.attrs[key] for key in f.attrs.keys()}
        total_steps = int(f["Fi0"].shape[0])
        if args.target_mode == "dataset":
            max_rollout_steps = total_steps - args.start_index - 1
            if max_rollout_steps <= 0:
                raise ValueError(
                    f"Dataset target mode requires start_index < {total_steps - 1}, got {args.start_index}"
                )
            if args.num_steps < 0:
                print(
                    f"Dataset target mode: using full available horizon of {max_rollout_steps} steps "
                    f"from start_index={args.start_index}."
                )
                args.num_steps = max_rollout_steps
            elif args.num_steps > max_rollout_steps:
                print(
                    f"Dataset target mode: clamping num_steps from {args.num_steps} to {max_rollout_steps} "
                    f"because only that many future steps are available from start_index={args.start_index}."
                )
                args.num_steps = max_rollout_steps
        fi0 = torch.as_tensor(f["Fi0"][args.start_index : args.start_index + 1], dtype=torch.float32, device=device)
        gi0 = torch.as_tensor(f["Gi0"][args.start_index : args.start_index + 1], dtype=torch.float32, device=device)
        cy = case_params["Y"] // 2
        if args.target_mode == "dataset":
            ref_rho = torch.as_tensor(
                f["rho"][args.start_index + 1 : args.start_index + args.num_steps + 1, cy],
                dtype=torch.float32,
            )
            ref_ux = torch.as_tensor(
                f["ux"][args.start_index + 1 : args.start_index + args.num_steps + 1, cy],
                dtype=torch.float32,
            )
            ref_T = torch.as_tensor(
                f["T"][args.start_index + 1 : args.start_index + args.num_steps + 1, cy],
                dtype=torch.float32,
            )
            ref_records = {
                "rho": ref_rho,
                "ux": ref_ux,
                "T": ref_T,
                "P": ref_rho * ref_T,
            }
            ref_label = "Dataset"

    if args.target_mode == "exact":
        exact_steps_needed = args.start_index + args.num_steps + 2
        print(f"Building exact Riemann solution for {exact_steps_needed} steps...")
        exact_rho, exact_ux, exact_uy, exact_T = build_exact_macro_rollout(
            args.case, case_params["X"], case_params["Y"], exact_steps_needed, device
        )
        cy = case_params["Y"] // 2
        ref_records = {
            "rho": exact_rho[args.start_index + 1 : args.start_index + args.num_steps + 1, cy].cpu(),
            "ux":  exact_ux [args.start_index + 1 : args.start_index + args.num_steps + 1, cy].cpu(),
            "T":   exact_T  [args.start_index + 1 : args.start_index + args.num_steps + 1, cy].cpu(),
            "P":   (exact_rho * exact_T)[args.start_index + 1 : args.start_index + args.num_steps + 1, cy].cpu(),
        }
        ref_label = "Exact"

    x_coords = np.arange(case_params["X"])
    dataset_delta_lines = build_train_to_eval_lines(args.case, dataset_attrs) if args.target_mode == "dataset" else None
    basis_text = None
    if args.basis_uax_override is not None or args.basis_uay_override is not None:
        basis_text = f"Basis override: Uax={_format_float_like(basis_uax)}, Uay={_format_float_like(basis_uay)}"
    context_lines = build_plot_context_lines(
        args.case,
        muy=case_params["muy"],
        ref_label=ref_label,
        start_index=args.start_index,
        dataset_delta_lines=dataset_delta_lines,
        basis_text=basis_text,
    )

    # Snapshot steps: uniformly spaced from 1 to num_steps inclusive
    if args.snapshot_steps is not None:
        snapshot_steps = sorted(set(args.snapshot_steps))
        invalid_steps = [step for step in snapshot_steps if step > args.num_steps]
        if invalid_steps:
            raise ValueError(
                f"--snapshot_steps entries must be <= num_steps ({args.num_steps}), "
                f"got {invalid_steps}"
            )
    else:
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
            err = relative_l2_error(records[key], ref_records[key])
            print(f"  {label} final {key} error: {err[-1]:.4f}  |  mean: {err.mean():.4f}  |  max: {err.max():.4f}")

    # --- Plots ---
    plot_spatial_profiles(
        all_records, ref_records, x_coords, snapshot_steps, model_labels,
        os.path.join(args.output_dir, f"case{args.case}_profiles.png"),
        args.case, ref_label=ref_label, muy=case_params["muy"], context_lines=context_lines,
    )
    plot_error_curves(
        all_records, ref_records, model_labels, args.num_steps,
        os.path.join(args.output_dir, f"case{args.case}_errors.png"),
        args.case, ref_label=ref_label, muy=case_params["muy"], context_lines=context_lines,
    )
    plot_energy_conservation(
        all_records, model_labels, args.num_steps,
        os.path.join(args.output_dir, f"case{args.case}_energy_conservation.png"),
        args.case, muy=case_params["muy"], context_lines=context_lines,
    )
    print("Done.")
