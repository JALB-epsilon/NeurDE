import argparse
import os

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import torch
import yaml

from eval_rollout_comparison import load_model, parse_model_arg, relative_l2_error, run_rollout
from exact_solution import build_exact_macro_rollout
from SOD_solver_batch import SODBatchSolver
from train_stage_1 import create_basis
from utilities import get_device, set_seed


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
EXACT_STYLE = {
    "color": "black",
    "linestyle": "--",
    "linewidth": 2.4,
    "label": "Exact",
    "zorder": 12,
}
DATASET_STYLE = {
    "color": "#c2185b",
    "linestyle": "-.",
    "linewidth": 2.2,
    "marker": "o",
    "markersize": 3.0,
    "markevery": 260,
    "label": "Dataset",
    "zorder": 11,
}


def plot_profiles_with_dual_references(
    all_records,
    model_labels,
    dataset_ref,
    exact_ref,
    x_coords,
    snapshot_steps,
    output_path,
    title,
):
    quantities = [
        ("rho", "Density"),
        ("ux", "Velocity X"),
        ("T", "Temperature"),
        ("P", "Pressure"),
    ]
    fig, axes = plt.subplots(
        len(quantities),
        len(snapshot_steps),
        figsize=(4.2 * len(snapshot_steps), 3.1 * len(quantities)),
        squeeze=False,
    )

    for col_index, step in enumerate(snapshot_steps):
        ref_index = step - 1
        for row_index, (key, label) in enumerate(quantities):
            axis = axes[row_index][col_index]
            exact_values = exact_ref[key][ref_index].numpy()
            dataset_values = dataset_ref[key][ref_index].numpy()

            # Shade the small mismatch region so exact-vs-dataset remains visible
            # even when the two references nearly overlap.
            axis.fill_between(
                x_coords,
                exact_values,
                dataset_values,
                color="#f8bbd0",
                alpha=0.28,
                zorder=2,
            )
            axis.plot(
                x_coords,
                exact_values,
                **EXACT_STYLE,
                path_effects=[pe.Stroke(linewidth=3.6, foreground="white"), pe.Normal()],
            )
            axis.plot(
                x_coords,
                dataset_values,
                **DATASET_STYLE,
                path_effects=[pe.Stroke(linewidth=3.4, foreground="white"), pe.Normal()],
            )
            for model_index, (records, model_label) in enumerate(zip(all_records, model_labels)):
                axis.plot(
                    x_coords,
                    records[key][ref_index].numpy(),
                    color=COLORS[model_index % len(COLORS)],
                    linewidth=1.8,
                    label=model_label,
                    zorder=8 - model_index,
                )
            axis.set_title(f"{label} @ step {step}", fontsize=9)
            axis.grid(alpha=0.25)
            if row_index == len(quantities) - 1:
                axis.set_xlabel("x", fontsize=8)
            if col_index == 0:
                axis.set_ylabel(label, fontsize=8)
            axis.tick_params(labelsize=7)
            if row_index == 0 and col_index == len(snapshot_steps) - 1:
                axis.legend(fontsize=7, loc="best")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def print_error_summary(all_records, model_labels, dataset_ref, exact_ref):
    print("Final relative L2 errors at the last plotted step:")
    for records, model_label in zip(all_records, model_labels):
        dataset_rho = relative_l2_error(records["rho"], dataset_ref["rho"])[-1]
        dataset_ux = relative_l2_error(records["ux"], dataset_ref["ux"])[-1]
        dataset_t = relative_l2_error(records["T"], dataset_ref["T"])[-1]
        exact_rho = relative_l2_error(records["rho"], exact_ref["rho"])[-1]
        exact_ux = relative_l2_error(records["ux"], exact_ref["ux"])[-1]
        exact_t = relative_l2_error(records["T"], exact_ref["T"])[-1]
        print(
            f"- {model_label}: "
            f"dataset [rho={dataset_rho:.4f}, ux={dataset_ux:.4f}, T={dataset_t:.4f}] | "
            f"exact [rho={exact_rho:.4f}, ux={exact_ux:.4f}, T={exact_t:.4f}]"
        )


def parse_snapshot_steps(raw_value):
    return sorted({int(part.strip()) for part in raw_value.split(",") if part.strip()})


def main():
    set_seed(0)

    parser = argparse.ArgumentParser(
        description="Plot auto-regressive rollout profiles with both dataset and exact references."
    )
    parser.add_argument("--model", type=parse_model_arg, action="append", required=True)
    parser.add_argument("--case", type=int, choices=[1, 2], default=2)
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--start_index", type=int, default=500)
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--snapshot_steps", type=str, default="1,2,5,10,15,20")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    device = get_device(args.device)
    snapshot_steps = parse_snapshot_steps(args.snapshot_steps)
    if any(step < 1 or step > args.num_steps for step in snapshot_steps):
        raise ValueError(f"snapshot steps must lie in [1, {args.num_steps}]")

    with open(os.path.join(SCRIPT_DIR, "Sod_cases_param.yml"), "r", encoding="utf-8") as stream:
        case_params = yaml.safe_load(stream)[args.case]
    case_params["device"] = device

    with open(os.path.join(SCRIPT_DIR, "Sod_cases_param_training.yml"), "r", encoding="utf-8") as stream:
        param_training = yaml.safe_load(stream)[args.case]

    sod_solver = SODBatchSolver(
        X=case_params["X"],
        Y=case_params["Y"],
        Qn=case_params["Qn"],
        alpha1=case_params["alpha1"],
        alpha01=case_params["alpha01"],
        vuy=case_params["vuy"],
        Pr=case_params["Pr"],
        muy=case_params["muy"],
        Uax=case_params["Uax"],
        Uay=case_params["Uay"],
        device=device,
    )
    basis = create_basis(case_params["Uax"], case_params["Uay"], device)

    data_path = os.path.join(SCRIPT_DIR, param_training["data_dir"])
    with h5py.File(data_path, "r") as file_handle:
        total_steps = int(file_handle["Fi0"].shape[0])
        if args.start_index + args.num_steps >= total_steps:
            raise ValueError(
                f"Requested start_index={args.start_index}, num_steps={args.num_steps}, "
                f"but dataset only supports up to {total_steps - args.start_index - 1} future steps."
            )
        fi0 = torch.as_tensor(
            file_handle["Fi0"][args.start_index : args.start_index + 1],
            dtype=torch.float32,
            device=device,
        )
        gi0 = torch.as_tensor(
            file_handle["Gi0"][args.start_index : args.start_index + 1],
            dtype=torch.float32,
            device=device,
        )
        center_y = case_params["Y"] // 2
        dataset_rho = torch.as_tensor(
            file_handle["rho"][args.start_index + 1 : args.start_index + args.num_steps + 1, center_y],
            dtype=torch.float32,
        )
        dataset_ux = torch.as_tensor(
            file_handle["ux"][args.start_index + 1 : args.start_index + args.num_steps + 1, center_y],
            dtype=torch.float32,
        )
        dataset_t = torch.as_tensor(
            file_handle["T"][args.start_index + 1 : args.start_index + args.num_steps + 1, center_y],
            dtype=torch.float32,
        )
    dataset_ref = {
        "rho": dataset_rho,
        "ux": dataset_ux,
        "T": dataset_t,
        "P": dataset_rho * dataset_t,
    }

    exact_steps_needed = args.start_index + args.num_steps + 2
    exact_rho, exact_ux, exact_uy, exact_t = build_exact_macro_rollout(
        args.case, case_params["X"], case_params["Y"], exact_steps_needed, device
    )
    center_y = case_params["Y"] // 2
    exact_ref = {
        "rho": exact_rho[args.start_index + 1 : args.start_index + args.num_steps + 1, center_y].cpu(),
        "ux": exact_ux[args.start_index + 1 : args.start_index + args.num_steps + 1, center_y].cpu(),
        "T": exact_t[args.start_index + 1 : args.start_index + args.num_steps + 1, center_y].cpu(),
        "P": (exact_rho * exact_t)[args.start_index + 1 : args.start_index + args.num_steps + 1, center_y].cpu(),
    }

    all_records = []
    model_labels = []
    for checkpoint_path, arch, geq_mode, label, logit_clip in args.model:
        model = load_model(
            checkpoint_path, arch, geq_mode, param_training, case_params, device, logit_clip=logit_clip
        )
        records = run_rollout(
            model, arch, geq_mode, sod_solver, basis, fi0, gi0, args.num_steps, device
        )
        all_records.append(records)
        model_labels.append(label)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    x_coords = np.arange(case_params["X"])
    title = (
        "Spatial profiles with dual references\n"
        f"SOD Case {args.case} | start_index={args.start_index} | "
        "Exact + Dataset + model rollouts"
    )
    plot_profiles_with_dual_references(
        all_records=all_records,
        model_labels=model_labels,
        dataset_ref=dataset_ref,
        exact_ref=exact_ref,
        x_coords=x_coords,
        snapshot_steps=snapshot_steps,
        output_path=args.output,
        title=title,
    )
    print_error_summary(all_records, model_labels, dataset_ref, exact_ref)
    print(f"Saved dual-reference plot to: {args.output}")


if __name__ == "__main__":
    main()
