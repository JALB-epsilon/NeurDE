import torch
from architectures import NeurDE, bounded_residual_population, project_conserved_moments
from utilities import *
import argparse
import yaml
import os
import h5py
from train_stage_1 import create_basis
from SOD_solver import SODSolver
from analytic_sod import analytic_reference_from_case
import numpy as np
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **kwargs):
        del kwargs
        return iterable


def build_effective_feq(
    *,
    learn_feq,
    feq_mode,
    feq_residual_scale,
    feq_collision_mix,
    project_feq_moments,
    feq_nullspace_gamma,
    inputs,
    basis,
    analytic_feq,
    feq_target,
    feq_pred_grid,
):
    if not learn_feq:
        return analytic_feq

    if feq_mode == "analytic_residual":
        effective_feq = bounded_residual_population(
            base_population=analytic_feq,
            predicted_population=feq_pred_grid,
            residual_scale=max(float(feq_residual_scale), 0.0) * max(min(float(feq_collision_mix), 1.0), 0.0),
        )
        if project_feq_moments:
            flat_macro_state = inputs.movedim(1, -1).reshape(-1, inputs.shape[1])
            effective_feq = reshape_equilibrium_prediction(
                project_conserved_moments(
                    reshape_equilibrium_target(effective_feq),
                    flat_macro_state,
                    basis,
                    nullspace_gamma=feq_nullspace_gamma,
                ),
                effective_feq.shape,
            )
        return effective_feq

    effective_feq = feq_pred_grid
    if feq_collision_mix < 1.0:
        effective_feq = feq_collision_mix * feq_pred_grid + (1.0 - feq_collision_mix) * feq_target
    return effective_feq


def apply_dataset_case_overrides(case_params, data_path):
    overridden = dict(case_params)
    scalar_keys = ("X", "Y", "Qn", "alpha1", "alpha01", "vuy", "Pr", "muy", "Uax", "Uay")
    int_keys = {"X", "Y", "Qn"}
    with h5py.File(data_path, "r") as handle:
        for key in scalar_keys:
            if key not in handle.attrs:
                continue
            raw_value = handle.attrs[key]
            overridden[key] = int(raw_value) if key in int_keys else float(raw_value)
    return overridden

if __name__ == "__main__":
    set_seed(0)

    parser = argparse.ArgumentParser(description='evaluation')
    parser.add_argument('--device', type=int, default=3, help='Device index')
    parser.add_argument("--compile", dest='compile', action='store_true', help='Compile', default=False)
    parser.add_argument('--save_model', action='store_true', help='Save model checkpoints (enabled by default)')
    parser.add_argument('--no-save_model', dest='save_model', action='store_false', help='Disable model checkpoint saving')
    parser.add_argument('--num_samples', type=int, default=500, help='Number of samples')
    parser.add_argument("--init_cond",  type=int, default=500, help='Number of samples')
    parser.add_argument("--save_frequency", default=50, help='Save model')
    parser.add_argument("--trained_path", type=str, default=None)
    parser.add_argument("--no_analytic_reference", dest="analytic_reference", action="store_false", default=True)
    parser.add_argument("--plot_every", type=int, default=50, help='Plot every N evaluation steps')
    parser.add_argument("--reference", type=str, choices=["solver", "analytic"], default="analytic", help='Reference trajectory for reported rollout errors')
    parser.add_argument("--feq_collision_mix", type=float, default=None, help='Override Feq collision mix used at evaluation time')
    parser.add_argument("--data_path", type=str, default=None, help='Optional dataset override for OOD evaluation')
    parser.add_argument("--training_config", type=str, default="Sod_cases_param_training.yml", help='Training YAML path')
    parser.set_defaults(save_model=True)
    args = parser.parse_args()

    device = get_device(args.device)

    if not args.trained_path:
        raise ValueError("--trained_path is required for evaluation.")

    args.trained_path = args.trained_path.replace("SOD_shock_tube/", "")
    print(args.trained_path)
    checkpoint_tag = os.path.splitext(os.path.basename(args.trained_path))[0]
    case_part = args.trained_path.split('/')[1]
    case_number = ''.join(filter(str.isdigit, case_part))
    args.case = int(case_number)
    print(args.case)

    with open("Sod_cases_param.yml", 'r') as stream:
        config = yaml.safe_load(stream)
    case_params = config[args.case]

    with open(args.training_config, 'r') as stream:
        training_config = yaml.safe_load(stream)
    param_training = training_config[args.case]
    learn_feq = resolve_learn_feq(param_training, "stage2", default=False)
    number_of_rollout = param_training["stage2"]["N"]
    data_path = args.data_path or param_training["data_dir"]
    case_params = apply_dataset_case_overrides(case_params, data_path)
    case_params['device'] = device

    print(f"Case {args.case}: SOD shock tube problem")
    print(f"Evaluation data: {data_path}")

    sod_solver = SODSolver(
        X=case_params['X'],
        Y=case_params['Y'],
        Qn=case_params['Qn'],
        alpha1=case_params['alpha1'],
        alpha01=case_params['alpha01'],
        vuy=case_params['vuy'],
        Pr=case_params['Pr'],
        muy=case_params['muy'],
        Uax=case_params['Uax'],
        Uay=case_params['Uay'],
        device=case_params['device']
    )
    logit_clip = param_training.get("logit_clip", 15.0)
    project_feq_moments = param_training.get("project_feq_moments", False)
    enforce_sod_symmetry = param_training.get("enforce_sod_symmetry", False)
    feq_nullspace_gamma = param_training["stage2"].get("feq_nullspace_gamma", param_training.get("feq_nullspace_gamma", 1.0))
    feq_nullspace_gamma_mode = param_training["stage2"].get("feq_nullspace_gamma_mode", param_training.get("feq_nullspace_gamma_mode", "fixed"))
    feq_nullspace_gamma_min = param_training["stage2"].get("feq_nullspace_gamma_min", param_training.get("feq_nullspace_gamma_min", 0.6))
    feq_nullspace_gamma_max = param_training["stage2"].get("feq_nullspace_gamma_max", param_training.get("feq_nullspace_gamma_max", 1.0))
    geq_mode = param_training["stage2"].get("geq_mode", param_training.get("geq_mode", "learned"))
    geq_nullspace_gamma = param_training["stage2"].get("geq_nullspace_gamma", param_training.get("geq_nullspace_gamma", 1.0))
    geq_nullspace_gamma_mode = param_training["stage2"].get("geq_nullspace_gamma_mode", param_training.get("geq_nullspace_gamma_mode", "fixed"))
    geq_nullspace_gamma_min = param_training["stage2"].get("geq_nullspace_gamma_min", param_training.get("geq_nullspace_gamma_min", 0.6))
    geq_nullspace_gamma_max = param_training["stage2"].get("geq_nullspace_gamma_max", param_training.get("geq_nullspace_gamma_max", 1.0))
    feq_mode = param_training["stage2"].get("feq_mode", "learned")
    feq_residual_scale = param_training["stage2"].get("feq_residual_scale", 1.0)
    feq_collision_mix = (
        args.feq_collision_mix
        if args.feq_collision_mix is not None
        else param_training["stage2"].get("feq_collision_mix_eval", 1.0)
    )

    if args.save_model:
        os.makedirs(param_training["stage2"]["model_dir"], exist_ok=True)
    all_F, all_G, all_Feq, all_Geq = load_data_stage_2(data_path)

    model = NeurDE(
        alpha_layer=[4] + [param_training["hidden_dim"]] * param_training["num_layers"],
        phi_layer=[2] + [param_training["hidden_dim"]] * param_training["num_layers"],
        activation='relu',
        learn_feq=learn_feq,
        learn_geq=True,
        logit_clip=logit_clip,
        project_feq_moments=project_feq_moments,
        enforce_sod_symmetry=enforce_sod_symmetry,
        feq_nullspace_gamma=feq_nullspace_gamma,
        feq_nullspace_gamma_mode=feq_nullspace_gamma_mode,
        feq_nullspace_gamma_min=feq_nullspace_gamma_min,
        feq_nullspace_gamma_max=feq_nullspace_gamma_max,
        geq_mode=geq_mode,
        geq_cv=sod_solver.Cv,
        geq_nullspace_gamma=geq_nullspace_gamma,
        geq_nullspace_gamma_mode=geq_nullspace_gamma_mode,
        geq_nullspace_gamma_min=geq_nullspace_gamma_min,
        geq_nullspace_gamma_max=geq_nullspace_gamma_max,
    ).to(device)



    if args.compile:
        model = torch.compile(model)
        sod_solver.collision = torch.compile(sod_solver.collision, dynamic=True, fullgraph=False)
        sod_solver.streaming = torch.compile(sod_solver.streaming, dynamic=True, fullgraph=False)
        sod_solver.shift_operator = torch.compile(sod_solver.shift_operator, dynamic=True, fullgraph=False)
        sod_solver.get_macroscopic = torch.compile(sod_solver.get_macroscopic, dynamic=True, fullgraph=False)
        print("Model compiled.")

    if args.trained_path:
        missing_keys, unexpected_keys = load_model_checkpoint(model, args.trained_path, map_location=device)
        print(f"Trained model loaded from {args.trained_path}")
        if missing_keys or unexpected_keys:
            print(f"Checkpoint mismatch. Missing: {missing_keys}, Unexpected: {unexpected_keys}")

  
    with h5py.File(data_path, "r") as f:
        all_rho = f["rho"][:]
        all_ux = f["ux"][:]
        all_uy = f["uy"][:]
        all_T = f["T"][:]
        all_Geq = f["Geq"][:]
        all_Feq = f["Feq"][:]
        all_Fi0 = f["Fi0"][:]
        all_Gi0 = f["Gi0"][:]
        analytic_rho = f["analytic_rho"][:] if "analytic_rho" in f else None
        analytic_ux = f["analytic_ux"][:] if "analytic_ux" in f else None
        analytic_T = f["analytic_T"][:] if "analytic_T" in f else None
        analytic_P = f["analytic_P"][:] if "analytic_P" in f else None

    all_P = all_rho * all_T
    if args.analytic_reference and analytic_rho is None:
        analytic_rho, analytic_ux, analytic_T, analytic_P = analytic_reference_from_case(
            args.case,
            case_params,
            steps=len(all_rho),
        )

    Uax, Uay = case_params["Uax"], case_params["Uay"]
    basis = create_basis(Uax, Uay, device)
   
    loss_func = calculate_relative_error

    feq_path = "learned" if learn_feq else "analytic"
    print(
        f"Testing Case {args.case} on {device}. feq_path={feq_path}, "
        f"feq_mode={feq_mode}, symmetry={enforce_sod_symmetry}, feq_gamma_mode={feq_nullspace_gamma_mode}, feq_gamma={feq_nullspace_gamma}, "
        f"geq_mode={geq_mode}, residual_scale={feq_residual_scale}, feq_mix={feq_collision_mix}, "
        f"geq_gamma_mode={geq_nullspace_gamma_mode}, geq_gamma={geq_nullspace_gamma}"
    )

    dtype = torch.get_default_dtype()
    start_idx = args.init_cond
    eval_steps = min(args.num_samples, len(all_Fi0) - start_idx)
    Fi0 = torch.tensor(all_Fi0[start_idx], dtype=dtype, device=device)
    Gi0 = torch.tensor(all_Gi0[start_idx], dtype=dtype, device=device)
    equilibrium_loss = 0.0
    solver_errors = []
    analytic_errors = []
    image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images', f'SOD_case{args.case}', 'test_NN', checkpoint_tag)
    os.makedirs(image_dir, exist_ok=True)

    def plot_rollout_state(step_idx, rho_pred, ux_pred, T_pred):
        plt.figure(figsize=(16, 6))
        plt.suptitle(f'SOD shock case {args.case} time {step_idx}', fontweight='bold', fontsize=25, y=0.95)

        linewidth = 4
        plt.subplot(221)
        plt.plot(detach(rho_pred[2, :]), linewidth=linewidth, label='nn')
        plt.plot(all_rho[step_idx, 2, :], linewidth=2, label='solver')
        if analytic_rho is not None:
            plt.plot(analytic_rho[step_idx, 2, :], linewidth=2, linestyle='--', label='analytic')
        plt.title('Density', fontsize=18)
        plt.legend()

        plt.subplot(222)
        plt.plot(detach(T_pred[2, :]), linewidth=linewidth, label='nn')
        plt.plot(all_T[step_idx, 2, :], linewidth=2, label='solver')
        if analytic_T is not None:
            plt.plot(analytic_T[step_idx, 2, :], linewidth=2, linestyle='--', label='analytic')
        plt.title('Temperature', fontsize=18)
        plt.legend()

        plt.subplot(223)
        plt.plot(detach(ux_pred[2, :]), linewidth=linewidth, label='nn')
        plt.plot(all_ux[step_idx, 2, :], linewidth=2, label='solver')
        if analytic_ux is not None:
            plt.plot(analytic_ux[step_idx, 2, :], linewidth=2, linestyle='--', label='analytic')
        plt.title('Velocity in x', fontsize=18)
        plt.legend()

        plt.subplot(224)
        P_pred = rho_pred * T_pred
        plt.plot(detach(P_pred[2, :]), linewidth=linewidth, label='nn')
        plt.plot(all_P[step_idx, 2, :], linewidth=2, label='solver')
        if analytic_P is not None:
            plt.plot(analytic_P[step_idx, 2, :], linewidth=2, linestyle='--', label='analytic')
        plt.title('Pressure', fontsize=18)
        plt.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=0.35, w_pad=0.35)
        plt.savefig(os.path.join(image_dir, f'SOD_case{args.case}_{step_idx}.png'))
        plt.close()

    with torch.no_grad():  
        for i in tqdm(range(eval_steps)):
            step_idx = start_idx + i
            rho, ux, uy, E = sod_solver.get_macroscopic(Fi0, Gi0)
            T = sod_solver.get_temp_from_energy(ux, uy, E)
            inputs = torch.stack([rho.unsqueeze(0), ux.unsqueeze(0), uy.unsqueeze(0), T.unsqueeze(0)], dim=1).to(device)
            analytic_feq = sod_solver.get_Feq(rho, ux, uy, T)
            if learn_feq:
                Feq_pred, Geq_pred = model(inputs, basis)
                Feq_target = torch.tensor(all_Feq[step_idx], dtype=dtype, device=device)
                Feq_pred_grid = reshape_equilibrium_prediction(Feq_pred, Feq_target.unsqueeze(0).shape).squeeze(0)
                Feq_pred_grid = build_effective_feq(
                    learn_feq=learn_feq,
                    feq_mode=feq_mode,
                    feq_residual_scale=feq_residual_scale,
                    feq_collision_mix=feq_collision_mix,
                    project_feq_moments=project_feq_moments,
                    feq_nullspace_gamma=feq_nullspace_gamma,
                    inputs=inputs,
                    basis=basis,
                    analytic_feq=analytic_feq,
                    feq_target=Feq_target,
                    feq_pred_grid=Feq_pred_grid,
                )
                feq_loss = loss_func(
                    reshape_equilibrium_target(Feq_pred_grid.unsqueeze(0)),
                    reshape_equilibrium_target(Feq_target.unsqueeze(0)),
                )
            else:
                Geq_pred = model(inputs, basis)
                Feq_pred_grid = analytic_feq
                feq_loss = None
            Geq_target = torch.tensor(all_Geq[step_idx], dtype=dtype, device=device)
            Geq_pred_grid = reshape_equilibrium_prediction(Geq_pred, Geq_target.unsqueeze(0).shape).squeeze(0)
            geq_loss = loss_func(Geq_pred, reshape_equilibrium_target(Geq_target.unsqueeze(0)))
            inner_loss = 0.5 * (feq_loss + geq_loss) if feq_loss is not None else geq_loss
            equilibrium_loss += inner_loss.item()

            solver_rho = torch.tensor(all_rho[step_idx], dtype=dtype, device=device)
            solver_ux = torch.tensor(all_ux[step_idx], dtype=dtype, device=device)
            solver_T = torch.tensor(all_T[step_idx], dtype=dtype, device=device)
            solver_P = solver_rho * solver_T
            solver_error = 0.25 * (
                loss_func(rho, solver_rho)
                + loss_func(ux, solver_ux)
                + loss_func(T, solver_T)
                + loss_func(rho * T, solver_P)
            )
            solver_errors.append(float(solver_error.item()))

            if analytic_rho is not None and analytic_ux is not None and analytic_T is not None and analytic_P is not None:
                analytic_rho_t = torch.tensor(analytic_rho[step_idx], dtype=dtype, device=device)
                analytic_ux_t = torch.tensor(analytic_ux[step_idx], dtype=dtype, device=device)
                analytic_T_t = torch.tensor(analytic_T[step_idx], dtype=dtype, device=device)
                analytic_P_t = torch.tensor(analytic_P[step_idx], dtype=dtype, device=device)
                analytic_error = 0.25 * (
                    loss_func(rho, analytic_rho_t)
                    + loss_func(ux, analytic_ux_t)
                    + loss_func(T, analytic_T_t)
                    + loss_func(rho * T, analytic_P_t)
                )
                analytic_errors.append(float(analytic_error.item()))

            if i == 0 or i == eval_steps - 1 or ((i + 1) % args.plot_every == 0):
                plot_rollout_state(step_idx, rho, ux, T)

            Fi0, Gi0 = sod_solver.collision(Fi0, Gi0, Feq_pred_grid, Geq_pred_grid, rho, ux, uy, T)
            Fi0, Gi0 = sod_solver.streaming(Fi0, Gi0)

    reference_name = args.reference
    selected_errors = analytic_errors if reference_name == "analytic" and analytic_errors else solver_errors
    steps = np.arange(start_idx, start_idx + len(selected_errors))
    cumulative_errors = np.cumsum(selected_errors) / np.arange(1, len(selected_errors) + 1)

    plt.figure(figsize=(12, 5))
    plt.plot(steps, solver_errors, label="solver step error", linewidth=2)
    if analytic_errors:
        plt.plot(steps, analytic_errors, label="analytic step error", linewidth=2)
    plt.plot(steps, cumulative_errors, label=f"{reference_name} cumulative mean", linewidth=3)
    plt.xlabel("time step")
    plt.ylabel("relative error")
    plt.title(f"SOD case {args.case} rollout error accumulation from step {start_idx}")
    plt.legend()
    plt.tight_layout()
    summary_path = os.path.join(image_dir, f'SOD_case{args.case}_error_accumulation_{start_idx}_{start_idx + eval_steps - 1}.png')
    plt.savefig(summary_path)
    plt.close()

    print(f"Average equilibrium loss over {eval_steps} steps: {equilibrium_loss / max(eval_steps, 1):.6f}")
    print(f"Average solver macro error over {eval_steps} steps: {np.mean(solver_errors):.6f}")
    print(f"Final solver macro error at step {start_idx + eval_steps - 1}: {solver_errors[-1]:.6f}")
    if analytic_errors:
        print(f"Average analytic macro error over {eval_steps} steps: {np.mean(analytic_errors):.6f}")
        print(f"Final analytic macro error at step {start_idx + eval_steps - 1}: {analytic_errors[-1]:.6f}")
    print(f"Saved error accumulation plot to: {summary_path}")
