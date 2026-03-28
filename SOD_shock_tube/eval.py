import torch
from architectures import NeurDE
from utilities import *
import argparse
import yaml
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
from train_stage_1 import create_basis
from SOD_solver import SODSolver
from exact_solution import build_exact_macro_rollout
import torch.nn as nn

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
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.set_defaults(save_model=True)
    args = parser.parse_args()

    device = get_device(args.device)
    dtype = resolve_torch_dtype(args.dtype)

    if not args.trained_path:
        raise ValueError("--trained_path is required for SOD evaluation.")
    args.trained_path = args.trained_path.replace("SOD_shock_tube/", "")
    print(args.trained_path)
    case_part = args.trained_path.split('/')[1]
    case_number = ''.join(filter(str.isdigit, case_part))
    args.case = int(case_number)
    print(args.case)

    with open("Sod_cases_param.yml", 'r') as stream:
        config = yaml.safe_load(stream)
    case_params = config[args.case]
    case_params['device'] = device

    print(f"Case {args.case}: SOD shock tube problem")

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
        device=case_params['device'],
        dtype=dtype,
    )

    with open("Sod_cases_param_training.yml", 'r') as stream:
        training_config = yaml.safe_load(stream)
    param_training = training_config[args.case]
    model_config = get_model_config(param_training)
    number_of_rollout = param_training["stage2"]["N"]
    supervision_mode = param_training["stage2"].get("supervision", "geq").lower()
    learn_target = resolve_stage_target(param_training["stage2"])

    os.makedirs(param_training["stage2"]["model_dir"], exist_ok=True)
    all_F, all_G, all_Feq, all_Geq = load_data_stage_2(param_training["data_dir"])

    model = NeurDE(
        alpha_layer=[4] + [param_training["hidden_dim"]] * param_training["num_layers"],
        phi_layer=[2] + [param_training["hidden_dim"]] * param_training["num_layers"],
        activation='relu',
        learn_feq=learn_target == "feq",
        learn_geq=learn_target == "geq",
        feq_mode=model_config["feq_mode"],
        geq_mode=model_config["geq_mode"],
        cv=1.0 / (case_params["vuy"] - 1.0),
        logit_clip=model_config["logit_clip"],
        feq_base_measure=model_config["feq_base_measure"],
        geq_base_measure=model_config["geq_base_measure"],
        newton_iters=model_config["newton_iters"],
        newton_tolerance=model_config["newton_tolerance"],
    ).to(device=device, dtype=dtype)



    if args.compile:
        model = torch.compile(model)
        sod_solver.collision = torch.compile(sod_solver.collision, dynamic=True, fullgraph=False)
        sod_solver.streaming = torch.compile(sod_solver.streaming, dynamic=True, fullgraph=False)
        sod_solver.shift_operator = torch.compile(sod_solver.shift_operator, dynamic=True, fullgraph=False)
        sod_solver.get_macroscopic = torch.compile(sod_solver.get_macroscopic, dynamic=True, fullgraph=False)
        print("Model compiled.")

    if args.trained_path:
        if args.compile:
            checkpoint = torch.load(args.trained_path, map_location=device)
            model.load_state_dict(checkpoint)
        elif not args.compile:
            checkpoint = torch.load(args.trained_path, map_location=device)
            new_state_dict = {}

            for k, v in checkpoint.items():
                if k.startswith("_orig_mod."):
                    new_k = k.replace("_orig_mod.", "")
                    new_state_dict[new_k] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict)
        print(f"Trained model loaded from {args.trained_path}")

  
    with h5py.File(param_training["data_dir"], "r") as f:
        all_rho = f["rho"][:]
        all_ux = f["ux"][:]
        all_uy = f["uy"][:]
        all_T = f["T"][:]
        all_Geq = f["Geq"][:]
        all_Feq = f["Feq"][:]
        all_Fi0 = f["Fi0"][:]
        all_Gi0 = f["Gi0"][:]

    all_P = all_rho * all_T

    use_exact_macro = supervision_mode in {"exact_macro", "macro"} and "exact_left_state" in case_params
    if use_exact_macro:
        exact_steps = args.init_cond + args.num_samples + 1
        exact_rho, exact_ux, exact_uy, exact_T = build_exact_macro_rollout(
            case_params,
            exact_steps,
            backend="torch",
            dtype=dtype_name_from_torch(dtype),
            device=device,
        )

    Uax, Uay = case_params["Uax"], case_params["Uay"]
    basis = create_basis(Uax, Uay, device, dtype=dtype)
   
    loss_func = calculate_relative_error
    macro_loss_func = calculate_batch_relative_error

    print(f"Testing Case {args.case} on {device}.")

    Fi0 = torch.as_tensor(all_Fi0[args.init_cond], device=device, dtype=dtype).unsqueeze(0)
    Gi0 = torch.as_tensor(all_Gi0[args.init_cond], device=device, dtype=dtype).unsqueeze(0)
    loss=0
    khi = None
    zetax = None
    zetay = None
    with torch.no_grad():  
            for i in tqdm(range(args.num_samples)):
                rho, ux, uy, E = sod_solver.get_macroscopic(Fi0, Gi0)
                T = sod_solver.get_temp_from_energy(ux, uy, E)
                inputs = torch.stack([rho, ux, uy, T], dim=1)
                equilibrium_pred_flat = model(inputs, basis)
   
                if use_exact_macro:
                    macro_target = torch.stack(
                        [
                            exact_rho[args.init_cond + i],
                            exact_ux[args.init_cond + i],
                            exact_uy[args.init_cond + i],
                            exact_T[args.init_cond + i],
                        ],
                        dim=0,
                    ).unsqueeze(0)
                    macro_pred = torch.stack([rho, ux, uy, T], dim=1)
                    inner_lose = macro_loss_func(macro_pred, macro_target)
                elif supervision_mode == "feq":
                    Feq_target = torch.as_tensor(all_Feq[args.init_cond + i], device=device, dtype=dtype).unsqueeze(0)
                    inner_lose = loss_func(equilibrium_pred_flat, Feq_target.permute(0, 2, 3, 1).reshape(-1, sod_solver.Qn))
                else:
                    Geq_target = torch.as_tensor(all_Geq[args.init_cond + i], device=device, dtype=dtype).unsqueeze(0)
                    inner_lose = loss_func(equilibrium_pred_flat, Geq_target.permute(0, 2, 3, 1).reshape(-1, sod_solver.Qn))
                loss += inner_lose
                equilibrium_pred = equilibrium_pred_flat.reshape(1, sod_solver.Y, sod_solver.X, sod_solver.Qn).permute(0, 3, 1, 2)
                if learn_target == "geq":
                    Feq = sod_solver.get_Feq(rho, ux, uy, T)
                    Geq = equilibrium_pred
                else:
                    Feq = equilibrium_pred
                    if khi is None:
                        khi = torch.zeros_like(ux)
                        zetax = torch.zeros_like(ux)
                        zetay = torch.zeros_like(ux)
                    Geq, khi, zetax, zetay = sod_solver.get_Geq_Newton_solver(rho, ux, uy, T, khi, zetax, zetay)
                Fi0, Gi0 = sod_solver.collision(Fi0, Gi0, Feq, Geq, rho, ux, uy, T)
                Fi0, Gi0 = sod_solver.streaming(Fi0, Gi0)

                rho_plot = rho[0]
                T_plot = T[0]
                ux_plot = ux[0]

                plt.figure(figsize=(16, 6))
                case_number = args.case
                # Larger title and reduced whitespace
                plt.suptitle(f'SOD shock case {case_number} time {i+args.init_cond}', fontweight='bold', fontsize=25, y=0.95) 

                linewidth = 5

                plt.subplot(221)
                plt.plot(detach(rho_plot[2, :]), linewidth=linewidth)
                plt.plot(detach(exact_rho[args.init_cond+i, 2, :]) if use_exact_macro else all_rho[args.init_cond+i, 2, :], linewidth=2)

                plt.title('Density', fontsize=18)  # Slightly increased fontsize

                plt.subplot(222)
                plt.plot(detach(T_plot[2, :]), linewidth=linewidth)
                plt.plot(detach(exact_T[args.init_cond+i, 2, :]) if use_exact_macro else all_T[args.init_cond+i, 2, :], linewidth=2)
                plt.title('Temperature', fontsize=18)

                plt.subplot(223)
                plt.plot(detach(ux_plot[2, :]), linewidth=linewidth)
                plt.plot(detach(exact_ux[args.init_cond+i, 2, :]) if use_exact_macro else all_ux[args.init_cond+i, 2, :], linewidth=2)
                plt.title('Velocity in x', fontsize=18)

                plt.subplot(224)
                P = rho_plot * T_plot
                plt.plot(detach(P[2, :]), linewidth=linewidth)
                if use_exact_macro:
                    exact_p = exact_rho[args.init_cond+i, 2, :] * exact_T[args.init_cond+i, 2, :]
                    plt.plot(detach(exact_p), linewidth=2)
                else:
                    plt.plot((all_P[args.init_cond+i, 2, :]), linewidth=2)
                plt.title('Pressure', fontsize=18)

 
                # Reduced whitespace - Key changes here:
                plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=0.35, w_pad=0.35)  

                image_dir = os.path.join(f'images/ SOD_case{case_number}/test_NN')
                os.makedirs(image_dir, exist_ok=True)
                plt.savefig(os.path.join(image_dir, f'SOD_case{case_number}_{i+args.init_cond}.png'))
                plt.close()
