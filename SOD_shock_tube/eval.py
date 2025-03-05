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
    parser.set_defaults(save_model=True)
    args = parser.parse_args()

    device = get_device(args.device)

    
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
        device=case_params['device']
    )

    with open("Sod_cases_param_training.yml", 'r') as stream:
        training_config = yaml.safe_load(stream)
    param_training = training_config[args.case]
    number_of_rollout = param_training["stage2"]["N"]

    os.makedirs(param_training["stage2"]["model_dir"], exist_ok=True)
    all_F, all_G, all_Feq, all_Geq = load_data_stage_2(param_training["data_dir"])

    model = NeurDE(
        alpha_layer=[4] + [param_training["hidden_dim"]] * param_training["num_layers"],
        branch_layer=[2] + [param_training["hidden_dim"]] * param_training["num_layers"],
        activation='relu'
    ).to(device)



    if args.compile:
        model = torch.compile(model)
        sod_solver.collision = torch.compile(sod_solver.collision, dynamic=True, fullgraph=False)
        sod_solver.streaming = torch.compile(sod_solver.streaming, dynamic=True, fullgraph=False)
        sod_solver.shift_operator = torch.compile(sod_solver.shift_operator, dynamic=True, fullgraph=False)
        sod_solver.get_macroscopic = torch.compile(sod_solver.get_macroscopic, dynamic=True, fullgraph=False)
        print("Model compiled.")

    if args.trained_path:
        if args.compile:
            checkpoint = torch.load(args.trained_path)
            model.load_state_dict(checkpoint)
        elif not args.compile:
            checkpoint = torch.load(args.trained_path)
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

    Uax, Uay = case_params["Uax"], case_params["Uay"]
    basis = create_basis(Uax, Uay, device)
   
    loss_func = calculate_relative_error

    print(f"Testing Case {args.case} on {device}.")

    Fi0 = torch.tensor(all_Fi0[args.num_samples], device=device)
    Gi0 = torch.tensor(all_Gi0[args.num_samples], device=device)
    loss=0
    with torch.no_grad():  
            for i in tqdm(range(args.num_samples)):
                rho, ux, uy, E = sod_solver.get_macroscopic(Fi0.squeeze(0), Gi0.squeeze(0))
                T = sod_solver.get_temp_from_energy(ux, uy, E)
                Feq = sod_solver.get_Feq(rho, ux, uy, T)
                inputs = torch.stack([rho.unsqueeze(0), ux.unsqueeze(0), uy.unsqueeze(0), T.unsqueeze(0)], dim=1).to(device)
                Geq_pred = model(inputs, basis)
   
                Geq_target = torch.tensor(all_Gi0[args.num_samples], device=device).unsqueeze(0)

                inner_lose = loss_func(Geq_pred, Geq_target.permute(0, 2, 3, 1).reshape(-1, 9))
                loss += inner_lose
                Fi0, Gi0 = sod_solver.collision(Fi0.squeeze(0), Gi0.squeeze(0), Feq, Geq_pred.permute(1, 0).reshape(9, sod_solver.Y, sod_solver.X), rho, ux, uy, T)
                Fi, Gi = sod_solver.streaming(Fi0, Gi0)
                Fi0 = Fi
                Gi0 = Gi



                plt.figure(figsize=(16, 6))
                case_number = args.case
                # Larger title and reduced whitespace
                plt.suptitle(f'SOD shock case {case_number} time {i+args.init_cond}', fontweight='bold', fontsize=25, y=0.95) 

                linewidth = 5

                plt.subplot(221)
                plt.plot(detach(rho[2, :]), linewidth=linewidth)
                plt.plot(all_rho[args.init_cond+i, 2, :], linewidth=2)

                plt.title('Density', fontsize=18)  # Slightly increased fontsize

                plt.subplot(222)
                plt.plot(detach(T[2, :]), linewidth=linewidth)
                plt.plot((all_T[args.init_cond+i, 2, :]), linewidth=2)
                plt.title('Temperature', fontsize=18)

                plt.subplot(223)
                plt.plot(detach(ux[2, :]), linewidth=linewidth)
                plt.plot((all_ux[args.init_cond+i, 2, :]), linewidth=2)
                plt.title('Velocity in x', fontsize=18)

                plt.subplot(224)
                P = rho * T
                plt.plot(detach(P[2, :]), linewidth=linewidth)
                plt.plot((all_P[args.init_cond+i, 2, :]), linewidth=2)
                plt.title('Pressure', fontsize=18)

 
                # Reduced whitespace - Key changes here:
                plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=0.35, w_pad=0.35)  

                image_dir = os.path.join(f'images/ SOD_case{case_number}/test_NN')
                os.makedirs(image_dir, exist_ok=True)
                plt.savefig(os.path.join(image_dir, f'SOD_case{case_number}_{i+args.init_cond}.png'))
                plt.close()
