import torch
from architectures import NeurDE
from utilities import *
import argparse
import yaml
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
from train_stage_1 import create_basis
from cylinder_solver import Cylinder_base
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
    parser.add_argument("--with_obs", action='store_true', help='With obstacle')
    parser.add_argument("--trained_path", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.set_defaults(save_model=True)
    parser.set_defaults(with_obs=True) # Ensure default is without_obs

    args = parser.parse_args()

    device = get_device(args.device)
    dtype = resolve_torch_dtype(args.dtype)

    
    if args.trained_path:
        args.trained_path = args.trained_path.replace("Cylinder/", "")
        print(args.trained_path)


    with open("cylinder_param.yml", 'r') as stream:
        case_params = yaml.safe_load(stream)
    case_params['device'] = device

    print(f"Cylinder shock tube problem")

    cylinder_solver = Cylinder_base(
                                    X=case_params['X'],
                                    Y=case_params['Y'],
                                    Qn=case_params['Qn'],
                                    radius=case_params['radius'],
                                    Ma0=case_params['Ma0'],
                                    Re=case_params['Re'],
                                    rho0=case_params['rho0'],
                                    T0=case_params['T0'],
                                    alpha1=case_params['alpha1'],
                                    alpha01=case_params['alpha01'],
                                    vuy=case_params['vuy'],
                                    Pr=case_params['Pr'],
                                    Ns=case_params['Ns'],
                                    device=device,
                                    dtype=dtype,
                                    )

    with open("cylinder_param_training.yml", 'r') as stream:
        param_training = yaml.safe_load(stream)
    model_config = get_model_config(param_training)
    number_of_rollout = param_training["stage2"]["N"]
    supervision_mode = str(param_training["stage2"].get("supervision", "geq")).lower()
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
        cylinder_solver.collision = torch.compile(cylinder_solver.collision, dynamic=True, fullgraph=False)
        cylinder_solver.streaming = torch.compile(cylinder_solver.streaming, dynamic=True, fullgraph=False)
        cylinder_solver.shift_operator = torch.compile(cylinder_solver.shift_operator, dynamic=True, fullgraph=False)
        cylinder_solver.get_macroscopic = torch.compile(cylinder_solver.get_macroscopic, dynamic=True, fullgraph=False)
        cylinder_solver.get_Feq = torch.compile(cylinder_solver.get_Feq, dynamic=True, fullgraph=False)
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

    all_Ma_GT = np.sqrt(all_ux ** 2 + all_uy ** 2) / np.sqrt(all_T*case_params['vuy'])

    cs0 = np.sqrt(case_params["vuy"]*case_params["T0"])
    U0 = case_params["Ma0"] * cs0
    Uax = U0 * case_params["Ns"]
    Uay = 0
    basis = create_basis(Uax, Uay, device, dtype=dtype)
   
    loss_func = calculate_relative_error

    print(f"Testing Case Cylinder on {device}.")

    Fi0 = torch.as_tensor(all_Fi0[args.init_cond], device=device, dtype=dtype).unsqueeze(0)
    Gi0 = torch.as_tensor(all_Gi0[args.init_cond], device=device, dtype=dtype).unsqueeze(0)
    loss=0
    khi = None
    zetax = None
    zetay = None
    print("Start testing")
    if args.with_obs is True:
        print("With obstacle")
    with torch.no_grad():  
            for i in tqdm(range(args.num_samples)):
                rho, ux, uy, E = cylinder_solver.get_macroscopic(Fi0, Gi0)
                T = cylinder_solver.get_temp_from_energy(ux, uy, E)
                inputs = torch.stack([rho, ux, uy, T], dim=1)
                equilibrium_pred_flat = model(inputs, basis)
   
                if supervision_mode == "feq":
                    equilibrium_target = torch.as_tensor(all_Feq[args.init_cond + i], device=device, dtype=dtype).unsqueeze(0)
                else:
                    equilibrium_target = torch.as_tensor(all_Geq[args.init_cond + i], device=device, dtype=dtype).unsqueeze(0)
                inner_lose = loss_func(equilibrium_pred_flat, equilibrium_target.permute(0, 2, 3, 1).reshape(-1, cylinder_solver.Qn))
                loss += inner_lose
                equilibrium_pred = equilibrium_pred_flat.reshape(1, cylinder_solver.Y, cylinder_solver.X, cylinder_solver.Qn).permute(0, 3, 1, 2)
                if learn_target == "geq":
                    Feq = cylinder_solver.get_Feq(rho, ux, uy, T)
                    Geq = equilibrium_pred
                else:
                    Feq = equilibrium_pred
                    if khi is None:
                        khi = torch.zeros_like(ux)
                        zetax = torch.zeros_like(ux)
                        zetay = torch.zeros_like(ux)
                    Geq, khi, zetax, zetay = cylinder_solver.get_Geq_Newton_solver(rho, ux, uy, T, khi, zetax, zetay)
                Fi0, Gi0 = cylinder_solver.collision(Fi0, Gi0, Feq, Geq, rho, ux, uy, T)
                Fi, Gi = cylinder_solver.streaming(Fi0, Gi0)
                if args.with_obs:
                    if khi is None:
                        khi_bc = torch.zeros_like(ux)
                        zetax_bc = torch.zeros_like(ux)
                        zetay_bc = torch.zeros_like(ux)
                    else:
                        khi_bc = khi
                        zetax_bc = zetax
                        zetay_bc = zetay

                    Fi_obs_cyl, Gi_obs_cyl, Fi_obs_Inlet, Gi_obs_Inlet = cylinder_solver.get_obs_distribution(
                                                                                                            rho,
                                                                                                            ux, 
                                                                                                            uy,
                                                                                                            T,
                                                                                                            khi_bc,
                                                                                                            zetax_bc,
                                                                                                            zetay_bc)

                    Fi_new, Gi_new = cylinder_solver.enforce_Obs_and_BC(Fi,
                                                                        Gi,
                                                                        Fi_obs_cyl,
                                                                        Gi_obs_cyl,
                                                                        Fi_obs_Inlet,
                                                                        Gi_obs_Inlet)

                    Fi0 = Fi_new
                    Gi0 = Gi_new

                else: 
                    Fi0 = Fi
                    Gi0 = Gi
                
                #plot the results of the Mach number 
                Ma_NN = cylinder_solver.get_local_Mach(ux, uy, T)[0]
                Ma_GT = all_Ma_GT[args.init_cond + i]
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(Ma_NN.cpu().numpy(), cmap='jet')
                plt.colorbar()
                plt.title(f'Mach number - NN')
                plt.subplot(1, 2, 2)
                plt.imshow(Ma_GT, cmap='jet')
                plt.colorbar()
                plt.title(f'Mach number - GT')
                plt.suptitle(f'Cylinder - Sample {i+args.init_cond}')

 
                # Reduced whitespace - Key changes here:
                plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=0.35, w_pad=0.35)  

                image_dir = os.path.join(f'images/Cylinder/test_NN')
                os.makedirs(image_dir, exist_ok=True)
                plt.savefig(os.path.join(image_dir, f'Cylinder_{i+args.init_cond}.png'))
                plt.close()
