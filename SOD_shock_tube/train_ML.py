import torch
import torch.nn as nn
import numpy as np
from architectures import DeepONetCP
from utilities import detach, get_device, dispatch_optimizer
import argparse
from SOD_solver import SODSolver
import h5py
import yaml
import os





if __name__ == "__main__":
    set_seed(42)
    args = argparse.ArgumentParser()
    args.add_argument("--batch_size", type=int, default=16)
    args.add_argument("--lr", type=float, default=0.001)
    args.add_argument("--epochs", type=int, default=500)
    args.add_argument("--hidden_dim", type=int, default=32)
    args.add_argument("--num_layers", type=int, default=4)
    args.add_argument('--case', type=int, choices=[1, 2], help='Choose case 1 or 2', default=1)
    args.add_argument("--num_rollouts", type=int, default=5)
    args.add_argument("--pre_trained_path", type=str, default=None)
    args = args.parse_args()
    

    '''data loading
    before running this file run dataGen first to obtain data
        '''
    batch_size = args.batch_size
    # dataset = torch.load(f"data_base/SOD_deepOnet_v1_500.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open("Sod_cases_param.yml", 'r') as f: 
        cases = yaml.load(f, Loader=yaml.FullLoader)    

    case_params = cases[args.case]
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
    # train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DeepONetCP(branch_layer=[4]+[args.hidden_dim]*args.num_layers,trunk_layer=[2]+[args.hidden_dim]*args.num_layers,activation='relu')


    if args.pre_trained_path is not None:
        checkpoint = torch.load(args.pre_trained_path)
        model.load_state_dict(checkpoint["model_state_dict"])

    model = model.cuda()
    optimizer = dispatch_optimizer(model,lr=args.lr)
    loss_func = nn.MSELoss()
    ex = sod_solver.ex
    ey = sod_solver.ey
    basis = torch.stack([ex,ey],dim=-1).cuda()
    initial_conditions_func = getattr(sod_solver, case_params['initial_conditions_func'])

    with h5py.File(f"data_base/{case_params["filename"]}","r") as f:
        all_ux = f["ux"][:]
        all_uy = f["uy"][:]
        all_T = f["T"][:]
        all_rho = f["rho"][:]
        all_Geq = f["Geq"][:]
        all_Feq = f["Feq"][:]
        all_Fi0 = f["Fi0"][:]
        all_Gi0 = f["Gi0"][:]


    from tqdm import tqdm
    for epoch in tqdm(range(args.epochs), desc="Training epochs"):
        loss_epoch = 0 
        avg_loss_G = 0
        avg_loss_F = 0
        for i in range(50):
            # Load initial conditions
            Fi0 = torch.tensor(all_Fi0[i]).float().cuda()
            Gi0 = torch.tensor(all_Gi0[i]).float().cuda()
            loss = 0
            optimizer.zero_grad()
            model.train()
            for rollout in range(args.num_rollouts):  # number of rollouts, e.g., 5
                rho, ux, uy, E = sod_solver.get_macroscopic(Fi0, Gi0)
                T = sod_solver.get_temp_from_energy(ux, uy, E)
                inputs = torch.stack([ux, uy, T, rho], dim=0).unsqueeze(0).float().cuda()
                Geq_target = torch.tensor(all_Geq[i + rollout]).float().cuda()
                Feq_target = torch.tensor(all_Feq[i + rollout]).float().cuda()
                Feq = sod_solver.get_Feq(rho, ux, uy, T)
                Geq_NN = model(inputs, basis)
                Geq_NN = Geq_NN.permute(1, 0).reshape(9, sod_solver.Y, sod_solver.X)
                data_loss_F = loss_func(Feq, Feq_target)
                data_loss_G = loss_func(Geq_NN, Geq_target)
                # Compute loss
                loss += data_loss_G
                loss_epoch += loss.item()
                avg_loss_G += data_loss_G.item()
                avg_loss_F += data_loss_F.item()

                # Update initial conditions for next rollout
                Fi0, Gi0 = sod_solver.collision(Fi0, Gi0, Feq, Geq_NN, rho, ux, uy, T) # maybe we need new vars to attach all gradients
                Fi, Gi = sod_solver.streaming(Fi0, Gi0)
                Fi0 = Fi
                Gi0 = Gi
            Geq_err = torch.norm(Geq_NN - Geq_target) / torch.norm(Geq_target)
            loss.backward()
            optimizer.step()
            print(f"Sample idx {i} - Feq: {data_loss_F.item():.3e}, Geq: {data_loss_G.item():.3e}, "
                    f"Geq relative error: {Geq_err.item():.2e} at rollout {rollout+1}/{args.num_rollouts}")
        n_total = (i + 1) * args.num_rollouts
        print(f"Epoch: {epoch}, loss: {loss_epoch / n_total:.3e}")
        print(f"Epoch: {epoch}, loss_F: {avg_loss_F / n_total:.3e}")
        print(f"Epoch: {epoch}, loss_G: {avg_loss_G / n_total:.3e}")
        if epoch % 10 == 0:
            os.makedirs("results", exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": vars(args)
            }, f"results/SOD_rollout_{args.case}.pt")