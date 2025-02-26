import torch
import torch.nn as nn
import numpy as np
from src import DeepONetCP
from utilities import detach, get_device
import argparse
from SOD_solver import SODSolver
import h5py
import yaml
import os
def dispatch_optimizer(model,lr=0.001):
    if isinstance(model,torch.nn.Module):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        return optimizer
    elif isinstance(model,list):
        optimizer = [torch.optim.Adam(model[i].parameters(), lr=lr) for i in range(3)]
        return optimizer
        


# def training_stage1(model, optimizer, loss_func, args, basis, all_ux, all_uy, all_T, all_rho, all_Geq):
#     from tqdm import tqdm
#     model.train()
#     for epoch in tqdm(range(args.epochs)):
#         loss_epoch = 0 
#         for i in range(500):
#             # Load data and prepare inputs
#             ux = torch.tensor(all_ux[i])
#             uy = torch.tensor(all_uy[i])
#             T  = torch.tensor(all_T[i])
#             rho = torch.tensor(all_rho[i])
#             inputs = torch.stack([ux, uy, T, rho], dim=0).unsqueeze(0).float().cuda()
            
#             # Ground truth for Geq
#             Geq_gt = torch.tensor(all_Geq[i])
#             Geq_gt = Geq_gt.permute(1, 2, 0).reshape(-1, 9).float().cuda() 
            
#             optimizer.zero_grad()
#             Geq_pred = model(inputs, basis)
#             loss = loss_func(Geq_pred, Geq_gt)  # MSE loss
#             loss.backward()
#             optimizer.step()
#             loss_epoch += loss.item()
        
#         print(f"Epoch: {epoch}, loss: {loss_epoch/500:.3e}")
#         os.makedirs("results", exist_ok=True)
#         torch.save({"model_state_dict": model.state_dict(),
#                     "optimizer_state_dict": optimizer.state_dict(),
#                     "config": vars(args)},
#                     f"results/SOD_vanila_no_rollout_case{args.case}.pt")
        

if __name__ == "__main__":

    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed(0)
    args = argparse.ArgumentParser()
    args.add_argument("--batch_size", type=int, default=16)
    args.add_argument("--lr", type=float, default=0.001)
    args.add_argument("--epochs", type=int, default=500)
    args.add_argument("--hidden_dim", type=int, default=32)
    args.add_argument("--num_layers", type=int, default=4)
    args.add_argument('--case', type=int, choices=[1, 2], help='Choose case 1 or 2', default=1)
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
    model = DeepONetCP(branch_layer=[4]+[args.hidden_dim]*args.num_layers,trunk_layer=[2]+[args.hidden_dim]*args.num_layers)
    model = model.cuda()
    optimizer = dispatch_optimizer(model,lr=args.lr)
    loss_func = nn.MSELoss()
    ex = SODSolver.ex
    ey = SODSolver.ey
    basis = torch.stack([ex,ey],dim=-1).cuda()
    # print(f"basis shape: {basis.shape}")
    with h5py.File(f"data_base/{case_params["filename"]}","r") as f:
        all_ux = f["ux"][:]
        all_uy = f["uy"][:]
        all_T = f["T"][:]
        all_rho = f["rho"][:]
        all_Geq = f["Geq"][:]

    from tqdm import tqdm
    for epoch in tqdm(range(args.epochs)):
        loss_epoch = 0 
        avg_loss_Hfunc = 0
        avg_loss_qx = 0
        avg_loss_qy = 0
        avg_loss_E = 0
        avg_loss_G = 0
        avg_loss_F = 0
        for i in range(50):
            # load IC 
            Fi0 = torch.tensor(all_Fi[i])
            Gi0 = torch.tensor(all_Gi[i])
            
            Fi0 = Fi0.float().cuda()
            Gi0 = Gi0.float().cuda()
            loss = 0
            optimizer.zero_grad()
            model.train()
            for rollout in range(args.num_rollouts): # number of rollout is 5
                rho, ux, uy, T,E,H, Ma,qsx,qsy,Feq,Gis = solver.before_Geq(Fi0, Gi0)
                inputs = torch.stack([ux,uy,T,rho],dim=0).unsqueeze(0).float().cuda()
                Geq_target = torch.tensor(all_Geq[i+rollout])
                Feq_target = torch.tensor(all_Feq[i+rollout])
                Feq_target = Feq_target.float().cuda()
                Geq_target = Geq_target.float().cuda()
                # Geq_target = Geq_target.permute(1,2,0).reshape(-1,9).float().cuda()
                # targets = targets.permute(1,2,0).reshape(-1,9).float().cuda()
                Geq_NN = model(inputs,basis)
                Geq_NN = Geq_NN.permute(1,0).reshape(9,solver.Y,solver.X)
                # EE_pred = torch.sum(Geq_NN,dim=0)
                # EE_target = torch.sum(Geq_target,dim=0)
                # raltive_error = torch.norm(EE_pred-EE_target)/torch.norm(EE_target)
                constraint_H, constraint_Energy, constraint_flux_x, constraint_flux_y = solver.get_Loss_v2(Feq,Geq_NN,Fi0,Gi0)
                loss_Hfunc = constraint_H.mean() # H ufnction result in nan
                loss_E = loss_func(constraint_Energy,torch.zeros_like(constraint_Energy))
                loss_qx = loss_func(constraint_flux_x,torch.zeros_like(constraint_flux_x))
                loss_qy = loss_func(constraint_flux_y,torch.zeros_like(constraint_flux_y))
                data_loss_F = loss_func(Feq,Feq_target)
                data_loss_G = loss_func(Geq_NN,Geq_target)
                # first stage: supervised learning 
                # second stage: take care of enerngy during rollout
                # third stage: Accelerate Lagrangian for extream accuracy
                Fi_new,Gi_new = solver.after_Geq(Geq_NN,Gis,Fi0,Gi0,Feq,rho, ux, uy, T,E,H, Ma,qsx,qsy)
                Fi0 = Fi_new.clone()
                Gi0 = Gi_new.clone()
                
                loss += data_loss_G
                loss_epoch += loss.item()
                avg_loss_qy += loss_qy.item()
                avg_loss_qx += loss_qx.item()
                avg_loss_E += loss_E.item()
                avg_loss_Hfunc += loss_Hfunc.item()
                avg_loss_G += data_loss_G.item()
                avg_loss_F += data_loss_F.item()
            Geq_err = torch.norm(Geq_NN-Geq_target)/torch.norm(Geq_target)
            loss.backward()
            optimizer.step()
            print(f"Feq: {data_loss_F.item():.3e}, Geq: {data_loss_G.item():.3e},H_function: {loss_Hfunc.item():.3e} loss_E: {loss_E.item():.3e}, loss_qx: {loss_qx.item():.3e}, loss_qy: {loss_qy.item():.3e}, Geq relative erorr {Geq_err.item():.2e} at rollout {rollout+1}/{args.num_rollouts} sample idx {i}")
        print(f"Epoch: {epoch}, loss: {loss_epoch/((i+1)*args.num_rollouts):.3e}")
        print(f"Epoch: {epoch}, loss_Hfunc: {avg_loss_Hfunc/((i+1)*args.num_rollouts):.3e}")
        print(f"Epoch: {epoch}, loss_E: {avg_loss_E/((i+1)*args.num_rollouts):.3e}")
        print(f"Epoch: {epoch}, loss_qx: {avg_loss_qx/((i+1)*args.num_rollouts):.3e}")
        print(f"Epoch: {epoch}, loss_qy: {avg_loss_qy/((i+1)*args.num_rollouts):.3e}")
        print(f"Epoch: {epoch}, loss_F: {avg_loss_F/((i+1)*args.num_rollouts):.3e}")
        print(f"Epoch: {epoch}, loss_G: {avg_loss_G/((i+1)*args.num_rollouts):.3e}")
        if epoch % 10 == 0:
            os.makedirs("results",exist_ok=True)
            torch.save({"model_state_dict":model.state_dict(),
                        "optimizer_state_dict":optimizer.state_dict(),
                        "config":vars(args)},
                    f"results/SOD_case2_rollout_{ID}.pt")