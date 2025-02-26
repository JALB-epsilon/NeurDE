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
        for i in range(500):
            # ground truth u v T rho 
            ux = torch.tensor(all_ux[i])
            uy = torch.tensor(all_uy[i])
            T = torch.tensor(all_T[i])
            rho = torch.tensor(all_rho[i])
            inputs = torch.stack([ux,uy,T,rho],dim=0).unsqueeze(0).float().cuda()
            targets = torch.tensor(all_Geq[i])
            targets = targets.permute(1,2,0).reshape(-1,9).float().cuda() 
            optimizer.zero_grad()
            pred = model(inputs,basis)
            loss = loss_func(pred,targets)
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
        print(f"Epoch: {epoch}, loss: {loss_epoch/i:.3e}")
        os.makedirs("results",exist_ok=True)
        torch.save({"model_state_dict":model.state_dict(),
                    "optimizer_state_dict":optimizer.state_dict(),
                    "config":vars(args)},
                f"results/SOD_vanila_no_rollout_case{args.case}.pt")
