import torch
import torch.nn as nn
from architectures import NeurDE
from utilities import *
import argparse
import yaml
from tqdm import tqdm
import os
from torch.utils.data import DataLoader

def create_basis(Uax, Uay, device):
    ex_values = [1, 0, -1, 0, 1, -1, -1, 1, 0]
    ey_values = [0, 1, 0, -1, 1, 1, -1, -1, 0]
    ex = torch.tensor(ex_values, dtype=torch.float32) + Uax
    ey = torch.tensor(ey_values, dtype=torch.float32) + Uay
    basis = torch.stack([ex, ey], dim=-1).to(device)
    return basis

if __name__ == "__main__":
    set_seed(0)

    parser = argparse.ArgumentParser(description='Train Stage 1')
    parser.add_argument('--device', type=int, default=3, help='Device index')
    parser.add_argument("--compile", dest='compile', action='store_true', help='Compile', default=False)
    parser.add_argument('--save_model', action='store_true', help='Save model checkpoints (enabled by default)')
    parser.add_argument('--no-save_model', dest='save_model', action='store_false', help='Disable model checkpoint saving')
    parser.add_argument('--case', type=int, choices=[1, 2], default=1, help='Case 1 or 2')
    parser.add_argument('--num_samples', type=int, default=500, help='Number of samples')
    parser.add_argument("--batch_size", type=int, default=32, help='Batch size')
    parser.add_argument("--save_frequency", default=50, help='Save model')
    parser.set_defaults(save_model=True)
    args = parser.parse_args()

    device = get_device(args.device)

    with open("Sod_cases_param.yml", 'r') as stream:
        config = yaml.safe_load(stream)
    case_params = config[args.case]
    case_params['device'] = device

    with open("Sod_cases_param _training.yml", 'r') as stream:
        training_config = yaml.safe_load(stream)
    param_training = training_config[args.case]

    os.makedirs(param_training["stage1"]["model_dir"], exist_ok=True)
    all_rho, all_ux, all_uy, all_T, all_Geq = load_equilibrium_state(param_training["data_dir"])

    dataset = SodDataset(all_rho[:args.num_samples], all_ux[:args.num_samples], all_uy[:args.num_samples], all_T[:args.num_samples], all_Geq[:args.num_samples])
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = NeurDE(
        alpha_layer=[4] + [param_training["hidden_dim"]] * param_training["num_layers"],
        branch_layer=[2] + [param_training["hidden_dim"]] * param_training["num_layers"],
        activation='relu'
    ).to(device)


    if args.compile:
        model = torch.compile(model)

    optimizer = dispatch_optimizer(model=model,
                                    lr=param_training["stage1"]["lr"],
                                    optimizer_type="AdamW")

    total_steps = len(dataloader) * param_training["stage1"]["epochs"]
    scheduler_type = param_training["stage1"]["scheduler"]
    scheduler_config = param_training["stage1"].get("scheduler_config", {}).get(scheduler_type,{}) 
    scheduler = get_scheduler(optimizer, scheduler_type, total_steps, scheduler_config)

    Uax, Uay = case_params["Uax"], case_params["Uay"]
    basis = create_basis(Uax, Uay, device)

    epochs = param_training["stage1"]["epochs"]
    loss_func = calculate_relative_error

    print(f"Training Case {args.case} on {device}. Epochs: {epochs}, Samples: {args.num_samples}")
    if args.compile:
        print("Model compiled.")

    best_losses = [float('inf')] * 3
    best_models = [None] * 3
    best_model_paths = [None] * 3

    save_frequency = args.save_frequency
    epochs_since_last_save = [0] * 3 

    for epoch in tqdm(range(epochs), desc="Epochs"):
        loss_epoch = 0
        for rho_batch, ux_batch, uy_batch, T_batch, Geq_batch in dataloader:
            input_data = torch.stack([rho_batch, ux_batch, uy_batch, T_batch], dim=1).to(device)
            targets = Geq_batch.permute(0, 2, 3, 1).reshape(-1, 9).to(device)
            optimizer.zero_grad()
            Geq_pred = model(input_data, basis)
            loss = loss_func(Geq_pred, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_epoch += loss.item()
            scheduler.step()

        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item():.6f}")

        current_loss = loss.item()

        if current_loss < max(best_losses):
            max_index = best_losses.index(max(best_losses))
            best_losses[max_index] = current_loss
            best_models[max_index] = model.state_dict()

            if args.save_model and epochs_since_last_save[max_index] >= save_frequency:
                if best_model_paths[max_index] and os.path.exists(best_model_paths[max_index]):
                    os.remove(best_model_paths[max_index])
                save_path = os.path.join(param_training["stage1"]["model_dir"], f"best_model_{args.case}_epoch_{epoch+1}_top_{max_index+1}_loss_{current_loss:.6f}.pt")
                torch.save(best_models[max_index], save_path)
                print(f"Top {max_index+1} model saved to: {save_path}")
                best_model_paths[max_index] = save_path
                epochs_since_last_save[max_index] = 0 #reset the counter
            else:
                epochs_since_last_save[max_index] +=1

        else:
            for i in range(3):
                epochs_since_last_save[i] +=1

    if not args.save_model:
        print("Model saving disabled.")

    print("Training complete.")
    
'''import torch
import torch.nn as nn
from architectures import NeurDE
from utilities import set_seed, get_device, load_equilibrium_state, dispatch_optimizer, calculate_relative_error
import argparse
import yaml
from tqdm import tqdm
import os
from torch.utils.data import Dataset, DataLoader

def create_basis(Uax, Uay, device):
    ex_values = [1, 0, -1, 0, 1, -1, -1, 1, 0]
    ey_values = [0, 1, 0, -1, 1, 1, -1, -1, 0]
    ex = torch.tensor(ex_values, dtype=torch.float32) + Uax
    ey = torch.tensor(ey_values, dtype=torch.float32) + Uay
    basis = torch.stack([ex, ey], dim=-1).to(device)
    return basis

class SodDataset(Dataset):
    def __init__(self, rho, ux, uy, T, Geq):
        self.rho = torch.tensor(rho, dtype=torch.float32)
        self.ux = torch.tensor(ux, dtype=torch.float32)
        self.uy = torch.tensor(uy, dtype=torch.float32)
        self.T = torch.tensor(T, dtype=torch.float32)
        self.Geq = torch.tensor(Geq, dtype=torch.float32)

    def __len__(self):
        return len(self.rho)

    def __getitem__(self, idx):
        return self.rho[idx], self.ux[idx], self.uy[idx], self.T[idx], self.Geq[idx]

if __name__ == "__main__":
    set_seed(0)

    parser = argparse.ArgumentParser(description='Train Stage 1')
    parser.add_argument('--device', type=int, default=3, help='Device index')
    parser.add_argument("--compile", dest='compile', action='store_true', help='Compile', default=False)
    parser.add_argument('--save_model', type=bool, default=True, help='Save model')
    parser.add_argument('--case', type=int, choices=[1, 2], default=1, help='Case 1 or 2')
    parser.add_argument('--no-save_model', dest='save_model', action='store_false')
    parser.add_argument('--num_samples', type=int, default=500, help='Number of samples')
    parser.add_argument("--batch_size", type=int, default=32, help='Batch size')
    parser.set_defaults(save_model=True)

    args = parser.parse_args()
    device = get_device(args.device)

    with open("Sod_cases_param.yml", 'r') as stream:
        config = yaml.safe_load(stream)
    case_params = config[args.case]
    case_params['device'] = device

    with open("Sod_cases_param _training.yml", 'r') as stream:
        training_config = yaml.safe_load(stream)
    param_training = training_config[args.case]

    os.makedirs(param_training["stage1"]["model_dir"], exist_ok=True)
    all_rho, all_ux, all_uy, all_T, all_Geq = load_equilibrium_state(param_training["data_dir"])

    dataset = SodDataset(all_rho[:args.num_samples], all_ux[:args.num_samples], all_uy[:args.num_samples], all_T[:args.num_samples], all_Geq[:args.num_samples])
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = NeurDE(
        alpha_layer=[4] + [param_training["hidden_dim"]] * param_training["num_layers"],
        branch_layer=[2] + [param_training["hidden_dim"]] * param_training["num_layers"],
        activation='relu'
    ).to(device)

    if args.compile:
        model = torch.compile(model)

    optimizer = dispatch_optimizer(model=model, 
                                   lr=param_training["stage1"]["lr"], 
                                   optimizer_type="AdamW")
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=param_training["stage1"]["step_size"], gamma=param_training["stage1"]["gamma"])

    Uax, Uay = case_params["Uax"], case_params["Uay"]
    basis = create_basis(Uax, Uay, device)

    epochs = param_training["stage1"]["epochs"]
    loss_func = calculate_relative_error

    print(f"Training Case {args.case} on {device}. Epochs: {epochs}, Samples: {args.num_samples}")
    if args.compile:
        print("Model compiled.")

    for epoch in tqdm(range(epochs), desc="Epochs"):
        loss_epoch = 0
        for rho_batch, ux_batch, uy_batch, T_batch, Geq_batch in dataloader:
            input_data = torch.stack([rho_batch, ux_batch, uy_batch, T_batch], dim=1).to(device)
            targets = Geq_batch.permute(0, 2, 3, 1).reshape(-1, 9).to(device)
            optimizer.zero_grad()
            Geq_pred = model(input_data, basis)
            loss = loss_func(Geq_pred, targets)
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()

        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item():.6f}")

        scheduler.step()

    if args.save_model:
        if epoch % 20 == 0:
            save_path = os.path.join(param_training["stage1"]["model_dir"], f"model_{args.case}_epoch_{epoch}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to: {save_path}")

    print("Training complete.")'''