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

    parser = argparse.ArgumentParser(description='Train Stage 2')
    parser.add_argument('--device', type=int, default=3, help='Device index')
    parser.add_argument("--compile", dest='compile', action='store_true', help='Compile', default=False)
    parser.add_argument('--save_model', action='store_true', help='Save model checkpoints (enabled by default)')
    parser.add_argument('--no_save_model', dest='save_model', action='store_false', help='Disable model checkpoint saving')
    parser.add_argument('--num_samples', type=int, default=500, help='Number of samples')
    parser.add_argument("--save_frequency", default=1, help='Save model')
    parser.add_argument("--TVD", dest='TVD', action='store_true', help='TVD norm', default=False)
    parser.add_argument("--pre_trained_path", type=str, default=None)
    parser.set_defaults(save_model=True)
    args = parser.parse_args()

    device = get_device(args.device)
    if args.pre_trained_path:
        args.pre_trained_path = args.pre_trained_path.replace("SOD_shock_tube/", "")
        print(args.pre_trained_path)
        case_part = args.pre_trained_path.split('/')[1]
        case_number = ''.join(filter(str.isdigit, case_part))
        args.case = int(case_number)
        print(args.case)

    else: 
        args.case = 1

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
    dataset = RolloutBatchDataset(all_Fi=all_F[:args.num_samples],
                                    all_Gi=all_G[:args.num_samples],
                                    all_Feq=all_Feq[:args.num_samples],
                                    all_Geq=all_Geq[:args.num_samples],
                                    number_of_rollout=number_of_rollout,
                                    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)  # batch size 1 to get each sequence.

 
    val_dataset = SodDataset_stage2(F = all_F[args.num_samples:args.num_samples+2*number_of_rollout],
                                    G=all_G[args.num_samples:args.num_samples+2*number_of_rollout],
                                    Feq=all_Feq[args.num_samples:args.num_samples+2*number_of_rollout],
                                    Geq=all_Geq[args.num_samples:args.num_samples+2*number_of_rollout],)
    
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
        sod_solver.get_Feq = torch.compile(sod_solver.get_Feq, dynamic=True, fullgraph=False)
        sod_solver.get_temp_from_energy = torch.compile(sod_solver.get_temp_from_energy, dynamic=True, fullgraph=False)
        print("Model compiled.")

    if args.pre_trained_path:
        if args.compile:
            checkpoint = torch.load(args.pre_trained_path)
            model.load_state_dict(checkpoint)
        elif not args.compile:
            checkpoint = torch.load(args.pre_trained_path)
            new_state_dict = {}

            for k, v in checkpoint.items():
                if k.startswith("_orig_mod."):
                    new_k = k.replace("_orig_mod.", "")
                    new_state_dict[new_k] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict)
        print(f"Pre-trained model loaded from {args.pre_trained_path}")


    optimizer = dispatch_optimizer(model=model,
                                    lr=param_training["stage2"]["lr"],
                                    optimizer_type="AdamW")

    total_steps = len(dataloader) * param_training["stage2"]["epochs"]
    scheduler_type = param_training["stage2"]["scheduler"]
    scheduler_config = param_training["stage2"].get("scheduler_config", {}).get(scheduler_type, {})
    scheduler = get_scheduler(optimizer, scheduler_type, total_steps, scheduler_config)
    
    Uax, Uay = case_params["Uax"], case_params["Uay"]
    basis = create_basis(Uax, Uay, device)

    epochs = param_training["stage2"]["epochs"]
    loss_func = calculate_relative_error

    print(f"Training Case {args.case} on {device}. Epochs: {epochs}, Samples: {args.num_samples}")

    best_losses = [float('inf')] * 3
    best_models = [None] * 3
    best_model_paths = [None] * 3

    save_frequency = args.save_frequency
    epochs_since_last_save = [0] * 3
    last_epoch_loss = 0.0

    # Get the first batch from the dataloader
    first_batch = next(iter(dataloader))
    Fi0, Gi0, Feq_seq, Geq_seq = first_batch
    Fi0 = Fi0[0, 0, ...].to(device)
    Gi0 = Gi0[0, 0, ...].to(device)

    if args.TVD:
        print("Using TVD")
        if args.compile:
            TVD_norm = torch.compile(TVD_norm, dynamic=True, fullgraph=False)
    current_loss = 0.0
    for epoch in tqdm(range(epochs), desc="Epochs"):
        loss_epoch = 0
        if args.TVD:
            ux_old = torch.zeros_like(Fi0[1, ...])
            T_old = torch.zeros_like(Fi0[1, ...])
            rho_old = torch.zeros_like(Fi0[1, ...])
            if args.TVD:
                tvd_weight = tvd_weight_scheduler(epoch, epochs, initial_weight=1e-8, final_weight=1)
        for batch_idx, (F_seq, G_seq, Feq_seq, Geq_seq) in enumerate(dataloader):
            optimizer.zero_grad()
            model.train()
            total_loss = 0
            F_seq = F_seq.to(device)
            G_seq = G_seq.to(device)
            Fi0 = F_seq[0, 0, ...]
            Gi0 = G_seq[0, 0, ...]
            for rollout in range(number_of_rollout):       
                rho, ux, uy, E = sod_solver.get_macroscopic(Fi0, Gi0)
                T = sod_solver.get_temp_from_energy(ux, uy, E)
                Feq = sod_solver.get_Feq(rho, ux, uy, T)
                inputs = torch.stack([rho.unsqueeze(0), ux.unsqueeze(0), uy.unsqueeze(0), T.unsqueeze(0)], dim=1).to(device)
                Geq_pred = model(inputs, basis)
                Geq_target = Geq_seq[0, rollout].to(device)
                inner_loss = loss_func(Geq_pred, Geq_target.permute(1, 2, 0).reshape(-1, 9))
                total_loss += inner_loss
                if args.TVD and rollout > 0:
                    loss_TVD = TVD_norm(ux, ux_old)+TVD_norm(T, T_old)+TVD_norm(rho, rho_old)
                    ux_old = ux.clone()
                    T_old = T.clone()
                    rho_old = rho.clone()
                    total_loss += tvd_weight*loss_TVD
                Fi0, Gi0 = sod_solver.collision(Fi0, Gi0, Feq, Geq_pred.permute(1, 0).reshape(sod_solver.Qn, sod_solver.Y, sod_solver.X), rho, ux, uy, T)
                Fi, Gi = sod_solver.streaming(Fi0, Gi0)
                Fi0 = Fi.detach()
                Gi0 = Gi.detach()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_epoch += total_loss.item()
            print(f"Epoch: {epoch}, Batch ID: {batch_idx}, Loss: {total_loss.item()/number_of_rollout:.6f}")

        scheduler.step()

        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, Loss: {loss_epoch/len(dataloader):.6f}")

        current_loss = loss_epoch / len(dataloader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            Fi0 = next(iter(val_dataset))[0].to(device)
            Gi0 = next(iter(val_dataset))[1].to(device)
            
            for F_val, G_val, Feq_val, Geq_val in val_dataset:
                rho, ux, uy, E = sod_solver.get_macroscopic(Fi0, Gi0)
                T = sod_solver.get_temp_from_energy(ux, uy, E)
                Feq = sod_solver.get_Feq(rho, ux, uy, T)
                inputs = torch.stack([rho.unsqueeze(0), ux.unsqueeze(0), uy.unsqueeze(0), T.unsqueeze(0)], dim=1).to(device)
                Geq_pred = model(inputs, basis)
                Geq_target = Geq_val.to(device)
                inner_loss = loss_func(Geq_pred, Geq_target.permute(1, 2, 0).reshape(-1, 9))
                val_loss += inner_loss
                Fi0, Gi0 = sod_solver.collision(Fi0, Gi0, Feq, Geq_pred.permute(1, 0).reshape(sod_solver.Qn, sod_solver.Y, sod_solver.X), rho, ux, uy, T)
                Fi, Gi = sod_solver.streaming(Fi0, Gi0)
                Fi0 = Fi.detach()
                Gi0 = Gi.detach()
            val_loss /= len(val_dataset)
            print("-" * 50)
            print(f"Validation Loss: {val_loss:.6f}")
            print("-" * 50)

        if val_loss < max(best_losses):
            max_index = best_losses.index(max(best_losses))
            best_losses[max_index] = val_loss
            best_models[max_index] = model.state_dict()

            if args.save_model and epochs_since_last_save[max_index] >= save_frequency:
                if best_model_paths[max_index] and os.path.exists(best_model_paths[max_index]):
                    os.remove(best_model_paths[max_index])
                save_path = os.path.join(param_training["stage2"]["model_dir"], f"best_model_{args.case}_epoch_{epoch+1}_top_{max_index+1}_val_loss_{val_loss:.6f}.pt")
                torch.save(best_models[max_index], save_path)
                print(f"Top {max_index+1} model saved to: {save_path}")
                best_model_paths[max_index] = save_path
                epochs_since_last_save[max_index] = 0  # reset the counter
            else:
                epochs_since_last_save[max_index] += 1

        else:
            for i in range(3):
                epochs_since_last_save[i] += 1

        if epoch % 250 == 0:
            print(f"Epoch: {epoch}, Loss: {current_loss:.6f}")
            save_path = os.path.join(param_training["stage2"]["model_dir"], f"model_{args.case}_loss_{current_loss:.6f}.pt")
            torch.save(model.state_dict(), save_path)
            


    # Save the last model with its loss
    if args.save_model:
        last_epoch_loss = current_loss
        last_model_path = os.path.join(param_training["stage2"]["model_dir"], f"last_model_{args.case}_epoch_{epochs}_loss_{last_epoch_loss:.6f}.pt")
        torch.save(model.state_dict(), last_model_path)
        print(f"Last model saved to: {last_model_path}")
    print("Training complete.")
