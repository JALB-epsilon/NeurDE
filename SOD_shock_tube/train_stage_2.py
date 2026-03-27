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

    if "TVD" in param_training["stage2"]:
        args.TVD = True


    print(f"TVD Enabled: {args.TVD}")

    os.makedirs(param_training["stage2"]["model_dir"], exist_ok=True)
    all_F, all_G, all_Feq, all_Geq = load_data_stage_2(param_training["data_dir"])
    dataset = RolloutBatchDataset(all_Fi=all_F[:args.num_samples],
                                    all_Gi=all_G[:args.num_samples],
                                    all_Feq=all_Feq[:args.num_samples],
                                    all_Geq=all_Geq[:args.num_samples],
                                    number_of_rollout=number_of_rollout,
                                    )

    stage2_batch_size = param_training["stage2"].get("batch_size", 1)
    dataloader = DataLoader(dataset, batch_size=stage2_batch_size, shuffle=False, num_workers=4, pin_memory=True)

 
    val_dataset = SodDataset_stage2(F = all_F[args.num_samples:args.num_samples+100],
                                    G=all_G[args.num_samples:args.num_samples+100],
                                    Feq=all_Feq[args.num_samples:args.num_samples+100],
                                    Geq=all_Geq[args.num_samples:args.num_samples+100],)
    
    model = NeurDE(
        alpha_layer=[4] + [param_training["hidden_dim"]] * param_training["num_layers"],
        phi_layer=[2] + [param_training["hidden_dim"]] * param_training["num_layers"],
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

    if args.TVD:
        print("Using TVD")
        if args.compile:
            TVD_norm = torch.compile(TVD_norm, dynamic=True, fullgraph=False)
    current_loss = 0.0
    for epoch in tqdm(range(epochs), desc="Epochs"):
        loss_epoch = 0
        if args.TVD:
            tvd_weight = 15
        for batch_idx, (F_seq, G_seq, Feq_seq, Geq_seq) in enumerate(dataloader):
            optimizer.zero_grad()
            model.train()
            F_seq = F_seq.to(device)
            G_seq = G_seq.to(device)
            Geq_seq = Geq_seq.to(device)
            batch_size = F_seq.shape[0]
            Fi0 = F_seq[:, 0, ...]
            Gi0 = G_seq[:, 0, ...]
            total_loss = torch.zeros((), device=device)
            if args.TVD:
                ux_old = None
                T_old = None
                rho_old = None
            for rollout in range(number_of_rollout):
                rho, ux, uy, E = sod_solver.get_macroscopic(Fi0, Gi0)
                T = sod_solver.get_temp_from_energy(ux, uy, E)
                Feq = sod_solver.get_Feq(rho, ux, uy, T)
                inputs = torch.stack([rho, ux, uy, T], dim=1)
                Geq_pred_flat = model(inputs, basis)
                Geq_target = Geq_seq[:, rollout]
                pred_batch = Geq_pred_flat.reshape(batch_size, sod_solver.Y * sod_solver.X, sod_solver.Qn)
                target_batch = Geq_target.permute(0, 2, 3, 1).reshape(batch_size, sod_solver.Y * sod_solver.X, sod_solver.Qn)
                inner_loss = torch.stack([
                    loss_func(pred_batch[sample_idx], target_batch[sample_idx])
                    for sample_idx in range(batch_size)
                ]).mean()
                total_loss = total_loss + inner_loss
                if args.TVD and ux_old is not None:
                    loss_TVD = TVD_norm(T, T_old) + TVD_norm(ux, ux_old) + TVD_norm(rho, rho_old)
                    total_loss = total_loss + tvd_weight * loss_TVD
                if args.TVD:
                    ux_old = ux.clone()
                    T_old = T.clone()
                    rho_old = rho.clone()
                Geq_pred = Geq_pred_flat.reshape(batch_size, sod_solver.Y, sod_solver.X, sod_solver.Qn).permute(0, 3, 1, 2)
                Fi0, Gi0 = sod_solver.collision(Fi0, Gi0, Feq, Geq_pred, rho, ux, uy, T)
                Fi0, Gi0 = sod_solver.streaming(Fi0, Gi0)
            total_loss = total_loss / float(number_of_rollout)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_epoch += total_loss.item()
            print(f"Epoch: {epoch}, Batch ID: {batch_idx}, Loss: {total_loss.item():.6f}")

        scheduler.step()

        current_loss = loss_epoch / len(dataloader)

        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, Loss: {current_loss:.6f}")

        
        
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
                #print(inner_loss)
                val_loss += inner_loss
                Fi0, Gi0 = sod_solver.collision(Fi0, Gi0, Feq, Geq_pred.permute(1, 0).reshape(sod_solver.Qn, sod_solver.Y, sod_solver.X), rho, ux, uy, T)
                Fi, Gi = sod_solver.streaming(Fi0, Gi0)
                Fi0 = Fi
                Gi0 = Gi
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

        if epoch % 200 == 0:
            print(f"Epoch: {epoch}, Loss: {current_loss:.6f}")
            save_path = os.path.join(param_training["stage2"]["model_dir"], f"model_{args.case}_epoch_{epoch}_loss_{val_loss:.6f}.pt")
            torch.save(model.state_dict(), save_path)
            


    # Save the last model with its loss
    if args.save_model:
        last_epoch_loss = current_loss
        last_model_path = os.path.join(param_training["stage2"]["model_dir"], f"last_model_{args.case}_epoch_{epochs}_loss_{last_epoch_loss:.6f}.pt")
        torch.save(model.state_dict(), last_model_path)
        print(f"Last model saved to: {last_model_path}")
    print("Training complete.")
