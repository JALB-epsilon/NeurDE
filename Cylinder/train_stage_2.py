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

    parser = argparse.ArgumentParser(description='Train Stage 2')
    parser.add_argument('--device', type=int, default=3, help='Device index')
    parser.add_argument("--compile", dest='compile', action='store_true', help='Compile', default=False)
    parser.add_argument('--save_model', action='store_true', help='Save model checkpoints (enabled by default)')
    parser.add_argument('--no_save_model', dest='save_model', action='store_false', help='Disable model checkpoint saving')
    parser.add_argument('--num_samples', type=int, default=500, help='Number of samples')
    parser.add_argument("--save_frequency", default=1, help='Save model')
    parser.add_argument("--pre_trained_path", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.set_defaults(save_model=True)
    args = parser.parse_args()

    device = get_device(args.device)
    dtype = resolve_torch_dtype(args.dtype)
    if args.pre_trained_path:
        args.pre_trained_path = args.pre_trained_path.replace("Cylinder/", "")
        print(args.pre_trained_path)


    with open("cylinder_param.yml", 'r') as stream:
        case_params = yaml.safe_load(stream)
    case_params['device'] = device

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
    number_of_rollout = param_training["stage2"]["N"]
    supervision_mode = str(param_training["stage2"].get("supervision", "geq")).lower()
    if supervision_mode != "geq":
        raise ValueError(f"Cylinder stage2 only supports geq supervision right now, got: {supervision_mode}")

    os.makedirs(param_training["stage2"]["model_dir"], exist_ok=True)
    all_F, all_G, all_Feq, all_Geq = load_data_stage_2(param_training["data_dir"])
    dataset = RolloutBatchDataset(all_Fi=all_F[:args.num_samples],
                                    all_Gi=all_G[:args.num_samples],
                                    all_Feq=all_Feq[:args.num_samples],
                                    all_Geq=all_Geq[:args.num_samples],
                                    number_of_rollout=number_of_rollout,
                                    dtype=dtype,
                                    )

    stage2_batch_size = param_training["stage2"].get("batch_size", 1)
    dataloader = DataLoader(dataset, batch_size=stage2_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    val_dataset = Cylinder_stage2(F = all_F[args.num_samples:args.num_samples+100],
                                        G=all_G[args.num_samples:args.num_samples+100],
                                        Feq=all_Feq[args.num_samples:args.num_samples+100],
                                        Geq=all_Geq[args.num_samples:args.num_samples+100],
                                        dtype=dtype,)

    model = NeurDE(
        alpha_layer=[4] + [param_training["hidden_dim"]] * param_training["num_layers"],
        phi_layer=[2] + [param_training["hidden_dim"]] * param_training["num_layers"],
        activation='relu'
    ).to(device=device, dtype=dtype)

    if args.compile:
        model = torch.compile(model)
        cylinder_solver.collision = torch.compile(cylinder_solver.collision, dynamic=True, fullgraph=False)
        cylinder_solver.streaming = torch.compile(cylinder_solver.streaming, dynamic=True, fullgraph=False)
        cylinder_solver.shift_operator = torch.compile(cylinder_solver.shift_operator, dynamic=True, fullgraph=False)
        cylinder_solver.get_macroscopic = torch.compile(cylinder_solver.get_macroscopic, dynamic=True, fullgraph=False)
        cylinder_solver.get_Feq = torch.compile(cylinder_solver.get_Feq, dynamic=True, fullgraph=False)
        cylinder_solver.get_temp_from_energy = torch.compile(cylinder_solver.get_temp_from_energy, dynamic=True, fullgraph=False)
        print("Model compiled.")

    if args.pre_trained_path:
        if args.compile:
            checkpoint = torch.load(args.pre_trained_path, map_location=device)
            model.load_state_dict(checkpoint)
        elif not args.compile:
            checkpoint = torch.load(args.pre_trained_path, map_location=device)
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
    
    cs0 = np.sqrt(case_params["vuy"]*case_params["T0"])
    U0 = case_params["Ma0"] * cs0
    Uax = U0 * case_params["Ns"]
    Uay = 0
    basis = create_basis(Uax, Uay, device, dtype=dtype)

    epochs = param_training["stage2"]["epochs"]
    loss_func = calculate_batch_relative_error

    print(
        f"Training Case Cylinder on {device}. Epochs: {epochs}, Samples: {args.num_samples}, "
        f"supervision={supervision_mode}"
    )

    best_losses = [float('inf')] * 3
    best_models = [None] * 3
    best_model_paths = [None] * 3

    save_frequency = args.save_frequency
    epochs_since_last_save = [0] * 3
    last_epoch_loss = 0.0

    for epoch in tqdm(range(epochs), desc="Epochs"):
        loss_epoch = 0
        for batch_idx, (F_seq, G_seq, Feq_seq, Geq_seq) in enumerate(dataloader):
            optimizer.zero_grad()
            model.train()
            F_seq = F_seq.to(device=device, dtype=dtype)
            G_seq = G_seq.to(device=device, dtype=dtype)
            Geq_seq = Geq_seq.to(device=device, dtype=dtype)
            batch_size = F_seq.shape[0]
            Fi0 = F_seq[:, 0, ...]
            Gi0 = G_seq[:, 0, ...]
            total_loss = torch.zeros((), device=device, dtype=dtype)
            for rollout in range(number_of_rollout):
                rho, ux, uy, E = cylinder_solver.get_macroscopic(Fi0, Gi0)
                T = cylinder_solver.get_temp_from_energy(ux, uy, E)
                Feq = cylinder_solver.get_Feq(rho, ux, uy, T)
                inputs = torch.stack([rho, ux, uy, T], dim=1)
                Geq_pred_flat = model(inputs, basis)
                Geq_target = Geq_seq[:, rollout]
                pred_batch = Geq_pred_flat.reshape(batch_size, cylinder_solver.Y * cylinder_solver.X, cylinder_solver.Qn)
                target_batch = Geq_target.permute(0, 2, 3, 1).reshape(batch_size, cylinder_solver.Y * cylinder_solver.X, cylinder_solver.Qn)
                inner_loss = loss_func(pred_batch, target_batch)
                total_loss = total_loss + inner_loss
                Geq_pred = Geq_pred_flat.reshape(batch_size, cylinder_solver.Y, cylinder_solver.X, cylinder_solver.Qn).permute(0, 3, 1, 2)
                Fi0, Gi0 = cylinder_solver.collision(Fi0, Gi0, Feq, Geq_pred, rho, ux, uy, T)
                Fi, Gi = cylinder_solver.streaming(Fi0, Gi0)

                khi = torch.zeros_like(ux)
                zetax = torch.zeros_like(ux)
                zetay = torch.zeros_like(ux)

                Fi_obs_cyl, Gi_obs_cyl, Fi_obs_Inlet, Gi_obs_Inlet = cylinder_solver.get_obs_distribution(
                                            rho,
                                            ux, 
                                            uy,
                                            T,
                                            khi,
                                            zetax,
                                            zetay)

                Fi0, Gi0 = cylinder_solver.enforce_Obs_and_BC(Fi,
                                            Gi,
                                            Fi_obs_cyl,
                                            Gi_obs_cyl,
                                            Fi_obs_Inlet,
                                            Gi_obs_Inlet)
            total_loss = total_loss / float(number_of_rollout)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_epoch += total_loss.item()
            print(f"Epoch: {epoch}, Batch ID: {batch_idx}, Loss: {total_loss.item():.6f}")
        current_loss = loss_epoch / len(dataloader)
        scheduler.step()

        if current_loss < max(best_losses):
            max_index = best_losses.index(max(best_losses))
            best_losses[max_index] = current_loss
            best_models[max_index] = model.state_dict()

            if args.save_model and epochs_since_last_save[max_index] >= save_frequency:
                if best_model_paths[max_index] and os.path.exists(best_model_paths[max_index]):
                    os.remove(best_model_paths[max_index])
                save_path = os.path.join(param_training["stage2"]["model_dir"], f"best_model_epoch_{epoch+1}_top_{max_index+1}_{current_loss:.12f}.pt")
                torch.save(best_models[max_index], save_path)
                print(f"Top {max_index+1} model saved to: {save_path}")
                best_model_paths[max_index] = save_path
                epochs_since_last_save[max_index] = 0  # reset the counter
            else:
                epochs_since_last_save[max_index] += 1

        else:
            for i in range(3):
                epochs_since_last_save[i] += 1

        if epoch % 50 == 0 and epoch > 0:
            print(f"Epoch: {epoch}, Loss: {current_loss:.6f}")
            save_path = os.path.join(param_training["stage2"]["model_dir"], f"model_epoch_{epoch}_loss_{current_loss:.6f}.pt")
            torch.save(model.state_dict(), save_path)
            

    # Save the last model with its loss
    if args.save_model:
        last_epoch_loss = current_loss
        last_model_path = os.path.join(param_training["stage2"]["model_dir"], f"last_model_epoch_{epochs}_loss_{last_epoch_loss:.6f}.pt")
        torch.save(model.state_dict(), last_model_path)
        print(f"Last model saved to: {last_model_path}")
    print("Training complete.")
