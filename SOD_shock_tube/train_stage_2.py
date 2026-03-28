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

    parser = argparse.ArgumentParser(description='Train Stage 2')
    parser.add_argument('--device', type=int, default=3, help='Device index')
    parser.add_argument('--case', type=int, choices=[1, 2], default=None, help='Case 1 or 2')
    parser.add_argument("--compile", dest='compile', action='store_true', help='Compile', default=False)
    parser.add_argument('--save_model', action='store_true', help='Save model checkpoints (enabled by default)')
    parser.add_argument('--no_save_model', dest='save_model', action='store_false', help='Disable model checkpoint saving')
    parser.add_argument('--num_samples', type=int, default=500, help='Number of samples')
    parser.add_argument("--save_frequency", type=int, default=1, help='Save model')
    parser.add_argument("--TVD", dest='TVD', action='store_true', help='TVD norm', default=False)
    parser.add_argument("--disable_tvd", action='store_true', help='Disable TVD even if enabled in the YAML')
    parser.add_argument("--pre_trained_path", type=str, default=None)
    parser.add_argument("--epochs_override", type=int, default=None)
    parser.add_argument("--rollout_override", type=int, default=None)
    parser.add_argument("--supervision_override", type=str, default=None)
    parser.add_argument("--learn_target_override", type=str, default=None)
    parser.add_argument("--model_dir_override", type=str, default=None)
    parser.add_argument("--batch_size_override", type=int, default=None)
    parser.add_argument("--lr_override", type=float, default=None)
    parser.add_argument("--feq_mode_override", type=str, default=None)
    parser.add_argument("--geq_mode_override", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.set_defaults(save_model=True)
    args = parser.parse_args()

    device = get_device(args.device)
    dtype = resolve_torch_dtype(args.dtype)
    if args.pre_trained_path:
        args.pre_trained_path = args.pre_trained_path.replace("SOD_shock_tube/", "")
        print(args.pre_trained_path)
        case_part = args.pre_trained_path.split('/')[1]
        case_number = ''.join(filter(str.isdigit, case_part))
        args.case = int(case_number)
        print(args.case)

    elif args.case is None:
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
        device=case_params['device'],
        dtype=dtype,
    )

    with open("Sod_cases_param_training.yml", 'r') as stream:
        training_config = yaml.safe_load(stream)
    param_training = training_config[args.case]
    if args.rollout_override is not None:
        param_training["stage2"]["N"] = int(args.rollout_override)
    if args.epochs_override is not None:
        param_training["stage2"]["epochs"] = int(args.epochs_override)
    if args.supervision_override is not None:
        param_training["stage2"]["supervision"] = str(args.supervision_override)
    if args.learn_target_override is not None:
        param_training["stage2"]["learn_target"] = str(args.learn_target_override)
    if args.model_dir_override is not None:
        param_training["stage2"]["model_dir"] = str(args.model_dir_override)
    if args.batch_size_override is not None:
        param_training["stage2"]["batch_size"] = int(args.batch_size_override)
    if args.lr_override is not None:
        param_training["stage2"]["lr"] = float(args.lr_override)
    if args.feq_mode_override is not None:
        param_training.setdefault("model", {})["feq_mode"] = str(args.feq_mode_override)
    if args.geq_mode_override is not None:
        param_training.setdefault("model", {})["geq_mode"] = str(args.geq_mode_override)
    model_config = get_model_config(param_training)
    number_of_rollout = param_training["stage2"]["N"]
    supervision_mode = param_training["stage2"].get("supervision", "geq").lower()
    if supervision_mode not in {"geq", "feq", "exact_macro", "macro"}:
        raise ValueError(f"Unsupported stage-2 supervision mode: {supervision_mode}")
    learn_target = resolve_stage_target(param_training["stage2"])
    if learn_target == "both" and supervision_mode not in {"exact_macro", "macro"}:
        raise ValueError("learn_target=both is only supported for macro-based stage-2 supervision.")
    use_analytic_feq = learn_target == "geq"
    use_analytic_geq = learn_target == "feq"
    needs_model_feq_base = False
    needs_model_geq_base = learn_target in {"geq", "both"} and model_config["geq_mode"] == "constrained"

    if "TVD" in param_training["stage2"] and not args.disable_tvd:
        args.TVD = True


    print(f"TVD Enabled: {args.TVD}")

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

 
    val_dataset = SodDataset_stage2(F = all_F[args.num_samples:args.num_samples+100],
                                    G=all_G[args.num_samples:args.num_samples+100],
                                    Feq=all_Feq[args.num_samples:args.num_samples+100],
                                    Geq=all_Geq[args.num_samples:args.num_samples+100],
                                    dtype=dtype,)

    if supervision_mode in {"exact_macro", "macro"}:
        exact_steps = args.num_samples + len(val_dataset) + 1
        exact_rho, exact_ux, exact_uy, exact_T = build_exact_macro_rollout(
            case_params,
            exact_steps,
            backend="torch",
            dtype=dtype_name_from_torch(dtype),
            device=device,
        )
    
    model = NeurDE(
        alpha_layer=[4] + [param_training["hidden_dim"]] * param_training["num_layers"],
        phi_layer=[2] + [param_training["hidden_dim"]] * param_training["num_layers"],
        activation='relu',
        learn_feq=learn_target in {"feq", "both"},
        learn_geq=learn_target in {"geq", "both"},
        feq_mode=model_config["feq_mode"],
        geq_mode=model_config["geq_mode"],
        cv=1.0 / (case_params["vuy"] - 1.0),
        logit_clip=model_config["logit_clip"],
        newton_iters=model_config["newton_iters"],
        newton_tolerance=model_config["newton_tolerance"],
    ).to(device=device, dtype=dtype)

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
    
    Uax, Uay = case_params["Uax"], case_params["Uay"]
    basis = create_basis(Uax, Uay, device, dtype=dtype)

    epochs = param_training["stage2"]["epochs"]
    loss_func = calculate_batch_relative_error

    print(
        f"Training Case {args.case} on {device}. Epochs: {epochs}, "
        f"Samples: {args.num_samples}, supervision={supervision_mode}, learn_target={learn_target}"
    )

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
            F_seq = F_seq.to(device=device, dtype=dtype)
            G_seq = G_seq.to(device=device, dtype=dtype)
            if supervision_mode == "geq":
                Geq_seq = Geq_seq.to(device=device, dtype=dtype)
            elif supervision_mode == "feq":
                Feq_seq = Feq_seq.to(device=device, dtype=dtype)
            batch_size = F_seq.shape[0]
            batch_start_indices = torch.arange(
                batch_idx * stage2_batch_size,
                batch_idx * stage2_batch_size + batch_size,
                device=device,
                dtype=torch.long,
            )
            Fi0 = F_seq[:, 0, ...]
            Gi0 = G_seq[:, 0, ...]
            total_loss = torch.zeros((), device=device, dtype=dtype)
            if args.TVD:
                ux_old = None
                T_old = None
                rho_old = None
            khi = None
            zetax = None
            zetay = None
            for rollout in range(number_of_rollout):
                rho, ux, uy, E = sod_solver.get_macroscopic(Fi0, Gi0)
                T = sod_solver.get_temp_from_energy(ux, uy, E)
                inputs = torch.stack([rho, ux, uy, T], dim=1)
                Feq_base = None
                Geq_base = None
                if use_analytic_feq or needs_model_feq_base:
                    Feq_base = sod_solver.get_Feq(rho, ux, uy, T)
                if use_analytic_geq or needs_model_geq_base:
                    if khi is None:
                        khi = torch.zeros_like(ux)
                        zetax = torch.zeros_like(ux)
                        zetay = torch.zeros_like(ux)
                    Geq_base, khi, zetax, zetay = sod_solver.get_Geq_Newton_solver(
                        rho,
                        ux,
                        uy,
                        T,
                        khi,
                        zetax,
                        zetay,
                    )
                equilibrium_pred = model(
                    inputs,
                    basis,
                    feq_base=Feq_base if needs_model_feq_base else None,
                    geq_base=Geq_base if needs_model_geq_base else None,
                )
                if learn_target == "both":
                    Feq_pred_flat, Geq_pred_flat = equilibrium_pred
                    Feq = Feq_pred_flat.reshape(batch_size, sod_solver.Y, sod_solver.X, sod_solver.Qn).permute(0, 3, 1, 2)
                    Geq = Geq_pred_flat.reshape(batch_size, sod_solver.Y, sod_solver.X, sod_solver.Qn).permute(0, 3, 1, 2)
                else:
                    equilibrium_pred_flat = equilibrium_pred
                    equilibrium_pred = equilibrium_pred_flat.reshape(batch_size, sod_solver.Y, sod_solver.X, sod_solver.Qn).permute(0, 3, 1, 2)
                    if learn_target == "geq":
                        Feq = Feq_base
                        Geq = equilibrium_pred
                    else:
                        Feq = equilibrium_pred
                        Geq = Geq_base
                Fi_next, Gi_next = sod_solver.collision(Fi0, Gi0, Feq, Geq, rho, ux, uy, T)
                Fi_next, Gi_next = sod_solver.streaming(Fi_next, Gi_next)

                if supervision_mode in {"exact_macro", "macro"}:
                    rho_next, ux_next, uy_next, E_next = sod_solver.get_macroscopic(Fi_next, Gi_next)
                    T_next = sod_solver.get_temp_from_energy(ux_next, uy_next, E_next)
                    target_indices = batch_start_indices + rollout + 1
                    target_macro = torch.stack(
                        [
                            exact_rho[target_indices],
                            exact_ux[target_indices],
                            exact_uy[target_indices],
                            exact_T[target_indices],
                        ],
                        dim=1,
                    )
                    pred_macro = torch.stack([rho_next, ux_next, uy_next, T_next], dim=1)
                    total_loss = total_loss + loss_func(pred_macro, target_macro)
                elif supervision_mode == "feq":
                    Feq_target = Feq_seq[:, rollout]
                    pred_batch = equilibrium_pred_flat.reshape(batch_size, sod_solver.Y * sod_solver.X, sod_solver.Qn)
                    target_batch = Feq_target.permute(0, 2, 3, 1).reshape(batch_size, sod_solver.Y * sod_solver.X, sod_solver.Qn)
                    total_loss = total_loss + loss_func(pred_batch, target_batch)
                else:
                    Geq_target = Geq_seq[:, rollout]
                    pred_batch = equilibrium_pred_flat.reshape(batch_size, sod_solver.Y * sod_solver.X, sod_solver.Qn)
                    target_batch = Geq_target.permute(0, 2, 3, 1).reshape(batch_size, sod_solver.Y * sod_solver.X, sod_solver.Qn)
                    total_loss = total_loss + loss_func(pred_batch, target_batch)

                if args.TVD and ux_old is not None:
                    loss_TVD = TVD_norm(T, T_old) + TVD_norm(ux, ux_old) + TVD_norm(rho, rho_old)
                    total_loss = total_loss + tvd_weight * loss_TVD
                if args.TVD:
                    ux_old = ux.clone()
                    T_old = T.clone()
                    rho_old = rho.clone()

                Fi0, Gi0 = Fi_next, Gi_next
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
            Fi0 = next(iter(val_dataset))[0].to(device).unsqueeze(0)
            Gi0 = next(iter(val_dataset))[1].to(device).unsqueeze(0)
            khi = None
            zetax = None
            zetay = None
            
            for val_idx, (F_val, G_val, Feq_val, Geq_val) in enumerate(val_dataset):
                rho, ux, uy, E = sod_solver.get_macroscopic(Fi0, Gi0)
                T = sod_solver.get_temp_from_energy(ux, uy, E)
                inputs = torch.stack([rho, ux, uy, T], dim=1)
                Feq_base = None
                Geq_base = None
                if use_analytic_feq or needs_model_feq_base:
                    Feq_base = sod_solver.get_Feq(rho, ux, uy, T)
                if use_analytic_geq or needs_model_geq_base:
                    if khi is None:
                        khi = torch.zeros_like(ux)
                        zetax = torch.zeros_like(ux)
                        zetay = torch.zeros_like(ux)
                    Geq_base, khi, zetax, zetay = sod_solver.get_Geq_Newton_solver(
                        rho,
                        ux,
                        uy,
                        T,
                        khi,
                        zetax,
                        zetay,
                    )
                equilibrium_pred = model(
                    inputs,
                    basis,
                    feq_base=Feq_base if needs_model_feq_base else None,
                    geq_base=Geq_base if needs_model_geq_base else None,
                )
                if learn_target == "both":
                    Feq_pred_flat, Geq_pred_flat = equilibrium_pred
                    Feq = Feq_pred_flat.reshape(1, sod_solver.Y, sod_solver.X, sod_solver.Qn).permute(0, 3, 1, 2)
                    Geq = Geq_pred_flat.reshape(1, sod_solver.Y, sod_solver.X, sod_solver.Qn).permute(0, 3, 1, 2)
                else:
                    equilibrium_pred_flat = equilibrium_pred
                    equilibrium_pred = equilibrium_pred_flat.reshape(1, sod_solver.Y, sod_solver.X, sod_solver.Qn).permute(0, 3, 1, 2)
                    if learn_target == "geq":
                        Feq = Feq_base
                        Geq = equilibrium_pred
                    else:
                        Feq = equilibrium_pred
                        Geq = Geq_base
                Fi_next, Gi_next = sod_solver.collision(Fi0, Gi0, Feq, Geq, rho, ux, uy, T)
                Fi_next, Gi_next = sod_solver.streaming(Fi_next, Gi_next)

                if supervision_mode in {"exact_macro", "macro"}:
                    rho_next, ux_next, uy_next, E_next = sod_solver.get_macroscopic(Fi_next, Gi_next)
                    T_next = sod_solver.get_temp_from_energy(ux_next, uy_next, E_next)
                    target_index = args.num_samples + val_idx + 1
                    target_macro = torch.stack(
                        [
                            exact_rho[target_index],
                            exact_ux[target_index],
                            exact_uy[target_index],
                            exact_T[target_index],
                        ],
                        dim=0,
                    ).unsqueeze(0)
                    pred_macro = torch.stack([rho_next, ux_next, uy_next, T_next], dim=1)
                    val_loss += loss_func(pred_macro, target_macro)
                elif supervision_mode == "feq":
                    Feq_target = Feq_val.to(device).unsqueeze(0)
                    pred_batch = equilibrium_pred_flat.reshape(1, sod_solver.Y * sod_solver.X, sod_solver.Qn)
                    target_batch = Feq_target.permute(0, 2, 3, 1).reshape(1, sod_solver.Y * sod_solver.X, sod_solver.Qn)
                    val_loss += loss_func(pred_batch, target_batch)
                else:
                    Geq_target = Geq_val.to(device).unsqueeze(0)
                    pred_batch = equilibrium_pred_flat.reshape(1, sod_solver.Y * sod_solver.X, sod_solver.Qn)
                    target_batch = Geq_target.permute(0, 2, 3, 1).reshape(1, sod_solver.Y * sod_solver.X, sod_solver.Qn)
                    val_loss += loss_func(pred_batch, target_batch)

                Fi0, Gi0 = Fi_next, Gi_next
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
