import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from architectures import NeurDE
from SOD_solver import SODSolver
from train_stage_1 import create_basis
from utilities import *


if __name__ == "__main__":
    set_seed(0)

    parser = argparse.ArgumentParser(description="Train Stage 2 main-style Geq baseline")
    parser.add_argument("--device", type=int, default=3, help="Device index")
    parser.add_argument("--case", type=int, choices=[1, 2], default=None)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--save_frequency", type=int, default=1)
    parser.add_argument("--pre_trained_path", type=str, required=True)
    parser.add_argument("--epochs_override", type=int, default=None)
    parser.add_argument("--model_dir_override", type=str, default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    if args.case is None:
        case_part = args.pre_trained_path.replace("SOD_shock_tube/", "").split("/")[1]
        case_number = "".join(filter(str.isdigit, case_part))
        args.case = int(case_number)

    with open("Sod_cases_param.yml", "r") as stream:
        config = yaml.safe_load(stream)
    case_params = config[args.case]
    case_params["device"] = device

    print(f"Case {args.case}: SOD shock tube problem")

    sod_solver = SODSolver(
        X=case_params["X"],
        Y=case_params["Y"],
        Qn=case_params["Qn"],
        alpha1=case_params["alpha1"],
        alpha01=case_params["alpha01"],
        vuy=case_params["vuy"],
        Pr=case_params["Pr"],
        muy=case_params["muy"],
        Uax=case_params["Uax"],
        Uay=case_params["Uay"],
        device=case_params["device"],
    )

    with open("Sod_cases_param_training.yml", "r") as stream:
        training_config = yaml.safe_load(stream)
    param_training = training_config[args.case]
    if args.epochs_override is not None:
        param_training["stage2"]["epochs"] = int(args.epochs_override)
    if args.model_dir_override is not None:
        param_training["stage2"]["model_dir"] = str(args.model_dir_override)
    number_of_rollout = param_training["stage2"]["N"]
    use_tvd = "TVD" in param_training["stage2"]

    print(f"TVD Enabled: {use_tvd}")

    os.makedirs(param_training["stage2"]["model_dir"], exist_ok=True)
    all_F, all_G, all_Feq, all_Geq = load_data_stage_2(param_training["data_dir"])
    dataset = RolloutBatchDataset(
        all_Fi=all_F[: args.num_samples],
        all_Gi=all_G[: args.num_samples],
        all_Feq=all_Feq[: args.num_samples],
        all_Geq=all_Geq[: args.num_samples],
        number_of_rollout=number_of_rollout,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    val_dataset = SodDataset_stage2(
        F=all_F[args.num_samples : args.num_samples + 100],
        G=all_G[args.num_samples : args.num_samples + 100],
        Feq=all_Feq[args.num_samples : args.num_samples + 100],
        Geq=all_Geq[args.num_samples : args.num_samples + 100],
    )

    model = NeurDE(
        alpha_layer=[4] + [param_training["hidden_dim"]] * param_training["num_layers"],
        phi_layer=[2] + [param_training["hidden_dim"]] * param_training["num_layers"],
        activation="relu",
    ).to(device)

    checkpoint = torch.load(args.pre_trained_path, map_location=device)
    new_state_dict = {}
    for key, value in checkpoint.items():
        if key.startswith("_orig_mod."):
            new_state_dict[key.replace("_orig_mod.", "")] = value
        else:
            new_state_dict[key] = value
    model.load_state_dict(new_state_dict)
    print(f"Pre-trained model loaded from {args.pre_trained_path}")

    optimizer = dispatch_optimizer(model=model, lr=param_training["stage2"]["lr"], optimizer_type="AdamW")
    total_steps = len(dataloader) * param_training["stage2"]["epochs"]
    scheduler_type = param_training["stage2"]["scheduler"]
    scheduler_config = param_training["stage2"].get("scheduler_config", {}).get(scheduler_type, {})
    scheduler = get_scheduler(optimizer, scheduler_type, total_steps, scheduler_config)

    basis = create_basis(case_params["Uax"], case_params["Uay"], device)
    epochs = param_training["stage2"]["epochs"]
    loss_func = calculate_relative_error

    print(f"Training Case {args.case} on {device}. Epochs: {epochs}, Samples: {args.num_samples}")

    best_losses = [float("inf")] * 3
    best_models = [None] * 3
    best_model_paths = [None] * 3
    epochs_since_last_save = [0] * 3

    if use_tvd:
        print("Using TVD")

    for epoch in tqdm(range(epochs), desc="Epochs"):
        loss_epoch = 0.0
        for batch_idx, (F_seq, G_seq, Feq_seq, Geq_seq) in enumerate(dataloader):
            optimizer.zero_grad()
            model.train()
            total_loss = torch.zeros((), device=device)
            F_seq = F_seq.to(device)
            G_seq = G_seq.to(device)
            batch_size = F_seq.shape[0]
            for sample_idx in range(batch_size):
                Fi0 = F_seq[sample_idx, 0, ...]
                Gi0 = G_seq[sample_idx, 0, ...]
                ux_old = None
                T_old = None
                rho_old = None
                tvd_weight = 15.0
                for rollout in range(number_of_rollout):
                    rho, ux, uy, E = sod_solver.get_macroscopic(Fi0, Gi0)
                    T = sod_solver.get_temp_from_energy(ux, uy, E)
                    Feq = sod_solver.get_Feq(rho, ux, uy, T)
                    inputs = torch.stack([rho.unsqueeze(0), ux.unsqueeze(0), uy.unsqueeze(0), T.unsqueeze(0)], dim=1).to(device)
                    geq_pred = model(inputs, basis)
                    geq_target = Geq_seq[sample_idx, rollout].to(device)
                    total_loss = total_loss + loss_func(geq_pred, geq_target.permute(1, 2, 0).reshape(-1, 9))
                    if use_tvd and ux_old is not None:
                        total_loss = total_loss + tvd_weight * (TVD_norm(T, T_old) + TVD_norm(ux, ux_old) + TVD_norm(rho, rho_old))
                    if use_tvd:
                        ux_old = ux.clone()
                        T_old = T.clone()
                        rho_old = rho.clone()
                    Fi0, Gi0 = sod_solver.collision(
                        Fi0,
                        Gi0,
                        Feq,
                        geq_pred.permute(1, 0).reshape(sod_solver.Qn, sod_solver.Y, sod_solver.X),
                        rho,
                        ux,
                        uy,
                        T,
                    )
                    Fi, Gi = sod_solver.streaming(Fi0, Gi0)
                    Fi0 = Fi.detach()
                    Gi0 = Gi.detach()
            total_loss = total_loss / batch_size
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_epoch += float(total_loss.item())
            print(f"Epoch: {epoch}, Batch ID: {batch_idx}, Loss: {float(total_loss.item()) / number_of_rollout:.6f}")

        scheduler.step()
        current_loss = loss_epoch / len(dataloader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            Fi0 = next(iter(val_dataset))[0].to(device)
            Gi0 = next(iter(val_dataset))[1].to(device)
            for _, _, _, Geq_val in val_dataset:
                rho, ux, uy, E = sod_solver.get_macroscopic(Fi0, Gi0)
                T = sod_solver.get_temp_from_energy(ux, uy, E)
                Feq = sod_solver.get_Feq(rho, ux, uy, T)
                inputs = torch.stack([rho.unsqueeze(0), ux.unsqueeze(0), uy.unsqueeze(0), T.unsqueeze(0)], dim=1).to(device)
                geq_pred = model(inputs, basis)
                geq_target = Geq_val.to(device)
                val_loss += float(loss_func(geq_pred, geq_target.permute(1, 2, 0).reshape(-1, 9)).item())
                Fi0, Gi0 = sod_solver.collision(
                    Fi0,
                    Gi0,
                    Feq,
                    geq_pred.permute(1, 0).reshape(sod_solver.Qn, sod_solver.Y, sod_solver.X),
                    rho,
                    ux,
                    uy,
                    T,
                )
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
            if epochs_since_last_save[max_index] >= args.save_frequency:
                if best_model_paths[max_index] and os.path.exists(best_model_paths[max_index]):
                    os.remove(best_model_paths[max_index])
                save_path = os.path.join(
                    param_training["stage2"]["model_dir"],
                    f"best_model_{args.case}_epoch_{epoch + 1}_top_{max_index + 1}_val_loss_{val_loss:.6f}.pt",
                )
                torch.save(best_models[max_index], save_path)
                print(f"Top {max_index + 1} model saved to: {save_path}")
                best_model_paths[max_index] = save_path
                epochs_since_last_save[max_index] = 0
            else:
                epochs_since_last_save[max_index] += 1
        else:
            for index in range(3):
                epochs_since_last_save[index] += 1

        if epoch % 200 == 0:
            save_path = os.path.join(
                param_training["stage2"]["model_dir"],
                f"model_{args.case}_epoch_{epoch}_loss_{val_loss:.6f}.pt",
            )
            torch.save(model.state_dict(), save_path)

    last_model_path = os.path.join(
        param_training["stage2"]["model_dir"],
        f"last_model_{args.case}_epoch_{epochs}_loss_{current_loss:.6f}.pt",
    )
    torch.save(model.state_dict(), last_model_path)
    print(f"Last model saved to: {last_model_path}")
    print("Training complete.")
