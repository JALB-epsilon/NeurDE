import argparse
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from architectures.model_constrained_geq import NeurDEConstrainedGeq
from exact_solution import build_exact_macro_rollout
from SOD_solver import SODSolver
from train_stage_1 import create_basis
from utilities import *


def predict_constrained_geq(model, sod_solver, basis, rho, ux, uy, T, khi, zetax, zetay):
    if khi is None:
        khi = np.zeros((sod_solver.Y, sod_solver.X), dtype=np.float32)
        zetax = np.zeros((sod_solver.Y, sod_solver.X), dtype=np.float32)
        zetay = np.zeros((sod_solver.Y, sod_solver.X), dtype=np.float32)
    geq_base, khi, zetax, zetay = sod_solver.get_Geq_Newton_solver(rho, ux, uy, T, khi, zetax, zetay)
    inputs = torch.stack([rho.unsqueeze(0), ux.unsqueeze(0), uy.unsqueeze(0), T.unsqueeze(0)], dim=1).to(sod_solver.device)
    geq_pred_flat = model(inputs, basis, geq_base=geq_base)
    geq_pred = geq_pred_flat.permute(1, 0).reshape(sod_solver.Qn, sod_solver.Y, sod_solver.X)
    return geq_pred, khi, zetax, zetay


if __name__ == "__main__":
    set_seed(0)

    parser = argparse.ArgumentParser(description="Train Stage 2 constrained Geq against exact macros")
    parser.add_argument("--device", type=int, default=3)
    parser.add_argument("--case", type=int, choices=[1, 2], default=None)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--save_frequency", type=int, default=1)
    parser.add_argument("--rollout_override", type=int, default=4)
    parser.add_argument("--epochs_override", type=int, default=100)
    parser.add_argument("--lr_override", type=float, default=1.0e-5)
    parser.add_argument("--pre_trained_path", type=str, required=True)
    parser.add_argument("--model_dir_override", type=str, required=True)
    parser.add_argument("--disable_tvd", action="store_true")
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

    with open("Sod_cases_param_training.yml", "r") as stream:
        training_config = yaml.safe_load(stream)
    param_training = training_config[args.case]
    param_training["stage2"]["epochs"] = int(args.epochs_override)
    param_training["stage2"]["N"] = int(args.rollout_override)
    param_training["stage2"]["lr"] = float(args.lr_override)
    param_training["stage2"]["model_dir"] = str(args.model_dir_override)
    os.makedirs(param_training["stage2"]["model_dir"], exist_ok=True)
    use_tvd = "TVD" in param_training["stage2"] and not args.disable_tvd

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

    all_F, all_G, all_Feq, all_Geq = load_data_stage_2(param_training["data_dir"])
    dataset = RolloutBatchDataset(
        all_Fi=all_F[: args.num_samples],
        all_Gi=all_G[: args.num_samples],
        all_Feq=all_Feq[: args.num_samples],
        all_Geq=all_Geq[: args.num_samples],
        number_of_rollout=param_training["stage2"]["N"],
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    val_dataset = SodDataset_stage2(
        F=all_F[args.num_samples : args.num_samples + 100],
        G=all_G[args.num_samples : args.num_samples + 100],
        Feq=all_Feq[args.num_samples : args.num_samples + 100],
        Geq=all_Geq[args.num_samples : args.num_samples + 100],
    )
    exact_steps = args.num_samples + len(val_dataset) + 1
    exact_rho, exact_ux, exact_uy, exact_T = build_exact_macro_rollout(
        args.case,
        case_params["X"],
        case_params["Y"],
        exact_steps,
        device,
    )

    model = NeurDEConstrainedGeq(
        alpha_layer=[4] + [param_training["hidden_dim"]] * param_training["num_layers"],
        phi_layer=[2] + [param_training["hidden_dim"]] * param_training["num_layers"],
        activation="relu",
        cv=1.0 / (case_params["vuy"] - 1.0),
    ).to(device)

    checkpoint = torch.load(args.pre_trained_path, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"Pre-trained constrained model loaded from {args.pre_trained_path}")

    optimizer = dispatch_optimizer(model=model, lr=param_training["stage2"]["lr"], optimizer_type="AdamW")
    total_steps = len(dataloader) * param_training["stage2"]["epochs"]
    scheduler_type = param_training["stage2"]["scheduler"]
    scheduler_config = param_training["stage2"].get("scheduler_config", {}).get(scheduler_type, {})
    scheduler = get_scheduler(optimizer, scheduler_type, total_steps, scheduler_config)
    basis = create_basis(case_params["Uax"], case_params["Uay"], device)
    loss_func = calculate_relative_error

    print(
        f"Training constrained exact-macro Case {args.case} on {device}. "
        f"Epochs: {param_training['stage2']['epochs']}, rollout={param_training['stage2']['N']}"
    )
    if use_tvd:
        print("Using TVD")

    best_loss = float("inf")
    best_path = None
    for epoch in tqdm(range(param_training["stage2"]["epochs"]), desc="Epochs"):
        loss_epoch = 0.0

        for batch_idx, (F_seq, G_seq, _, _) in enumerate(dataloader):
            optimizer.zero_grad()
            F_seq = F_seq.to(device)
            G_seq = G_seq.to(device)
            total_loss = torch.zeros((), device=device)
            batch_size = F_seq.shape[0]
            batch_start = batch_idx * args.batch_size
            for sample_idx in range(batch_size):
                Fi0 = F_seq[sample_idx, 0, ...]
                Gi0 = G_seq[sample_idx, 0, ...]
                khi = None
                zetax = None
                zetay = None
                ux_old = None
                T_old = None
                rho_old = None
                tvd_weight = 15.0

                for rollout in range(param_training["stage2"]["N"]):
                    rho, ux, uy, E = sod_solver.get_macroscopic(Fi0, Gi0)
                    T = sod_solver.get_temp_from_energy(ux, uy, E)
                    Feq = sod_solver.get_Feq(rho, ux, uy, T)
                    Geq_pred, khi, zetax, zetay = predict_constrained_geq(
                        model, sod_solver, basis, rho, ux, uy, T, khi, zetax, zetay
                    )
                    Fi0, Gi0 = sod_solver.collision(Fi0, Gi0, Feq, Geq_pred, rho, ux, uy, T)
                    Fi, Gi = sod_solver.streaming(Fi0, Gi0)
                    rho_next, ux_next, uy_next, E_next = sod_solver.get_macroscopic(Fi, Gi)
                    T_next = sod_solver.get_temp_from_energy(ux_next, uy_next, E_next)
                    target_idx = batch_start + sample_idx + rollout + 1
                    pred_macro = torch.stack([rho_next, ux_next, uy_next, T_next], dim=0)
                    target_macro = torch.stack(
                        [exact_rho[target_idx], exact_ux[target_idx], exact_uy[target_idx], exact_T[target_idx]],
                        dim=0,
                    )
                    total_loss = total_loss + loss_func(pred_macro, target_macro)
                    if use_tvd and ux_old is not None:
                        total_loss = total_loss + tvd_weight * (TVD_norm(T, T_old) + TVD_norm(ux, ux_old) + TVD_norm(rho, rho_old))
                    if use_tvd:
                        ux_old = ux.clone()
                        T_old = T.clone()
                        rho_old = rho.clone()
                    Fi0 = Fi
                    Gi0 = Gi

            total_loss = total_loss / batch_size

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_epoch += float(total_loss.item())

        scheduler.step()
        current_loss = loss_epoch / len(dataloader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            Fi0 = val_dataset[0][0].to(device)
            Gi0 = val_dataset[0][1].to(device)
            khi = None
            zetax = None
            zetay = None
            for step, _ in enumerate(val_dataset):
                rho, ux, uy, E = sod_solver.get_macroscopic(Fi0, Gi0)
                T = sod_solver.get_temp_from_energy(ux, uy, E)
                Feq = sod_solver.get_Feq(rho, ux, uy, T)
                Geq_pred, khi, zetax, zetay = predict_constrained_geq(
                    model, sod_solver, basis, rho, ux, uy, T, khi, zetax, zetay
                )
                Fi0, Gi0 = sod_solver.collision(Fi0, Gi0, Feq, Geq_pred, rho, ux, uy, T)
                Fi, Gi = sod_solver.streaming(Fi0, Gi0)
                rho_next, ux_next, uy_next, E_next = sod_solver.get_macroscopic(Fi, Gi)
                T_next = sod_solver.get_temp_from_energy(ux_next, uy_next, E_next)
                target_idx = args.num_samples + step + 1
                pred_macro = torch.stack([rho_next, ux_next, uy_next, T_next], dim=0)
                target_macro = torch.stack(
                    [exact_rho[target_idx], exact_ux[target_idx], exact_uy[target_idx], exact_T[target_idx]],
                    dim=0,
                )
                val_loss += float(loss_func(pred_macro, target_macro).item())
                Fi0 = Fi
                Gi0 = Gi
            val_loss /= len(val_dataset)
            print(f"Validation Loss: {val_loss:.6f}")

        if val_loss < best_loss and (epoch + 1) % args.save_frequency == 0:
            if best_path and os.path.exists(best_path):
                os.remove(best_path)
            best_loss = val_loss
            best_path = os.path.join(
                param_training["stage2"]["model_dir"],
                f"best_model_{args.case}_epoch_{epoch + 1}_val_loss_{val_loss:.6f}.pt",
            )
            torch.save(model.state_dict(), best_path)
            print(f"Best constrained stage-2 model saved to: {best_path}")

    last_model_path = os.path.join(
        param_training["stage2"]["model_dir"],
        f"last_model_{args.case}_epoch_{param_training['stage2']['epochs']}_loss_{current_loss:.6f}.pt",
    )
    torch.save(model.state_dict(), last_model_path)
    print(f"Last model saved to: {last_model_path}")
