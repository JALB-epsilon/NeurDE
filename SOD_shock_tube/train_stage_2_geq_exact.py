import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from architectures import NeurDE
from exact_solution import build_exact_macro_rollout
from SOD_solver_batch import SODBatchSolver
from train_stage_1 import create_basis
from utilities import *


def reshape_population_batch(population_flat, batch_size, qn, y_nodes, x_nodes):
    return population_flat.reshape(batch_size, y_nodes, x_nodes, qn).permute(0, 3, 1, 2).contiguous()


def batch_relative_error(pred, target):
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    eps = 1.0e-7
    return torch.mean(
        torch.linalg.vector_norm(pred_flat - target_flat, dim=1)
        / (torch.linalg.vector_norm(target_flat, dim=1) + eps)
    )


def normalized_population_shape(population, eps=1.0e-12):
    population = population.clamp_min(eps)
    return population / population.sum(dim=1, keepdim=True).clamp_min(eps)


def levermore_shape_kl(pred_population, base_population, eps=1.0e-12):
    pred_shape = normalized_population_shape(pred_population, eps=eps)
    base_shape = normalized_population_shape(base_population, eps=eps)
    return (base_shape * (base_shape.log() - pred_shape.log())).sum(dim=1).mean()


def energy_moment_relative_error(population, rho, ux, uy, T, cv, eps=1.0e-12):
    target_energy = 2.0 * rho * (cv * T + 0.5 * (ux.square() + uy.square()))
    actual_energy = population.sum(dim=1)
    return ((actual_energy - target_energy).abs() / (target_energy.abs() + eps)).mean()


def linear_decay_weight(epoch, base_weight, decay_epochs):
    if base_weight <= 0.0:
        return 0.0
    if decay_epochs <= 0:
        return float(base_weight)
    decay = max(0.0, 1.0 - float(epoch) / float(decay_epochs))
    return float(base_weight) * decay


def predict_geq(model, sod_solver, basis, rho, ux, uy, T, geq_mode, khi=None, zetax=None, zetay=None):
    inputs = torch.stack([rho, ux, uy, T], dim=1).to(sod_solver.device)
    model_kwargs = {}
    if geq_mode in {"hybrid", "hybrid_energy"}:
        geq_base, khi, zetax, zetay = sod_solver.get_Geq_Newton_solver_batch(rho, ux, uy, T, khi, zetax, zetay)
        model_kwargs["geq_base"] = geq_base
    geq_pred_flat = model(inputs, basis, **model_kwargs)
    geq_pred = reshape_population_batch(geq_pred_flat, rho.shape[0], sod_solver.Qn, sod_solver.Y, sod_solver.X)
    return geq_pred, khi, zetax, zetay


if __name__ == "__main__":
    set_seed(0)

    parser = argparse.ArgumentParser(description="Train Stage 2 Geq against exact macro rollout targets")
    parser.add_argument("--device", type=int, default=3)
    parser.add_argument("--case", type=int, choices=[1, 2], default=None)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_frequency", type=int, default=1)
    parser.add_argument("--rollout_override", type=int, default=4)
    parser.add_argument("--epochs_override", type=int, default=100)
    parser.add_argument("--lr_override", type=float, default=1.0e-5)
    parser.add_argument("--pre_trained_path", type=str, required=True)
    parser.add_argument("--model_dir_override", type=str, required=True)
    parser.add_argument("--disable_tvd", action="store_true")
    parser.add_argument(
        "--geq_mode",
        type=str,
        default="hybrid",
        choices=["positive", "positive_energy", "hybrid", "hybrid_energy", "energy_projected"],
    )
    parser.add_argument("--logit_clip", type=float, default=15.0)
    parser.add_argument("--shape_anchor_weight", type=float, default=0.0)
    parser.add_argument("--shape_anchor_decay_epochs", type=int, default=0)
    parser.add_argument("--exact_geq_teacher_weight", type=float, default=0.0)
    parser.add_argument("--exact_geq_teacher_decay_epochs", type=int, default=0)
    parser.add_argument("--soft_energy_weight", type=float, default=0.0)
    parser.add_argument("--soft_energy_decay_epochs", type=int, default=0)
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

    sod_solver = SODBatchSolver(
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
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
    )
    val_dataset = RolloutBatchDataset(
        all_Fi=all_F[args.num_samples : args.num_samples + 100],
        all_Gi=all_G[args.num_samples : args.num_samples + 100],
        all_Feq=all_Feq[args.num_samples : args.num_samples + 100],
        all_Geq=all_Geq[args.num_samples : args.num_samples + 100],
        number_of_rollout=param_training["stage2"]["N"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
    )
    exact_steps = args.num_samples + 100 + param_training["stage2"]["N"]
    exact_rho, exact_ux, exact_uy, exact_T = build_exact_macro_rollout(
        args.case,
        case_params["X"],
        case_params["Y"],
        exact_steps,
        device,
    )

    model = NeurDE(
        alpha_layer=[4] + [param_training["hidden_dim"]] * param_training["num_layers"],
        phi_layer=[2] + [param_training["hidden_dim"]] * param_training["num_layers"],
        activation="relu",
        geq_mode=args.geq_mode,
        cv=1.0 / (case_params["vuy"] - 1.0),
        logit_clip=args.logit_clip,
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
    step_scheduler_per_batch = scheduler_type == "OneCycleLR"
    step_scheduler_on_plateau = scheduler_type == "ReduceLROnPlateau"
    basis = create_basis(case_params["Uax"], case_params["Uay"], device)
    print(
        f"Training exact-macro Case {args.case} on {device}. Epochs: {param_training['stage2']['epochs']}, "
        f"rollout={param_training['stage2']['N']}, geq_mode={args.geq_mode}"
    )
    if use_tvd:
        print("Using TVD")
    if args.shape_anchor_weight > 0.0:
        decay_desc = args.shape_anchor_decay_epochs if args.shape_anchor_decay_epochs > 0 else "constant"
        print(f"Using Levermore shape anchor: weight={args.shape_anchor_weight}, decay_epochs={decay_desc}")
    if args.exact_geq_teacher_weight > 0.0:
        decay_desc = args.exact_geq_teacher_decay_epochs if args.exact_geq_teacher_decay_epochs > 0 else "constant"
        print(f"Using exact-Sod Geq teacher: weight={args.exact_geq_teacher_weight}, decay_epochs={decay_desc}")
    if args.soft_energy_weight > 0.0:
        decay_desc = args.soft_energy_decay_epochs if args.soft_energy_decay_epochs > 0 else "constant"
        print(f"Using soft energy penalty: weight={args.soft_energy_weight}, decay_epochs={decay_desc}")

    best_loss = float("inf")
    best_path = None
    for epoch in tqdm(range(param_training["stage2"]["epochs"]), desc="Epochs"):
        loss_epoch = 0.0
        shape_anchor_weight = linear_decay_weight(
            epoch=epoch,
            base_weight=args.shape_anchor_weight,
            decay_epochs=args.shape_anchor_decay_epochs,
        )
        exact_geq_teacher_weight = linear_decay_weight(
            epoch=epoch,
            base_weight=args.exact_geq_teacher_weight,
            decay_epochs=args.exact_geq_teacher_decay_epochs,
        )
        soft_energy_weight = linear_decay_weight(
            epoch=epoch,
            base_weight=args.soft_energy_weight,
            decay_epochs=args.soft_energy_decay_epochs,
        )

        for batch_idx, (F_seq, G_seq, _, _) in enumerate(dataloader):
            optimizer.zero_grad()
            F_seq = F_seq.to(device)
            G_seq = G_seq.to(device)
            total_loss = torch.zeros((), device=device)
            batch_size = F_seq.shape[0]
            batch_start = batch_idx * args.batch_size
            batch_indices = torch.arange(batch_start, batch_start + batch_size, device=device, dtype=torch.long)
            Fi0 = F_seq[:, 0, ...]
            Gi0 = G_seq[:, 0, ...]
            khi = None
            zetax = None
            zetay = None
            ux_old = None
            T_old = None
            rho_old = None
            tvd_weight = 15.0
            exact_khi = None
            exact_zetax = None
            exact_zetay = None

            for rollout in range(param_training["stage2"]["N"]):
                rho, ux, uy, E = sod_solver.get_macroscopic(Fi0, Gi0)
                T = sod_solver.get_temp_from_energy(ux, uy, E)
                Feq = sod_solver.get_Feq(rho, ux, uy, T)
                Geq_pred, khi, zetax, zetay = predict_geq(
                    model, sod_solver, basis, rho, ux, uy, T, args.geq_mode, khi, zetax, zetay
                )
                if shape_anchor_weight > 0.0 and args.geq_mode == "positive_energy":
                    with torch.no_grad():
                        geq_anchor, khi, zetax, zetay = sod_solver.get_Geq_Newton_solver_batch(
                            rho, ux, uy, T, khi, zetax, zetay
                        )
                    total_loss = total_loss + shape_anchor_weight * levermore_shape_kl(Geq_pred, geq_anchor)
                if soft_energy_weight > 0.0 and args.geq_mode == "positive":
                    total_loss = total_loss + soft_energy_weight * energy_moment_relative_error(
                        Geq_pred,
                        rho,
                        ux,
                        uy,
                        T,
                        sod_solver.Cv,
                    )
                current_idx = batch_indices + rollout
                if exact_geq_teacher_weight > 0.0 and args.geq_mode in {"positive", "positive_energy"}:
                    exact_rho_step = exact_rho[current_idx]
                    exact_ux_step = exact_ux[current_idx]
                    exact_uy_step = exact_uy[current_idx]
                    exact_T_step = exact_T[current_idx]
                    with torch.no_grad():
                        exact_geq_target, exact_khi, exact_zetax, exact_zetay = sod_solver.get_Geq_Newton_solver_batch(
                            exact_rho_step,
                            exact_ux_step,
                            exact_uy_step,
                            exact_T_step,
                            exact_khi,
                            exact_zetax,
                            exact_zetay,
                        )
                    exact_geq_pred, _, _, _ = predict_geq(
                        model,
                        sod_solver,
                        basis,
                        exact_rho_step,
                        exact_ux_step,
                        exact_uy_step,
                        exact_T_step,
                        args.geq_mode,
                    )
                    total_loss = total_loss + exact_geq_teacher_weight * levermore_shape_kl(
                        exact_geq_pred,
                        exact_geq_target,
                    )
                Fi0, Gi0 = sod_solver.collision(Fi0, Gi0, Feq, Geq_pred, rho, ux, uy, T)
                Fi0, Gi0 = sod_solver.streaming(Fi0, Gi0)
                rho_next, ux_next, uy_next, E_next = sod_solver.get_macroscopic(Fi0, Gi0)
                T_next = sod_solver.get_temp_from_energy(ux_next, uy_next, E_next)
                target_idx = current_idx + 1
                pred_macro = torch.stack([rho_next, ux_next, uy_next, T_next], dim=1)
                target_macro = torch.stack(
                    [exact_rho[target_idx], exact_ux[target_idx], exact_uy[target_idx], exact_T[target_idx]],
                    dim=1,
                )
                total_loss = total_loss + batch_relative_error(pred_macro, target_macro)
                if use_tvd and ux_old is not None:
                    total_loss = total_loss + tvd_weight * (
                        TVD_norm(T, T_old) + TVD_norm(ux, ux_old) + TVD_norm(rho, rho_old)
                    )
                if use_tvd:
                    ux_old = ux.clone()
                    T_old = T.clone()
                    rho_old = rho.clone()

            total_loss = total_loss / float(param_training["stage2"]["N"])
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if step_scheduler_per_batch:
                scheduler.step()
            loss_epoch += float(total_loss.item())
        current_loss = loss_epoch / len(dataloader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (F_seq, G_seq, _, _) in enumerate(val_loader):
                F_seq = F_seq.to(device)
                G_seq = G_seq.to(device)
                batch_size = F_seq.shape[0]
                batch_start = args.num_samples + batch_idx * args.batch_size
                batch_indices = torch.arange(batch_start, batch_start + batch_size, device=device, dtype=torch.long)
                Fi0 = F_seq[:, 0, ...]
                Gi0 = G_seq[:, 0, ...]
                khi = None
                zetax = None
                zetay = None
                batch_loss = torch.zeros((), device=device)
                for rollout in range(param_training["stage2"]["N"]):
                    rho, ux, uy, E = sod_solver.get_macroscopic(Fi0, Gi0)
                    T = sod_solver.get_temp_from_energy(ux, uy, E)
                    Feq = sod_solver.get_Feq(rho, ux, uy, T)
                    Geq_pred, khi, zetax, zetay = predict_geq(
                        model, sod_solver, basis, rho, ux, uy, T, args.geq_mode, khi, zetax, zetay
                    )
                    Fi0, Gi0 = sod_solver.collision(Fi0, Gi0, Feq, Geq_pred, rho, ux, uy, T)
                    Fi0, Gi0 = sod_solver.streaming(Fi0, Gi0)
                    rho_next, ux_next, uy_next, E_next = sod_solver.get_macroscopic(Fi0, Gi0)
                    T_next = sod_solver.get_temp_from_energy(ux_next, uy_next, E_next)
                    target_idx = batch_indices + rollout + 1
                    pred_macro = torch.stack([rho_next, ux_next, uy_next, T_next], dim=1)
                    target_macro = torch.stack(
                        [exact_rho[target_idx], exact_ux[target_idx], exact_uy[target_idx], exact_T[target_idx]],
                        dim=1,
                    )
                    batch_loss = batch_loss + batch_relative_error(pred_macro, target_macro)
                batch_loss = batch_loss / float(param_training["stage2"]["N"])
                val_loss += float(batch_loss.item())
            val_loss /= len(val_loader)
            print(f"Validation Loss: {val_loss:.6f}")

        if step_scheduler_on_plateau:
            scheduler.step(val_loss)
        elif not step_scheduler_per_batch:
            scheduler.step()

        if val_loss < best_loss and (epoch + 1) % args.save_frequency == 0:
            if best_path and os.path.exists(best_path):
                os.remove(best_path)
            best_loss = val_loss
            best_path = os.path.join(
                param_training["stage2"]["model_dir"],
                f"best_model_{args.case}_epoch_{epoch + 1}_val_loss_{val_loss:.6f}.pt",
            )
            torch.save(model.state_dict(), best_path)
            print(f"Best exact-macro stage-2 model saved to: {best_path}")

    last_model_path = os.path.join(
        param_training["stage2"]["model_dir"],
        f"last_model_{args.case}_epoch_{param_training['stage2']['epochs']}_loss_{current_loss:.6f}.pt",
    )
    torch.save(model.state_dict(), last_model_path)
    print(f"Last model saved to: {last_model_path}")
