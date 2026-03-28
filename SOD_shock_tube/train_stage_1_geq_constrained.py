import argparse
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from architectures.model_constrained_geq import NeurDEConstrainedGeq
from SOD_solver import SODSolver
from train_stage_1 import create_basis
from utilities import *


def build_geq_base_batch(sod_solver, rho_batch, ux_batch, uy_batch, T_batch):
    geq_list = []
    for index in range(rho_batch.shape[0]):
        khi = np.zeros((sod_solver.Y, sod_solver.X), dtype=np.float32)
        zetax = np.zeros((sod_solver.Y, sod_solver.X), dtype=np.float32)
        zetay = np.zeros((sod_solver.Y, sod_solver.X), dtype=np.float32)
        geq_base, _, _, _ = sod_solver.get_Geq_Newton_solver(
            rho_batch[index],
            ux_batch[index],
            uy_batch[index],
            T_batch[index],
            khi,
            zetax,
            zetay,
        )
        geq_list.append(geq_base)
    return torch.stack(geq_list, dim=0)


if __name__ == "__main__":
    set_seed(0)

    parser = argparse.ArgumentParser(description="Train Stage 1 constrained Geq")
    parser.add_argument("--device", type=int, default=3)
    parser.add_argument("--case", type=int, choices=[1, 2], default=1)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs_override", type=int, default=20)
    parser.add_argument("--model_dir_override", type=str, required=True)
    parser.add_argument("--save_frequency", type=int, default=10)
    args = parser.parse_args()

    device = get_device(args.device)

    with open("Sod_cases_param.yml", "r") as stream:
        config = yaml.safe_load(stream)
    case_params = config[args.case]
    case_params["device"] = device

    with open("Sod_cases_param_training.yml", "r") as stream:
        training_config = yaml.safe_load(stream)
    param_training = training_config[args.case]
    param_training["stage1"]["epochs"] = int(args.epochs_override)
    param_training["stage1"]["model_dir"] = str(args.model_dir_override)
    os.makedirs(param_training["stage1"]["model_dir"], exist_ok=True)

    all_rho, all_ux, all_uy, all_T, all_Geq = load_equilibrium_state(param_training["data_dir"])
    dataset = SodDataset_stage1(
        all_rho[: args.num_samples],
        all_ux[: args.num_samples],
        all_uy[: args.num_samples],
        all_T[: args.num_samples],
        all_Geq[: args.num_samples],
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

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

    model = NeurDEConstrainedGeq(
        alpha_layer=[4] + [param_training["hidden_dim"]] * param_training["num_layers"],
        phi_layer=[2] + [param_training["hidden_dim"]] * param_training["num_layers"],
        activation="relu",
        cv=1.0 / (case_params["vuy"] - 1.0),
    ).to(device)

    optimizer = dispatch_optimizer(model=model, lr=param_training["stage1"]["lr"], optimizer_type="AdamW")
    total_steps = len(dataloader) * param_training["stage1"]["epochs"]
    scheduler_type = param_training["stage1"]["scheduler"]
    scheduler_config = param_training["stage1"].get("scheduler_config", {}).get(scheduler_type, {})
    scheduler = get_scheduler(optimizer, scheduler_type, total_steps, scheduler_config)
    basis = create_basis(case_params["Uax"], case_params["Uay"], device)
    loss_func = calculate_relative_error

    print(f"Training constrained Geq Case {args.case} on {device}. Epochs: {param_training['stage1']['epochs']}")

    best_loss = float("inf")
    best_path = None
    for epoch in tqdm(range(param_training["stage1"]["epochs"]), desc="Epochs"):
        for rho_batch, ux_batch, uy_batch, T_batch, Geq_batch in dataloader:
            rho_batch = rho_batch.to(device)
            ux_batch = ux_batch.to(device)
            uy_batch = uy_batch.to(device)
            T_batch = T_batch.to(device)
            targets = Geq_batch.permute(0, 2, 3, 1).reshape(-1, 9).to(device)
            geq_base = build_geq_base_batch(sod_solver, rho_batch, ux_batch, uy_batch, T_batch)
            input_data = torch.stack([rho_batch, ux_batch, uy_batch, T_batch], dim=1)
            optimizer.zero_grad()
            geq_pred = model(input_data, basis, geq_base=geq_base)
            loss = loss_func(geq_pred, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        current_loss = float(loss.item())
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Loss: {current_loss:.6f}")
        if current_loss < best_loss and (epoch + 1) % args.save_frequency == 0:
            if best_path and os.path.exists(best_path):
                os.remove(best_path)
            best_loss = current_loss
            best_path = os.path.join(
                param_training["stage1"]["model_dir"],
                f"best_model_{args.case}_epoch_{epoch + 1}_loss_{current_loss:.3f}.pt",
            )
            torch.save(model.state_dict(), best_path)
            print(f"Best constrained model saved to: {best_path}")

    last_model_path = os.path.join(
        param_training["stage1"]["model_dir"],
        f"last_model_{args.case}_epoch_{param_training['stage1']['epochs']}_loss_{current_loss:.3f}.pt",
    )
    torch.save(model.state_dict(), last_model_path)
    print(f"Last model saved to: {last_model_path}")
