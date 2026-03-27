import torch
from architectures import NeurDE
from utilities import *
import argparse
import yaml
import os
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **kwargs):
        del kwargs
        return iterable

def create_basis(Uax, Uay, device):
    # Keep the basis identical to the solver's shifted lattice velocities.
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
    parser.add_argument("--save_frequency", type=int, default=25, help='Save model')
    parser.add_argument("--epochs_override", type=int, default=None, help='Optional override for stage-1 epochs')
    parser.add_argument("--training_config", type=str, default="Sod_cases_param_training.yml", help='Training YAML path')
    parser.set_defaults(save_model=True)
    args = parser.parse_args()

    device = get_device(args.device)

    with open("Sod_cases_param.yml", 'r') as stream:
        config = yaml.safe_load(stream)
    case_params = config[args.case]
    case_params['device'] = device

    with open(args.training_config, 'r') as stream:
        training_config = yaml.safe_load(stream)
    param_training = training_config[args.case]
    learn_feq = resolve_learn_feq(param_training, "stage1", default=False)
    logit_clip = param_training.get("logit_clip", 15.0)
    project_feq_moments = param_training.get("project_feq_moments", False)
    enforce_sod_symmetry = param_training.get("enforce_sod_symmetry", False)
    feq_nullspace_gamma = param_training["stage1"].get("feq_nullspace_gamma", param_training.get("feq_nullspace_gamma", 1.0))
    feq_nullspace_gamma_mode = param_training["stage1"].get("feq_nullspace_gamma_mode", param_training.get("feq_nullspace_gamma_mode", "fixed"))
    feq_nullspace_gamma_min = param_training["stage1"].get("feq_nullspace_gamma_min", param_training.get("feq_nullspace_gamma_min", 0.6))
    feq_nullspace_gamma_max = param_training["stage1"].get("feq_nullspace_gamma_max", param_training.get("feq_nullspace_gamma_max", 1.0))
    geq_mode = param_training["stage1"].get("geq_mode", param_training.get("geq_mode", "learned"))
    geq_moment_weight = param_training["stage1"].get("geq_moment_weight", param_training.get("geq_moment_weight", 0.0))
    geq_nullspace_gamma = param_training["stage1"].get("geq_nullspace_gamma", param_training.get("geq_nullspace_gamma", 1.0))
    geq_nullspace_gamma_mode = param_training["stage1"].get("geq_nullspace_gamma_mode", param_training.get("geq_nullspace_gamma_mode", "fixed"))
    geq_nullspace_gamma_min = param_training["stage1"].get("geq_nullspace_gamma_min", param_training.get("geq_nullspace_gamma_min", 0.6))
    geq_nullspace_gamma_max = param_training["stage1"].get("geq_nullspace_gamma_max", param_training.get("geq_nullspace_gamma_max", 1.0))

    if args.save_model:
        os.makedirs(param_training["stage1"]["model_dir"], exist_ok=True)

    all_rho, all_ux, all_uy, all_T, all_Feq, all_Geq = load_equilibrium_state(param_training["data_dir"])

    dataset = SodDataset_stage1(
        all_rho[:args.num_samples],
        all_ux[:args.num_samples],
        all_uy[:args.num_samples],
        all_T[:args.num_samples],
        all_Feq[:args.num_samples],
        all_Geq[:args.num_samples],
    )
    use_workers = 4 if str(device).startswith("cuda") else 0
    use_pin_memory = str(device).startswith("cuda")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=use_workers, pin_memory=use_pin_memory)

    cv = 1.0 / (case_params["vuy"] - 1)
    model = NeurDE(
        alpha_layer=[4] + [param_training["hidden_dim"]] * param_training["num_layers"],
        phi_layer=[2] + [param_training["hidden_dim"]] * param_training["num_layers"],
        activation='relu',
        learn_feq=learn_feq,
        learn_geq=True,
        logit_clip=logit_clip,
        project_feq_moments=project_feq_moments,
        enforce_sod_symmetry=enforce_sod_symmetry,
        feq_nullspace_gamma=feq_nullspace_gamma,
        feq_nullspace_gamma_mode=feq_nullspace_gamma_mode,
        feq_nullspace_gamma_min=feq_nullspace_gamma_min,
        feq_nullspace_gamma_max=feq_nullspace_gamma_max,
        geq_mode=geq_mode,
        geq_cv=cv,
        geq_nullspace_gamma=geq_nullspace_gamma,
        geq_nullspace_gamma_mode=geq_nullspace_gamma_mode,
        geq_nullspace_gamma_min=geq_nullspace_gamma_min,
        geq_nullspace_gamma_max=geq_nullspace_gamma_max,
    ).to(device)


    if args.compile:
        model = torch.compile(model)
        print("Model compiled.")

    optimizer = dispatch_optimizer(model=model,
                                    lr=param_training["stage1"]["lr"],
                                    optimizer_type="AdamW")

    epochs = args.epochs_override or param_training["stage1"]["epochs"]
    total_steps = len(dataloader) * epochs
    scheduler_type = param_training["stage1"]["scheduler"]
    scheduler_config = param_training["stage1"].get("scheduler_config", {}).get(scheduler_type,{}) 
    scheduler = get_scheduler(optimizer, scheduler_type, total_steps, scheduler_config)

    Uax, Uay = case_params["Uax"], case_params["Uay"]
    basis = create_basis(Uax, Uay, device)

    loss_func = calculate_relative_error

    feq_path = "learned" if learn_feq else "analytic"
    print(
        f"Training Case {args.case} on {device}. Epochs: {epochs}, Samples: {args.num_samples}, "
        f"feq_path={feq_path}, geq_mode={geq_mode}, symmetry={enforce_sod_symmetry}, geq_moment_weight={geq_moment_weight}, "
        f"feq_gamma_mode={feq_nullspace_gamma_mode}, feq_gamma={feq_nullspace_gamma}, "
        f"feq_gamma_bounds=({feq_nullspace_gamma_min}, {feq_nullspace_gamma_max}), "
        f"geq_gamma_mode={geq_nullspace_gamma_mode}, geq_gamma={geq_nullspace_gamma}, "
        f"geq_gamma_bounds=({geq_nullspace_gamma_min}, {geq_nullspace_gamma_max})"
    )
  


    best_losses = [float('inf')] * 3
    best_models = [None] * 3
    best_model_paths = [None] * 3

    save_frequency = args.save_frequency
    epochs_since_last_save = [0] * 3 
    last_epoch_loss = 0.0 
    for epoch in tqdm(range(epochs), desc="Epochs"):
        loss_epoch = 0
        for rho_batch, ux_batch, uy_batch, T_batch, Feq_batch, Geq_batch in dataloader:
            input_data = torch.stack([rho_batch, ux_batch, uy_batch, T_batch], dim=1).to(device)
            flat_macro_state = input_data.movedim(1, -1).reshape(-1, input_data.shape[1])
            geq_targets = Geq_batch.permute(0, 2, 3, 1).reshape(-1, 9).to(device)
            optimizer.zero_grad()
            if learn_feq:
                feq_targets = Feq_batch.permute(0, 2, 3, 1).reshape(-1, 9).to(device)
                Feq_pred, Geq_pred = model(input_data, basis)
                loss_feq = loss_func(Feq_pred, feq_targets)
                loss_geq = loss_func(Geq_pred, geq_targets)
                loss = 0.5 * (loss_feq + loss_geq)
            else:
                Geq_pred = model(input_data, basis)
                loss = loss_func(Geq_pred, geq_targets)
            if geq_moment_weight > 0.0:
                geq_moment_loss = calculate_geq_moment_loss(Geq_pred, flat_macro_state, basis, cv=cv)
                loss = loss + geq_moment_weight * geq_moment_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_epoch += loss.item()
            scheduler.step()

        current_loss = loss_epoch / max(len(dataloader), 1)
        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, Loss: {current_loss:.6f}")

        if current_loss < max(best_losses):
            max_index = best_losses.index(max(best_losses))
            best_losses[max_index] = current_loss
            best_models[max_index] = model.state_dict()

            if args.save_model and epochs_since_last_save[max_index] >= save_frequency:
                if best_model_paths[max_index] and os.path.exists(best_model_paths[max_index]):
                    os.remove(best_model_paths[max_index])
                save_path = os.path.join(param_training["stage1"]["model_dir"], f"best_model_{args.case}_epoch_{epoch+1}_top_{max_index+1}_loss_{current_loss:.3f}.pt")
                torch.save(best_models[max_index], save_path)
                print(f"Top {max_index+1} model saved to: {save_path}")
                best_model_paths[max_index] = save_path
                epochs_since_last_save[max_index] = 0 #reset the counter
            else:
                epochs_since_last_save[max_index] +=1

        else:
            for i in range(3):
                epochs_since_last_save[i] +=1

    # Save the last model with its loss
    if args.save_model:
        last_epoch_loss= current_loss
        last_model_path = os.path.join(param_training["stage1"]["model_dir"], f"last_model_{args.case}_epoch_{epochs}_loss_{last_epoch_loss:.3f}.pt")
        torch.save(model.state_dict(), last_model_path)
        print(f"Last model saved to: {last_model_path}")

    if not args.save_model:
        print("Model saving disabled.")

    print("Training complete.")
