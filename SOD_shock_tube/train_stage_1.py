import torch
from architectures import NeurDE
from utilities import *
import argparse
import yaml
from tqdm import tqdm
import os
from torch.utils.data import DataLoader

def create_basis(Uax, Uay, device, dtype=torch.float32):
    dtype = resolve_torch_dtype(dtype)
    ex_values = [1, 0, -1, 0, 1, -1, -1, 1, 0]
    ey_values = [0, 1, 0, -1, 1, 1, -1, -1, 0]
    ex = torch.tensor(ex_values, dtype=dtype) + Uax
    ey = torch.tensor(ey_values, dtype=dtype) + Uay
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
    parser.add_argument("--save_frequency", default=25, help='Save model')
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.set_defaults(save_model=True)
    args = parser.parse_args()

    device = get_device(args.device)
    dtype = resolve_torch_dtype(args.dtype)

    with open("Sod_cases_param.yml", 'r') as stream:
        config = yaml.safe_load(stream)
    case_params = config[args.case]
    case_params['device'] = device

    with open("Sod_cases_param_training.yml", 'r') as stream:
        training_config = yaml.safe_load(stream)
    param_training = training_config[args.case]
    model_config = get_model_config(param_training)
    supervision_mode = str(param_training["stage1"].get("supervision", "geq")).lower()
    if supervision_mode not in {"feq", "geq"}:
        raise ValueError(f"SOD stage1 only supports feq or geq supervision, got: {supervision_mode}")

    os.makedirs(param_training["stage1"]["model_dir"], exist_ok=True) 

    all_rho, all_ux, all_uy, all_T, all_Feq, all_Geq = load_equilibrium_state(param_training["data_dir"])

    dataset = SodDataset_stage1(
        all_rho[:args.num_samples],
        all_ux[:args.num_samples],
        all_uy[:args.num_samples],
        all_T[:args.num_samples],
        all_Feq[:args.num_samples],
        all_Geq[:args.num_samples],
        dtype=dtype,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = NeurDE(
        alpha_layer=[4] + [param_training["hidden_dim"]] * param_training["num_layers"],
        phi_layer=[2] + [param_training["hidden_dim"]] * param_training["num_layers"],
        activation='relu',
        learn_feq=supervision_mode == "feq",
        learn_geq=supervision_mode == "geq",
        feq_mode=model_config["feq_mode"],
        geq_mode=model_config["geq_mode"],
        cv=1.0 / (case_params["vuy"] - 1.0),
        logit_clip=model_config["logit_clip"],
        feq_base_measure=model_config["feq_base_measure"],
        geq_base_measure=model_config["geq_base_measure"],
        newton_iters=model_config["newton_iters"],
        newton_tolerance=model_config["newton_tolerance"],
    ).to(device=device, dtype=dtype)


    if args.compile:
        model = torch.compile(model)
        print("Model compiled.")

    optimizer = dispatch_optimizer(model=model,
                                    lr=param_training["stage1"]["lr"],
                                    optimizer_type="AdamW")

    total_steps = len(dataloader) * param_training["stage1"]["epochs"]
    scheduler_type = param_training["stage1"]["scheduler"]
    scheduler_config = param_training["stage1"].get("scheduler_config", {}).get(scheduler_type,{}) 
    scheduler = get_scheduler(optimizer, scheduler_type, total_steps, scheduler_config)

    Uax, Uay = case_params["Uax"], case_params["Uay"]
    basis = create_basis(Uax, Uay, device, dtype=dtype)

    epochs = param_training["stage1"]["epochs"]
    loss_func = calculate_relative_error

    print(
        f"Training Case {args.case} on {device}. Epochs: {epochs}, Samples: {args.num_samples}, "
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
        for rho_batch, ux_batch, uy_batch, T_batch, Feq_batch, Geq_batch in dataloader:
            input_data = torch.stack([rho_batch, ux_batch, uy_batch, T_batch], dim=1).to(device)
            target_batch = Feq_batch if supervision_mode == "feq" else Geq_batch
            targets = target_batch.permute(0, 2, 3, 1).reshape(-1, 9).to(device)
            optimizer.zero_grad()
            equilibrium_pred = model(input_data, basis)
            loss = loss_func(equilibrium_pred, targets)
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
