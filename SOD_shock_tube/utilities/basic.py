import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import h5py
from torch.utils.data import Dataset
import torch.nn.functional as F

def detach(x):
    return x.detach().cpu().numpy()

def get_device(device_index):
    # Check number of available CUDA devices
    n_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_cuda == 0:
        print("CUDA not available. Switching to CPU.")
        return 'cpu'
    if device_index < 0 or device_index >= n_cuda:
        print(f"Requested CUDA device {device_index} not available. Using cuda:0 instead.")
        return 'cuda:0'
    return f'cuda:{device_index}'

def set_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

def load_equilibrium_state(file_path):
    with h5py.File(file_path, "r") as f:
        all_rho = f["rho"][:]
        all_ux = f["ux"][:]
        all_uy = f["uy"][:]
        all_T = f["T"][:]
        all_Feq = f["Feq"][:]
        all_Geq = f["Geq"][:]
        return all_rho, all_ux, all_uy, all_T, all_Feq, all_Geq
    
def load_data_stage_2(file_path):
    with h5py.File(file_path, "r") as f:
        all_F = f["Fi0"][:]
        all_G = f["Gi0"][:]
        all_Feq = f["Feq"][:]
        all_Geq = f["Geq"][:]
        return all_F, all_G, all_Feq, all_Geq
    
# loss function
def calculate_relative_error(pred, target):
    eps = 1e-7
    return torch.norm(pred - target) / (torch.norm(target)+eps)


def calculate_geq_moment_loss(pred_population, flat_macro_state, basis, cv, eps=1.0e-7):
    basis = basis.to(device=pred_population.device, dtype=pred_population.dtype)
    ex = basis[:, 0]
    ey = basis[:, 1]

    rho = flat_macro_state[:, 0]
    ux = flat_macro_state[:, 1]
    uy = flat_macro_state[:, 2]
    T = flat_macro_state[:, 3]

    E = cv * T + 0.5 * (ux * ux + uy * uy)
    H = E + T

    pred_m0 = pred_population.sum(dim=-1)
    pred_mx = pred_population @ ex
    pred_my = pred_population @ ey

    target_m0 = 2.0 * rho * E
    target_mx = 2.0 * rho * ux * H
    target_my = 2.0 * rho * uy * H

    def _relative(pred, target):
        return torch.norm(pred - target) / (torch.norm(target) + eps)

    return (_relative(pred_m0, target_m0) + _relative(pred_mx, target_mx) + _relative(pred_my, target_my)) / 3.0


def reshape_equilibrium_target(target):
    return target.movedim(-3, -1).reshape(-1, target.shape[-3])


def reshape_equilibrium_prediction(pred, target_shape):
    q_dim = target_shape[-3]
    return pred.reshape(*target_shape[:-3], target_shape[-2], target_shape[-1], q_dim).movedim(-1, -3)


def resolve_learn_feq(config, stage_name, default=True):
    stage_config = config.get(stage_name, {})
    if "learn_feq" in stage_config:
        return stage_config["learn_feq"]
    if "learn_feq" in config:
        return config["learn_feq"]
    return default


def _normalize_checkpoint_state_dict(checkpoint):
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    normalized = {}
    for key, value in state_dict.items():
        new_key = key.replace("_orig_mod.", "") if key.startswith("_orig_mod.") else key
        normalized[new_key] = value
    return normalized


def _expand_single_head_checkpoint(state_dict, target_keys=None):
    if any(key.startswith(("feq_head.", "geq_head.")) for key in state_dict):
        return state_dict

    target_keys = tuple(target_keys or ())
    has_feq_head = any(key.startswith("feq_head.") for key in target_keys)
    has_geq_head = any(key.startswith("geq_head.") for key in target_keys)
    if has_feq_head or has_geq_head:
        target_prefixes = []
        if has_feq_head:
            target_prefixes.append("feq_head.")
        if has_geq_head:
            target_prefixes.append("geq_head.")
    else:
        target_prefixes = ["feq_head.", "geq_head."]

    expanded = {}
    for key, value in state_dict.items():
        if key.startswith(("alpha.", "phi.")):
            for prefix in target_prefixes:
                expanded[f"{prefix}{key}"] = value.clone()
        else:
            expanded[key] = value
    return expanded


def load_model_checkpoint(model, checkpoint_path, map_location=None):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = _expand_single_head_checkpoint(
        _normalize_checkpoint_state_dict(checkpoint),
        target_keys=model.state_dict().keys(),
    )
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    return missing_keys, unexpected_keys


def TVD_norm(U_new, U_old):
    """
    Compute the TVD norm of the difference between two fields.
    """
    if U_new.shape != U_old.shape:
        raise ValueError("Input tensors U_new and U_old must have the same shape.")

    if U_new.ndim < 2:
        raise ValueError("Input tensors must have at least 2 dimensions.")

    diff_new = U_new[..., 2, 1:] - U_new[..., 2, :-1]
    diff_old = U_old[..., 2, 1:] - U_old[..., 2, :-1]

    TV_new = torch.abs(diff_new).sum(dim=-1)
    TV_old = torch.abs(diff_old).sum(dim=-1)

    TVD = F.relu(TV_new - TV_old) ** 2
    TVD = torch.where(TVD <= 1e-7, torch.zeros_like(TVD), TVD)
    return TVD.mean()


def local_variation_increase_penalty(U_new, U_old):
    if U_new.shape != U_old.shape:
        raise ValueError("Input tensors U_new and U_old must have the same shape.")

    diff_new = U_new[..., 1:] - U_new[..., :-1]
    diff_old = U_old[..., 1:] - U_old[..., :-1]
    variation_growth = F.relu(torch.abs(diff_new) - torch.abs(diff_old))
    return variation_growth.square().mean()


def local_curvature_increase_penalty(U_new, U_old):
    if U_new.shape != U_old.shape:
        raise ValueError("Input tensors U_new and U_old must have the same shape.")
    if U_new.shape[-1] < 3:
        return torch.zeros((), device=U_new.device, dtype=U_new.dtype)

    diff_new = U_new[..., 1:] - U_new[..., :-1]
    diff_old = U_old[..., 1:] - U_old[..., :-1]
    curv_new = diff_new[..., 1:] - diff_new[..., :-1]
    curv_old = diff_old[..., 1:] - diff_old[..., :-1]
    curvature_growth = F.relu(torch.abs(curv_new) - torch.abs(curv_old))
    return curvature_growth.square().mean()

def tvd_weight_scheduler(epoch, milestones, weights):
    """
    A scheduler to change the TVD weight at specific milestone epochs.
    """
    if not milestones:
        return weights[0] if weights else 1.0  

    if epoch < milestones[0]:
        return weights[0]

    for i in range(len(milestones) - 1):
        if milestones[i] <= epoch < milestones[i + 1]:
            return weights[i + 1]

    return weights[-1]

class SodDataset_stage1(Dataset):
    def __init__(self, rho, ux, uy, T, Feq, Geq):
        self.rho = torch.tensor(rho, dtype=torch.float32)
        self.ux = torch.tensor(ux, dtype=torch.float32)
        self.uy = torch.tensor(uy, dtype=torch.float32)
        self.T = torch.tensor(T, dtype=torch.float32)
        self.Feq = torch.tensor(Feq, dtype=torch.float32)
        self.Geq = torch.tensor(Geq, dtype=torch.float32)

    def __len__(self):
        return len(self.rho)

    def __getitem__(self, idx):
        return self.rho[idx], self.ux[idx], self.uy[idx], self.T[idx], self.Feq[idx], self.Geq[idx]
    

class SodDataset_stage2(Dataset):
    def __init__(self, F, G, Feq, Geq):
        self.F = torch.tensor(F, dtype=torch.float32)
        self.G = torch.tensor(G, dtype=torch.float32)
        self.Feq = torch.tensor(Feq, dtype=torch.float32)
        self.Geq = torch.tensor(Geq, dtype=torch.float32)

    def __len__(self):
        return len(self.F)

    def __getitem__(self, idx):
        return self.F[idx], self.G[idx], self.Feq[idx], self.Geq[idx]
    


class RolloutBatchDataset(Dataset):
    def __init__(self, all_Fi, all_Gi, all_Feq, all_Geq, number_of_rollout):
        self.all_Fi = all_Fi
        self.all_Gi = all_Gi
        self.all_Feq = all_Feq
        self.all_Geq = all_Geq
        self.number_of_rollout = number_of_rollout
        self.num_sequences = len(all_Fi)-number_of_rollout+1  # Use the total length

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        Fi_sequence = torch.tensor(self.all_Fi[idx:idx + self.number_of_rollout]).float() # should be +1?
        Gi_sequence = torch.tensor(self.all_Gi[idx:idx + self.number_of_rollout]).float()
        Feq_targets = torch.tensor(self.all_Feq[idx:idx + self.number_of_rollout]).float()
        Geq_targets = torch.tensor(self.all_Geq[idx:idx + self.number_of_rollout]).float()

        return Fi_sequence, Gi_sequence, Feq_targets, Geq_targets


class RolloutMacroBatchDataset(Dataset):
    def __init__(
        self,
        all_Fi,
        all_Gi,
        all_Feq,
        all_Geq,
        input_rho,
        input_ux,
        input_uy,
        input_T,
        number_of_rollout,
        target_rho=None,
        target_ux=None,
        target_uy=None,
        target_T=None,
    ):
        self.all_Fi = all_Fi
        self.all_Gi = all_Gi
        self.all_Feq = all_Feq
        self.all_Geq = all_Geq
        self.input_rho = input_rho
        self.input_ux = input_ux
        self.input_uy = input_uy
        self.input_T = input_T
        self.target_rho = input_rho if target_rho is None else target_rho
        self.target_ux = input_ux if target_ux is None else target_ux
        self.target_uy = input_uy if target_uy is None else target_uy
        self.target_T = input_T if target_T is None else target_T
        self.number_of_rollout = number_of_rollout
        self.num_sequences = len(all_Fi) - number_of_rollout

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        end = idx + self.number_of_rollout + 1
        Fi_sequence = torch.tensor(self.all_Fi[idx:end]).float()
        Gi_sequence = torch.tensor(self.all_Gi[idx:end]).float()
        Feq_targets = torch.tensor(self.all_Feq[idx:end]).float()
        Geq_targets = torch.tensor(self.all_Geq[idx:end]).float()
        input_rho_sequence = torch.tensor(self.input_rho[idx:end]).float()
        input_ux_sequence = torch.tensor(self.input_ux[idx:end]).float()
        input_uy_sequence = torch.tensor(self.input_uy[idx:end]).float()
        input_T_sequence = torch.tensor(self.input_T[idx:end]).float()
        target_rho_sequence = torch.tensor(self.target_rho[idx:end]).float()
        target_ux_sequence = torch.tensor(self.target_ux[idx:end]).float()
        target_uy_sequence = torch.tensor(self.target_uy[idx:end]).float()
        target_T_sequence = torch.tensor(self.target_T[idx:end]).float()
        return (
            Fi_sequence,
            Gi_sequence,
            Feq_targets,
            Geq_targets,
            input_rho_sequence,
            input_ux_sequence,
            input_uy_sequence,
            input_T_sequence,
            target_rho_sequence,
            target_ux_sequence,
            target_uy_sequence,
            target_T_sequence,
        )


def plot_simulation_results(rho, ux, T, P, i, case_number):
    """Plots and saves simulation results with larger title and reduced whitespace."""

    plt.figure(figsize=(16, 6))

    # Larger title and reduced whitespace
    plt.suptitle(f'SOD shock case {case_number} time {i}', fontweight='bold', fontsize=25, y=0.95) 

    linewidth = 5

    plt.subplot(221)
    plt.plot(detach(rho[2, :]), linewidth=linewidth)
    plt.title('Density', fontsize=18)  # Slightly increased fontsize

    plt.subplot(222)
    plt.plot(detach(T[2, :]), linewidth=linewidth)
    plt.title('Temperature', fontsize=18)

    plt.subplot(223)
    plt.plot(detach(ux[2, :]), linewidth=linewidth)
    plt.title('Velocity in x', fontsize=18)

    plt.subplot(224)
    plt.plot(detach(P[2, :]), linewidth=linewidth)
    plt.title('Pressure', fontsize=18)

    # Reduced whitespace - Key changes here:
    plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=0.35, w_pad=0.35)  

    main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_dir = os.path.join(main_dir, 'images', f'SOD_case{case_number}')
    os.makedirs(image_dir, exist_ok=True)
    plt.savefig(os.path.join(image_dir, f'SOD_case{case_number}_{i}.png'))
    plt.close()
