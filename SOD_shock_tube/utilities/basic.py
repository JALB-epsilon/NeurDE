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
    device_map = {-1: 'cpu', 0: 'cuda:0', 1: 'cuda:1', 2: 'cuda:2', 3: 'cuda:3'}
    selected_device = device_map.get(device_index, 'cpu')
    if selected_device.startswith('cuda') and not torch.cuda.is_available():
        print(f"CUDA not available. Switching to CPU {torch.cuda.is_available()}.")
        selected_device = 'cpu'
    return selected_device

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
        all_Geq = f["Geq"][:]
        return all_rho, all_ux, all_uy, all_T, all_Geq
    
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


def TVD_norm(U_new, U_old):
    """
    Compute the TVD norm of the difference between two fields.
    """
    if U_new.shape != U_old.shape:
        raise ValueError("Input tensors U_new and U_old must have the same shape.")

    if U_new.ndim < 2:
        raise ValueError("Input tensors must have at least 2 dimensions.")

    diff_new = U_new[2, 1:] - U_new[2, :-1]
    diff_old = U_old[2, 1:] - U_old[2, :-1]

    TV_new = torch.abs(diff_new).sum()
    TV_old = torch.abs(diff_old).sum()

    TVD = F.relu(TV_new - TV_old)**2

    TVD = torch.where(TVD <= 1e-7, torch.tensor(0.0, device=TVD.device), TVD)

    return TVD

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
    def __init__(self, rho, ux, uy, T, Geq):
        self.rho = torch.tensor(rho, dtype=torch.float32)
        self.ux = torch.tensor(ux, dtype=torch.float32)
        self.uy = torch.tensor(uy, dtype=torch.float32)
        self.T = torch.tensor(T, dtype=torch.float32)
        self.Geq = torch.tensor(Geq, dtype=torch.float32)

    def __len__(self):
        return len(self.rho)

    def __getitem__(self, idx):
        return self.rho[idx], self.ux[idx], self.uy[idx], self.T[idx], self.Geq[idx]
    

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
        Fi_sequence = torch.tensor(self.all_Fi[idx:idx + self.number_of_rollout]).float()
        Gi_sequence = torch.tensor(self.all_Gi[idx:idx + self.number_of_rollout]).float()
        Feq_targets = torch.tensor(self.all_Feq[idx:idx + self.number_of_rollout]).float()
        Geq_targets = torch.tensor(self.all_Geq[idx:idx + self.number_of_rollout]).float()

        return Fi_sequence, Gi_sequence, Feq_targets, Geq_targets


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
    image_dir = os.path.join(main_dir, f'images/ SOD_case{case_number}')
    os.makedirs(image_dir, exist_ok=True)
    plt.savefig(os.path.join(image_dir, f'SOD_case{case_number}_{i}.png'))
    plt.close()
