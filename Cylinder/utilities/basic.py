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


class CylinderDataset(Dataset):
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
    


class Cylinder_stage2(Dataset):
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


def plot_simulation_results(Field_GT, time_value):
    """Plots and saves Ground Truth simulation results."""

    fig, (ax, cax) = plt.subplots(1, 2, figsize=(8, 4.5), gridspec_kw={"width_ratios": [1, 0.05]})  # Adjusted figsize and gridspec

    # Plot settings
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('auto')

    # Plot Ground Truth
    im = ax.imshow(Field_GT, cmap='jet')
    ax.set_title(r'Ref: local Mach number ($\mathrm{Ma}$)', fontsize=16, fontweight='bold')

    # Colorbar
    norm = plt.Normalize(vmin=np.min(Field_GT), vmax=np.max(Field_GT))
    sm = plt.cm.ScalarMappable(cmap='jet', norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cax, orientation='vertical')

    fig.suptitle(f"Time: {time_value} (Supersonic flow around a circular cylinder)", fontsize=16, fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap with colorbar

    # Save to Images directory
    main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_dir = os.path.join(main_dir, 'images', 'Cylinder')
    os.makedirs(image_dir, exist_ok=True)
    plt.savefig(os.path.join(image_dir, f'Cylinder_{time_value}.png'), bbox_inches='tight')

    plt.close(fig)