import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import h5py
from torch.utils.data import Dataset

def detach(x):
    return x.detach().cpu().numpy()

def get_device(device_index):
    device_map = {0: 'cpu', 1: 'cuda:1', 2: 'cuda:2', 3: 'cuda:3'}
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
    

# loss function
def calculate_relative_error(pred, target):
    return torch.norm(pred - target) / torch.norm(target)


class SodDataset(Dataset):
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
