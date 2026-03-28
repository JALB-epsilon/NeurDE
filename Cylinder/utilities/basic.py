import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import h5py
from torch.utils.data import Dataset
import torch.nn.functional as F

_TORCH_DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
}

_NUMPY_DTYPE_MAP = {
    "float32": np.float32,
    "float64": np.float64,
}


def resolve_torch_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        if dtype not in _TORCH_DTYPE_MAP.values():
            raise ValueError(f"Unsupported torch dtype: {dtype}")
        return dtype
    dtype_name = str(dtype).lower()
    if dtype_name not in _TORCH_DTYPE_MAP:
        raise ValueError(f"Unsupported dtype '{dtype}'. Choose from: {sorted(_TORCH_DTYPE_MAP)}")
    return _TORCH_DTYPE_MAP[dtype_name]


def resolve_numpy_dtype(dtype):
    if isinstance(dtype, np.dtype):
        dtype_name = dtype.name
    elif isinstance(dtype, type) and issubclass(dtype, np.generic):
        dtype_name = np.dtype(dtype).name
    else:
        dtype_name = str(dtype).lower()
    if dtype_name not in _NUMPY_DTYPE_MAP:
        raise ValueError(f"Unsupported dtype '{dtype}'. Choose from: {sorted(_NUMPY_DTYPE_MAP)}")
    return _NUMPY_DTYPE_MAP[dtype_name]


def dtype_name_from_torch(dtype):
    resolved = resolve_torch_dtype(dtype)
    for name, value in _TORCH_DTYPE_MAP.items():
        if value == resolved:
            return name
    raise ValueError(f"Unsupported torch dtype: {dtype}")


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


def calculate_batch_relative_error(pred, target):
    if pred.shape != target.shape:
        raise ValueError("Prediction and target must have the same shape.")
    if pred.ndim < 2:
        return calculate_relative_error(pred, target)

    eps = 1e-7
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    numerator = torch.linalg.vector_norm(pred_flat - target_flat, dim=1)
    denominator = torch.linalg.vector_norm(target_flat, dim=1) + eps
    return (numerator / denominator).mean()


class CylinderDataset(Dataset):
    def __init__(self, rho, ux, uy, T, Geq, dtype=torch.float32):
        self.dtype = resolve_torch_dtype(dtype)
        self.rho = torch.as_tensor(rho, dtype=self.dtype)
        self.ux = torch.as_tensor(ux, dtype=self.dtype)
        self.uy = torch.as_tensor(uy, dtype=self.dtype)
        self.T = torch.as_tensor(T, dtype=self.dtype)
        self.Geq = torch.as_tensor(Geq, dtype=self.dtype)

    def __len__(self):
        return len(self.rho)

    def __getitem__(self, idx):
        return self.rho[idx], self.ux[idx], self.uy[idx], self.T[idx], self.Geq[idx]
    


class Cylinder_stage2(Dataset):
    def __init__(self, F, G, Feq, Geq, dtype=torch.float32):
        self.dtype = resolve_torch_dtype(dtype)
        self.F = torch.as_tensor(F, dtype=self.dtype)
        self.G = torch.as_tensor(G, dtype=self.dtype)
        self.Feq = torch.as_tensor(Feq, dtype=self.dtype)
        self.Geq = torch.as_tensor(Geq, dtype=self.dtype)

    def __len__(self):
        return len(self.F)

    def __getitem__(self, idx):
        return self.F[idx], self.G[idx], self.Feq[idx], self.Geq[idx]
    


class RolloutBatchDataset(Dataset):
    def __init__(self, all_Fi, all_Gi, all_Feq, all_Geq, number_of_rollout, dtype=torch.float32):
        if number_of_rollout <= 0:
            raise ValueError("number_of_rollout must be positive.")
        self.all_Fi = all_Fi
        self.all_Gi = all_Gi
        self.all_Feq = all_Feq
        self.all_Geq = all_Geq
        self.number_of_rollout = int(number_of_rollout)
        self.dtype = resolve_torch_dtype(dtype)

        available = len(all_Fi)
        if available < self.number_of_rollout:
            raise ValueError(
                f"Need at least {self.number_of_rollout} snapshots, got {available}."
            )

        # Stage-2 targets are aligned with the current state before each solver step,
        # so a rollout of length N consumes snapshots [start, start + N - 1].
        self.num_sequences = available - self.number_of_rollout + 1

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        stop = idx + self.number_of_rollout
        Fi_sequence = torch.as_tensor(self.all_Fi[idx:stop], dtype=self.dtype)
        Gi_sequence = torch.as_tensor(self.all_Gi[idx:stop], dtype=self.dtype)
        Feq_targets = torch.as_tensor(self.all_Feq[idx:stop], dtype=self.dtype)
        Geq_targets = torch.as_tensor(self.all_Geq[idx:stop], dtype=self.dtype)

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
