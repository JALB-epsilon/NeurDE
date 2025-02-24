import numpy as np
import torch
import os
import matplotlib.pyplot as plt

def detach(x):
    return x.detach().cpu().numpy()


def get_device(device_index):
    device_map = {0: 'cpu', 1: 'cuda:1', 2: 'cuda:2', 3: 'cuda:3'}
    selected_device = device_map.get(device_index, 'cpu')
    if selected_device.startswith('cuda') and not torch.cuda.is_available():
        print(f"CUDA not available. Switching to CPU {torch.cuda.is_available()}.")
        selected_device = 'cpu'
    return selected_device


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