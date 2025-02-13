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
        selected_device = 'cpu'
    return 


def plot_simulation_results(rho, ux, T, P, i, case_number):
    """Plots and saves simulation results (temperature, density, velocity, pressure)"""

    plt.figure(figsize=(15, 10))
    
    # Add a bold title for the whole figure
    plt.suptitle(f'SOD shock case {case_number} time {i}', fontweight='bold')

    plt.subplot(221)
    plt.plot(detach(rho[2, :]))
    plt.title('Density')

    plt.subplot(222)
    plt.plot(detach(T[2, :]))
    plt.title('Temperature')

    plt.subplot(223)
    plt.plot(detach(ux[2, :]))
    plt.title('Velocity in x')

    plt.subplot(224)
    plt.plot(detach(P[2, :]))
    plt.title('Pressure')

    # Adjust the layout to minimize white space
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust rect to make room for suptitle

    # Get the directory of the *main script* (not basic.py)
    main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up two levels

    image_dir = os.path.join(main_dir, f'images/ SOD_case{case_number}')  # images folder in the main directory
    os.makedirs(image_dir, exist_ok=True)
    plt.savefig(os.path.join(image_dir, f'SOD_case{case_number}_{i}.png'))
    plt.close()
