# Neural Equilibria for Long-Term Prediction of Nonlinear Conservation Laws

This repository contains code for implementing . [**neural equilibria models for long-term prediction of nonlinear conservation laws(arXiv)**](https://arxiv.org/abs/2501.06933)

```bibtex
@article{benitez2025neural,
  title={Neural equilibria for long-term prediction of nonlinear conservation laws},
  author={Benitez, J and Guo, Junyi and Hegazy, Kareem and Dokmani{\'c}, Ivan and Mahoney, Michael W and de Hoop, Maarten V},
  journal={arXiv preprint arXiv:2501.06933},
  year={2025}
}
```


## Getting Started

### Prerequisites

* Anaconda or Miniconda installed.

### Installation

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/JALB-epsilon/NeurDE.git](https://github.com/JALB-epsilon/NeurDE.git)
    cd NeurDE
    ```

2.  **Create and activate the Conda environment:**

    ```bash
    conda env create -f environment.yml
    conda activate NeurDE
    ```

    **Note:** The `environment.yml` file ensures you have all the necessary dependencies.

## Repository Structure

The repository is organized into three main directories, each focusing on a specific experimental setup:

1.  **`SOD_shock_tube`**
    * Contains the code for the Sod shock tube experiment, as presented in the paper.
2.  **`Cylinder`**
    * Implements the Cylinder case, utilizing a slow-but-accurate matrix inversion for handling obstacles and boundary conditions.
3.  **`Cylinder_faster`**
    * Offers a faster alternative to the `Cylinder` case by replacing solvers in Newton's method with sparse solvers. This approach, while significantly faster, may exhibit reduced accuracy and stability regarding rollouts.

## Running the Code (SOD Shock Tube Example)

This section demonstrates how to run the code using the Sod shock tube experiment. The process can be adapted for other directories with minor adjustments.

1.  **Generate the data:**

    ```bash
    python3 SOD_solver.py --compile --device [device_number] --case [case_number]
    ```

    * Replace `[device_number]` and `[case_number]` with the appropriate device and case numbers for the Sod shock tube experiment.

2.  **Run the training stage 1:**

    ```bash
    python3 train_stage_1.py --device [device_number] --compile --case [case_number]
    ```

    * Replace `[device_number]` and `[case_number]` with the corresponding device and case numbers.

3.  **Run the training stage 2:**

    ```bash
    python3 train_stage_2.py --compile  --device [device_number] --pre_trained_path [PATH]
    ```

    * Replace `[device_number]`, `[case_number]`, and `[PATH]` with the correct device, case, and path to the pre-trained model file.

4.  **Run the evaluation script:**

    ```bash
    python3 eval.py --compile  --device [device_number] --trained_path [PATH]
    ```

    * Replace `[device_number]`, `[case_number]`, and `[PATH]` with the appropriate device, case, and path to the trained model file.

    **Note:** When using TVD (as in case 2), the latest saved model file may yield more accurate results than the one with the lowest validation loss. We recommend experimenting with different saved files.
