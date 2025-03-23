# Neural Equilibria for Long-Term Prediction of Nonlinear Conservation Laws  
**Efficient Implementation and Experimental Code Repository**

This repository contains the code for implementing [**neural equilibria models for long-term prediction of nonlinear conservation laws (arXiv)**](https://arxiv.org/abs/2501.06933).  
These models are designed to predict complex nonlinear dynamics governed by conservation laws, ensuring long-term stability and accuracy.

---

## 📚 Citation  
If you use this repository, please cite the paper using the following BibTeX:

```bibtex
@article{benitez2025neural,
  title={Neural equilibria for long-term prediction of nonlinear conservation laws},
  author={Benitez, J and Guo, Junyi and Hegazy, Kareem and Dokmani{\'c}, Ivan and Mahoney, Michael W and de Hoop, Maarten V},
  journal={arXiv preprint arXiv:2501.06933},
  year={2025}
}
```

## 🚀 Getting Started

### ✅ Prerequisites
Ensure you have the following prerequisites installed:

*Anaconda or Miniconda (recommended for dependency management)

*Required libraries (provided in environment.yml)

## 📦 Installation

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

3. **Verify environment installation:**

    ```bash
    conda info --envs
    ```

## 📁 Repository Structure



The repository is organized into three main directories, each focusing on a specific experimental setup:

| Directory           | Description                                                                                                                                                                                                                                                            |
|--------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `SOD_shock_tube`    | Contains the code for the Sod shock tube experiment, as presented in the paper.                                                                                                                                                                                        |
| `Cylinder`          | Implements the Cylinder case, utilizing a slow-but-accurate matrix inversion for handling obstacles and boundary conditions. In this implementation, boundary conditions are enforced by detaching the gradient during training.                                    |
| `Cylinder_faster` | Offers a faster alternative to the `Cylinder` case by replacing solvers in Newton's method with sparse solvers. This approach, while significantly faster, may exhibit reduced accuracy and stability regarding rollouts. In this implementation, boundary conditions are enforced by detaching the gradient during training. | 


### **Directory structure**
```
NeurDE/
├── SOD_shock_tube/
│   ├── SOD_solver.py
│   ├── train_stage_1.py
│   ├── train_stage_2.py
│   ├── eval.py
│   └── ... (other files)
├── Cylinder/
│   ├── ... (cylinder-related files)
├── Cylinder_faster/
│   ├── ... (faster cylinder-related files)
├── environment.yml
└── README.md
```

## ▶️ Running the Code

The following commands demonstrate how to run the code for the Sod shock tube experiment. You can modify these commands for other experiments by changing the directory and case parameters.

### ⚡️ Run SOD Shock Tube Experiment

1.  **Generate the data:**

    ```bash
    python3 SOD_solver.py --compile --device [device_number] --case [case_number]
    ```

    * `[device_number]`: Specifies the GPU device number to use. You can find available device numbers using `nvidia-smi`. If you want to use the CPU, you can specify `-1`.
    * `[case_number]`: Selects the Sod shock tube case. For example, `1` might represent a standard Sod shock tube setup, and `2` might represent a case with a TVD scheme.
    * `--compile`: Compiles the JIT functions for faster execution.

2.  **Run the training stage 1:**

    ```bash
    python3 train_stage_1.py --device [device_number] --compile --case [case_number]
    ```

    * `[device_number]`: Specifies the GPU device number.
    * `[case_number]`: Specifies the Sod shock tube case number.
    * `--compile`: Compiles the JIT functions for faster execution.

3.  **Run the training stage 2:**

    ```bash
    python3 train_stage_2.py --compile  --device [device_number] --pre_trained_path [PATH]
    ```

    * `[device_number]`: Specifies the GPU device number.
    * `[PATH]`: Specifies the full path to the pre-trained model file (e.g., `saved_models/stage_1_model.pth`).
    * `--compile`: Compiles the JIT functions for faster execution.

4.  **Run the evaluation script:**

    ```bash
    python3 eval.py --compile  --device [device_number] --trained_path [PATH]
    ```

    * `[device_number]`: Specifies the GPU device number.
    * `[PATH]`: Specifies the full path to the trained model file (e.g., `saved_models/stage_2_model.pth`).
    * `--compile`: Compiles the JIT functions for faster execution.

    **Note:** When using TVD (as in case 2), the latest saved model file may yield more accurate results than the one with the lowest validation loss. We recommend experimenting with different saved files located at the specified `[PATH]`.

## 📊 **Important Notes:** 
* In the paper, we use `torch.roll` for streaming. In this code, we removed the use of this function by defining the indices of the streaming directly. otherwise we have to make a for loop in the number of discrete velocities.
* We also use a sparse solver for the matrix inversion required in Newton's method for all the cases: for the cylinder, the BC and Obstacle uses a numpy inversion while the faster cylinder use a sparse solver for all the cases in the cylinder test. 

## 📞 Contact and Support
* The original code used for the paper can be provided upon request. Please open a GitHub issue or contact [antonio.lara@rice.edu] to request the original code---it is slower.*


## 🤝 Contributing
Contributions are welcome! 
