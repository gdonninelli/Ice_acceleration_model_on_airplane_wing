# Ice Acceleration Model on Airplane Wing

This project reproduces the CNN-based approach from the AMSC Project paper for predicting aerodynamic coefficients on airfoil profiles. The core idea is to replace traditional CFD simulations with a data-driven surrogate model: given the airfoil geometry (encoded as a Signed Distance Function) and flight conditions (Reynolds number and Angle of Attack), the model predicts the resulting aerodynamic forces.

Unlike a standard purely data-driven neural network, this implementation adopts a **physics-informed cost function** (SIMM loss). The loss combines a standard MSE term on the predicted coefficients with a physics-based regularization term that enforces the known linear relationship between the lift coefficient and the angle of attack in the small-angle regime. This physical prior acts as a soft constraint, guiding the model toward physically consistent predictions and improving generalization.

The pipeline is composed of two main modules:
- **SDF Generator** — converts raw airfoil boundary coordinates into a grid-based Signed Distance Function representation.
- **CNN Model** — a custom C++/MPI CNN framework built from scratch, performing manual forward/backward passes with distributed gradient averaging via MPI.

![SDF to Coefficient Pipeline](assets/SDF.png)

---

## Repository Structure

```text
.
├── README.md
├── assets/
│   ├── SDF.png
│   ├── cnn_architecture.png
│   └── training trend.png
├── SDF/                          # SDF Generator subproject
│   ├── main.cpp
│   ├── SDFGenerator.cpp
│   ├── SDFGenerator.hpp
│   ├── visualization.ipynb
│   └── data/
├── CNN/                          # CNN Model subproject
│   ├── main.cpp
│   └── src/
│       ├── core/
│       │   ├── Loss.cpp / Loss.hpp
│       │   └── Tensor.cpp / Tensor.hpp
│       ├── data/
│       │   └── NPZDataLoader.cpp / NPZDataLoader.hpp
│       ├── layers/
│       │   ├── ConcatenateLayer.cpp / .hpp
│       │   ├── Conv2DLayer.cpp / .hpp
│       │   ├── DenseLayer.cpp / .hpp
│       │   ├── FlattenLayer.cpp / .hpp
│       │   ├── Layer.hpp
│       │   └── LeakyReLULayer.cpp / .hpp
│       ├── model/
│       │   └── CNNModel.cpp / CNNModel.hpp
│       └── optimizers/
│           ├── AdamOptimizer.cpp / .hpp
│           └── Optimizer.hpp
└── dataset/                      # Dataset directory (created at setup)
```

---

## SDF Generator

The SDF Generator reads airfoil boundary coordinates from `.dat` files and computes a Signed Distance Function on a fixed-size grid. The output is saved as a text matrix.

### Prerequisites

- C++ compiler supporting C++17
- MPI library for parallel processing

### Data Format

Input files are `.dat` files containing whitespace-separated `x` and `z` coordinates per line.

### Compilation

```bash
cd SDF
mpic++ main.cpp SDFGenerator.cpp -o sdfgen -std=c++17
```

### How to Run

1. Create a `data` directory in the same location as the compiled `sdfgen` executable.
2. Place your airfoil `.dat` files (e.g., `n0012.dat`) inside the `data` directory.
3. Execute with `mpirun`:

```bash
mpirun -np 4 ./sdfgen
```

### Output

For each input `.dat` file (e.g., `airfoil.dat`), a corresponding SDF matrix file is generated as `airfoil_matrix.txt`. This file contains SDF grid values in space-separated matrix format: positive values are outside the airfoil, negative values are inside.

---

## CNN Model

The CNN model is a custom C++ Convolutional Neural Network framework built from scratch. It performs manual forward/backward passes, distributed gradient averaging using MPI, and trains a regression model to predict aerodynamic coefficients from SDF matrices and scalar features (Reynolds number, Angle of Attack).

The architecture consists of:

- **2D Convolution** (1→8 channels, kernel 5x5, stride 5)
- **LeakyReLU** activation
- **Flatten** layer
- **Concatenate** layer (merging conv features with AoA and Re scalars)
- **Dense** layers (256 → 128 → 64 → 1)

The loss function is a physics-informed **SIMM loss** combining MSE on predictions with a physics term that enforces the relationship between predicted coefficients and the angle of attack for small angles.

![CNN Architecture](assets/cnn_architecture.png)

### Prerequisites

- C++ compiler supporting C++20
- MPI library for parallel processing
- Python 3 and NumPy (`pip install numpy`) for decoding `.npz` dataset files natively

### Data Format

Input files are `.npz` files containing SDF matrices, scalar features (Re, AoA), and target labels. Place dataset files (`cnn_dataset_train.npz`, `cnn_dataset_test.npz`) in a `dataset/` directory at the repository root.

Download the dataset from Kaggle:
[SDF Symmetric Airfoil High Reynolds Number](https://www.kaggle.com/datasets/giulioenzodonninelli/sdf-symmetric-airfoil-high-reynolds-number)

```bash
mkdir -p dataset
kaggle datasets download -d giulioenzodonninelli/sdf-symmetric-airfoil-high-reynolds-number -p dataset --unzip
```

### Compilation

```bash
cd CNN
mpicxx -std=c++20 -O3 -Isrc main.cpp src/core/*.cpp src/data/*.cpp src/layers/*.cpp src/model/*.cpp src/optimizers/*.cpp -o cnn_executable
```

### How to Run

From the repository root:

```bash
mpirun -n 4 ./CNN/cnn_executable
```

Training parameters (epochs, batch size, learning rate) can be configured in `CNN/main.cpp`.

### Output

The program prints epoch progression with training and validation loss values to stdout, including the SIMM physics-informed loss components.

![Training and Validation Loss](assets/training%20trend.png)

---

## CINECA Leonardo Cluster Deployment

This section describes how to download, compile, and run the CNN model on the CINECA Leonardo cluster.

### 1. Downloading & Syncing

**Option A: Clone directly on the cluster (recommended)**

```bash
cd ~
git clone https://github.com/AleVerri-03/Ice_acceleration_model_on_airplane_wing.git
cd Ice_acceleration_model_on_airplane_wing
```

**Option B: Sync local changes via rsync (from local terminal)**

```bash
rsync -avz ./CNN <username>@login.leonardo.cineca.it:~/
```

### 2. Dataset Setup

Symlink the dataset directory on Leonardo to avoid duplicating large files:

```bash
cd /leonardo/home/userexternal/<username>/Ice_acceleration_model_on_airplane_wing/CNN
ln -s ../dataset ./dataset
```

### 3. Environment Setup & Compilation

Load the required modules and compile on the **login node**:

```bash
module purge
module load python/3.11.7
module load gcc/11.3.0
module load intel-oneapi-compilers
module load intel-oneapi-mpi

mpicxx -cxx=g++ -std=c++20 -O3 -static-libstdc++ -Isrc main.cpp src/core/*.cpp src/data/*.cpp src/layers/*.cpp src/model/*.cpp src/optimizers/*.cpp -o cnn_mpi_executable
```

### 4. Submitting Slurm Jobs

Create a job script `run_cnn.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=cnn_training
#SBATCH --account=<project_account>
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --time=01:00:00
#SBATCH --output=cnn_%j.out
#SBATCH --error=cnn_%j.err

cd /leonardo/home/userexternal/<username>/Ice_acceleration_model_on_airplane_wing/CNN

module purge
module load python/3.11.7
module load gcc/11.3.0
module load intel-oneapi-compilers
module load intel-oneapi-mpi

srun ./cnn_mpi_executable
```

Submit and monitor:

```bash
sbatch run_cnn.sh        # Submit
squeue -u <username>     # Check status
tail -f cnn_*.out        # Live output
cat cnn_*.err            # Check errors
scancel <JOB_ID>         # Cancel job
```

### 5. Budget & Storage

- **Check budget:** `saldo -u <username> -b --dcgp`
- **Cost formula:** `Nodes × Cores × Hours` (e.g., 1 node × 64 cores × 1h = 64 core-hours)
- **Storage:** `$HOME` has strict quotas; move large files to `$SCRATCH` if needed.
