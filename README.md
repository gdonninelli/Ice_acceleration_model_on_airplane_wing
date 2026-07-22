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
│   ├── CMakeLists.txt
│   ├── main.cpp
│   ├── tests/
│   │   └── test_cross_validation.cpp
│   └── src/
│       ├── core/
│       │   ├── Loss.cpp / Loss.hpp
│       │   └── Tensor.cpp / Tensor.hpp
│       ├── data/
│       │   └── Dataset.cpp / Dataset.hpp
│       ├── layers/
│       │   ├── ActivationLayer.cpp / .hpp
│       │   ├── ConcatenateLayer.cpp / .hpp
│       │   ├── Conv2DLayer.cpp / .hpp
│       │   ├── DenseLayer.cpp / .hpp
│       │   ├── FlattenLayer.cpp / .hpp
│       │   ├── Layer.hpp
│       │   ├── LeakyReLULayer.cpp / .hpp
│       │   ├── ReLULayer.cpp / .hpp
│       │   ├── SigmoidLayer.cpp / .hpp
│       │   └── TanhLayer.cpp / .hpp
│       ├── model/
│       │   ├── CNNModel.cpp / CNNModel.hpp
│       │   └── ModelFactory.cpp / ModelFactory.hpp
│       ├── optimizers/
│       │   ├── AdamOptimizer.cpp / .hpp
│       │   └── Optimizer.hpp
│       ├── training/
│       │   └── Trainer.cpp / Trainer.hpp
│       └── tuning/
│           ├── CrossValidator.cpp / .hpp
│           ├── SearchSpace.hpp
│           └── TrialConfig.cpp / .hpp
├── cross_validation.md             # Tuning architecture and extension guide
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
- **LeakyReLU** activation (ReLU, Tanh and Sigmoid are also available via the shared `ActivationLayer` base class)
- **Flatten** layer
- **Concatenate** layer (merging conv features with Reynolds and AoA scalars)
- **Dense** layers (128 → 64 → 1 in the baseline)

The architecture is now constructed from typed layer recipes with automatic shape inference. `CrossValidator` can evaluate typed candidate configurations for kernels, channels, activations, dense topology, optimizer settings, and developer-provided components without adding parameter-specific methods. See [cross_validation.md](cross_validation.md).

The loss function is a physics-informed **SIMM loss** combining MSE on predictions with a physics term that enforces the relationship between predicted coefficients and the angle of attack for small angles.

![CNN Architecture](assets/cnn_architecture.png)

### Prerequisites

- C++ compiler supporting C++20
- MPI library for parallel processing
- Python 3 and NumPy (`pip install numpy`) for decoding `.npz` dataset files natively

### Data Format

Input files are `.npz` files containing SDF matrices, scalar features ordered as `[Reynolds, AoA in degrees]`, and target labels. Place dataset files (`cnn_dataset_train.npz`, `cnn_dataset_test.npz`) in a `dataset/` directory at the repository root.

Download the dataset from Kaggle:
[SDF Symmetric Airfoil High Reynolds Number](https://www.kaggle.com/datasets/giulioenzodonninelli/sdf-symmetric-airfoil-high-reynolds-number)

```bash
mkdir -p dataset
kaggle datasets download -d giulioenzodonninelli/sdf-symmetric-airfoil-high-reynolds-number -p dataset --unzip
```

### Compilation

```bash
cd CNN
mpicxx -std=c++20 -O3 -Isrc main.cpp src/core/*.cpp src/data/*.cpp src/layers/*.cpp src/model/*.cpp src/optimizers/*.cpp src/training/*.cpp src/tuning/*.cpp -o cnn_executable
```

Alternatively, from the repository root, use CMake and CTest:

```bash
cmake -S CNN -B build/CNN -DCMAKE_BUILD_TYPE=Release
cmake --build build/CNN --parallel
ctest --test-dir build/CNN --output-on-failure
```

### How to Run

From the repository root:

```bash
mpirun -n 4 ./CNN/cnn_executable
```

The activation function can be selected from the command line (default: `leakyrelu`):

```bash
mpirun -n 4 ./CNN/cnn_executable --activation tanh
mpirun -n 4 ./CNN/cnn_executable --activation leakyrelu --alpha 0.1
```

Valid activations are `leakyrelu`, `relu`, `tanh` and `sigmoid`; `--alpha` sets the negative slope and only affects `leakyrelu`.

Training parameters are available from the command line:

```bash
mpirun -n 4 ./CNN/cnn_executable --epochs 100 --batch-size 256 --learning-rate 1e-5
```

Run deterministic random K-fold evaluation for the configured model:

```bash
mpirun -n 4 ./CNN/cnn_executable --cross-validate --folds 5 --epochs 100 --batch-size 256 --seed 42
```

The global batch size is independent of MPI rank count. Fold normalization is fitted from training samples only, the test NPZ remains untouched until cross-validation is complete, and scoring uses physical-unit validation MSE. Developers can pass a typed `ParameterGrid` to `CrossValidator::tune()` when comparing configurations.

### Output

Ordinary training prints the SIMM training objective and final physical-unit test MSE. Cross-validation records each fold's training objective and validation physical MSE every 10 epochs, then reports the final fold scores, aggregate mean and deviation, and final untouched-test MSE.

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

mpicxx -cxx=g++ -std=c++20 -O3 -static-libstdc++ -Isrc main.cpp src/core/*.cpp src/data/*.cpp src/layers/*.cpp src/model/*.cpp src/optimizers/*.cpp src/training/*.cpp src/tuning/*.cpp -o cnn_mpi_executable
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
