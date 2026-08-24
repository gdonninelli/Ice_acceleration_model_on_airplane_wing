# Optimizer Comparison Experiment

Cross-validation experiment comparing the performance of different first-order and adaptive optimization algorithms implemented for the CNN surrogate model, built on the typed `CrossValidator` and `ParameterGrid` infrastructure documented in [`docs/cross_validation.md`](../../../docs/cross_validation.md).

## Scientific Question

**How do different optimization algorithms perform when training the CNN surrogate model on the airfoil ice-acceleration dataset under the SIMM physics-informed loss?**

Specifically, we compare five optimizers:
1. **Vanilla Stochastic Gradient Descent (SGD)**
2. **Stochastic Gradient Descent with Momentum (SGD+Momentum)**
3. **Adaptive Gradient Algorithm (AdaGrad)**
4. **Root Mean Square Propagation (RMSprop)**
5. **Adaptive Moment Estimation (Adam)**

## Background and Optimizer Formulations

All optimizers update model parameter tensors $\theta_t$ using gradients $g_t = \nabla_\theta \mathcal{L}(\theta_t)$ computed across mini-batches synchronized with MPI all-reduce across processes:

| Optimizer | Update Rule | Key Hyperparameters | Characteristics |
|---|---|---|---|
| **SGD** | $\theta_{t+1} = \theta_t - \gamma g_t$ | $\gamma$ (learning rate) | Basic gradient descent; can oscillate in ravines or stall at saddle points. |
| **SGD+Momentum** | $v_{t} = \mu v_{t-1} + g_t$<br>$\theta_{t+1} = \theta_t - \gamma v_t$ | $\gamma$, $\mu = 0.9$ (momentum) | Accumulates velocity in consistent directions; accelerates through flat areas and dampens oscillations. |
| **AdaGrad** | $G_t = G_{t-1} + g_t^2$<br>$\theta_{t+1} = \theta_t - \frac{\gamma}{\sqrt{G_t + \epsilon}} g_t$ | $\gamma$, $\epsilon = 10^{-8}$ | Adapts learning rates per-parameter based on historical squared gradients; monotonic learning rate decay. |
| **RMSprop** | $v_t = \rho v_{t-1} + (1 - \rho) g_t^2$<br>$\theta_{t+1} = \theta_t - \frac{\gamma}{\sqrt{v_t + \epsilon}} g_t$ | $\gamma$, $\rho = 0.9$ (decay rate), $\epsilon = 10^{-8}$ | Uses exponentially decaying average of squared gradients to resolve AdaGrad's premature learning rate decay. |
| **Adam** | $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$<br>$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$<br>$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$<br>$\theta_{t+1} = \theta_t - \frac{\gamma}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$ | $\gamma$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$ | Combines momentum (first moments) with adaptive scale (second moments) with bias correction. |

## Experimental Design

### 1. Isolated State per Candidate and Fold
As required by `docs/cross_validation.md`, every candidate and fold constructs a brand new optimizer instance using `OptimizerRecipe`. Optimizer internal states (velocity vectors $v_t$, accumulated squared gradients $G_t$, moving averages $m_t, v_t$, and timesteps $t$) are never shared between folds or candidates.

### 2. Fair Cross-Validation Setup
- **Dataset**: `dataset/cnn_dataset_train.npz` (1713 samples)
- **Splitter**: `RandomKFold(k=5, shuffle=true, seed=42)`
- **Common baseline across all candidates**:
  - Exact same winning architecture (`conv5x5-dense-1024-512-256-128`): `Conv2D(8, 5x5, stride 5)` -> `LeakyReLU(0.05)` -> `Flatten` -> `Dense(1024)` -> `LeakyReLU` -> `Dense(512)` -> `LeakyReLU` -> `Dense(256)` -> `LeakyReLU` -> `Dense(128)` -> `LeakyReLU` -> `Dense(1)`
  - Exact same loss: Data MSE + SIMM physics residual with weight `0.25`
  - Exact same global batch size: `64`
  - Exact same fold seeds and training indices
- **Selection Metric**: Sample-weighted physical-unit validation MSE ($\text{MSE}_{\text{val}}$).
- **Untouched Test Set Evaluation**: `dataset/cnn_dataset_test.npz` is evaluated only once after the best optimizer is selected by cross-validation, by refitting the winning configuration on all training data.

## Compilation

### Option A: Direct MPI Compiler (`mpicxx`)
From the repository root:
```bash
mkdir -p build/experiments

mpicxx -std=c++20 -O3 -ICNN/src \
  CNN/experiments/optimizer_comparison/main.cpp \
  CNN/src/core/*.cpp CNN/src/data/*.cpp CNN/src/layers/*.cpp \
  CNN/src/model/*.cpp CNN/src/optimizers/*.cpp \
  CNN/src/training/*.cpp CNN/src/tuning/*.cpp \
  -o build/experiments/optimizer_comparison
```

### Option B: CMake
```bash
cmake -S CNN -B build/CNN -DCMAKE_BUILD_TYPE=Release
cmake --build build/CNN --target optimizer_comparison --parallel
```

## Running the Experiment

### 1. Quick Smoke Test (2 folds, 2 epochs)
Verify the setup runs cleanly:
```bash
mpirun -n 2 ./build/experiments/optimizer_comparison --smoke
```

### 2. Standard 5-Fold Comparison (100 epochs per fold)
```bash
mpirun -n 4 ./build/experiments/optimizer_comparison \
  --mode compare \
  --folds 5 \
  --epochs 100 \
  --batch-size 64 \
  --seed 42 \
  --results-dir results/cross_validation/optimizer_comparison
```

## Generated Outputs

The experiment writes results to the specified output directory:
- `fold_results.csv`: Per-candidate and per-fold training MSE, validation MSE, and SIMM objective.
- `training_history.csv`: History at each 10-epoch validation checkpoint per fold.
- `summary.txt`: Human-readable summary of cross-validation performance and final untouched test MSE.

## Results

The recorded comparison used 5 folds, 100 epochs per fold, global batch size 64,
learning rate $10^{-5}$, and seed 42. The ranking below uses the sample-weighted physical-unit validation
MSE reported in [`run/cv_summary.csv`](../../../results/cross_validation/optimizer_comparison/run/cv_summary.csv).
The fold ranges are included to show the variability between validation splits.

| Rank | Optimizer | Mean validation MSE +/- SD | Fold range |
|---:|---|---:|---:|
| **1** | **Adam (lr = 1e-5)** | **0.004427 +/- 0.000887** | 0.002879 - 0.005044 |
| 2 | RMSprop (lr = 1e-5) | 0.005216 +/- 0.000653 | 0.004518 - 0.006173 |
| 3 | AdaGrad (lr = 1e-5) | 0.006185 +/- 0.001238 | 0.004239 - 0.007439 |
| 4 | SGD + Momentum (lr = 1e-5, m = 0.9) | 0.007151 +/- 0.001412 | 0.004791 - 0.008473 |
| 5 | SGD (lr = 1e-5) | 0.017927 +/- 0.006229 | 0.013063 - 0.028647 |

Adam was selected for the final refit. It achieved the lowest mean validation
error with strong fold consistency, approximately 15.1% lower than RMSprop and
38.1% lower than SGD with momentum. Plain SGD struggled at this learning rate,
yielding the highest validation error and fold variance.

After selecting Adam using cross-validation, the model was refit on all training
data and evaluated once on the untouched test set. The resulting physical-unit
test MSE was **0.00365482**. The complete recorded summary is available in
[`summary.txt`](../../../results/cross_validation/optimizer_comparison/summary.txt),
with comparison plots in
[`plots/`](../../../results/cross_validation/optimizer_comparison/plots/).

## Analysis and Plotting

To analyze the resulting CSVs and plot boxplots and learning curves:
```bash
python3 CNN/experiments/optimizer_comparison/analyze.py \
  --results-dir results/cross_validation/optimizer_comparison
```
