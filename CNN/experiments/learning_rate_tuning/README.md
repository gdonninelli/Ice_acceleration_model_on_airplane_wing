# Learning Rate Schedule Tuning Experiment

Cross-validation experiment investigating the impact of learning rate magnitudes and dynamic scheduling policies on the CNN surrogate model trained with the winning **Adam** optimizer under the **SIMM physics-informed loss**, built on the typed `CrossValidator` and `ParameterGrid` infrastructure documented in [`docs/cross_validation.md`](../../../docs/cross_validation.md).

## Scientific Question

**How do different initial learning rate scales ($10^{-5}, 10^{-3}, 10^{-1}$) and dynamic scheduling strategies (Constant, Step Decay, Cosine Annealing, and Warmup + Cosine Decay) affect training stability, convergence speed, and generalization error on the airfoil ice-acceleration dataset?**

Following the sequential hyperparameter tuning workflow:
1. **CNN Topology**: Fixed to the winner from topology search (`conv5x5-dense-1024-512-256-128`).
2. **Optimizer**: Fixed to the winner from optimizer comparison (**Adam** with $\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$).
3. **Loss**: SIMM physics-informed residual weight fixed to $0.25$.

## Background and Schedule Formulations

The learning rate $\eta_e$ at epoch $e \in [0, E-1]$ (with total epochs $E$) governs the step size in parameter space:

$$\theta_{t+1} = \theta_t - \eta_e \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

### 1. Constant Learning Rate
$$\eta_e = \eta_0 \quad \text{for all } e \in [0, E-1]$$
Serves as the static baseline across three orders of magnitude ($\eta_0 \in \{10^{-5}, 10^{-3}, 10^{-1}\}$).

### 2. Step Decay (Multi-Step)
$$\eta_e = \eta_0 \times \gamma^{\lfloor e / S \rfloor}$$
- **$S=30, \gamma=0.1$**: Coarse, aggressive reductions at epochs 30, 60, 90 (3 drops).
- **$S=15, \gamma=0.5$**: Frequent, smoother reductions at epochs 15, 30, 45, 60, 75, 90 (6 drops).

### 3. Cosine Annealing Decay
$$\eta_e = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{e}{E-1}\pi\right)\right)$$
Smooth, continuous monotonic decrease from $\eta_{\max} = 10^{-1}$ to $\eta_{\min} = 10^{-5}$ over the full 100-epoch horizon.

### 4. Warm-up followed by Cosine Decay
$$\eta_e = \begin{cases}
\eta_{\text{start}} + \frac{e}{E_{\text{warmup}}}(\eta_{\text{peak}} - \eta_{\text{start}}) & \text{if } e < E_{\text{warmup}} \\ 
\eta_{\min} + \frac{1}{2}(\eta_{\text{peak}} - \eta_{\min})\left(1 + \cos\left(\frac{e - E_{\text{warmup}}}{E - E_{\text{warmup}} - 1}\pi\right)\right) & \text{if } e \ge E_{\text{warmup}} 
\end{cases}$$
Linearly ramps from $\eta_{\text{start}} = 10^{-5}$ to $\eta_{\text{peak}} = 10^{-1}$ over $E_{\text{warmup}} = 5$ epochs to prevent initial gradient instability, then anneals smoothly to $\eta_{\min} = 10^{-5}$.

---

## Evaluated Candidate Grid (11 Configurations)

| Candidate ID | Schedule Type | Base / Peak $\eta$ | Min $\eta$ | Step / Warmup Parameter |
|:---|:---|:---:|:---:|:---|
| `const_1e-5` | Constant | $10^{-5}$ | — | Fixed baseline rate |
| `const_1e-3` | Constant | $10^{-3}$ | — | Intermediate baseline |
| `const_1e-1` | Constant | $10^{-1}$ | — | Aggressive baseline |
| `step_s30_1e-1` | Step Decay | $10^{-1}$ | — | Step $S=30$ epochs, $\gamma=0.1$ |
| `step_s30_1e-3` | Step Decay | $10^{-3}$ | — | Step $S=30$ epochs, $\gamma=0.1$ |
| `step_s30_1e-5` | Step Decay | $10^{-5}$ | — | Step $S=30$ epochs, $\gamma=0.1$ (control) |
| `step_s15_1e-1` | Step Decay | $10^{-1}$ | — | Step $S=15$ epochs, $\gamma=0.5$ |
| `step_s15_1e-3` | Step Decay | $10^{-3}$ | — | Step $S=15$ epochs, $\gamma=0.5$ |
| `step_s15_1e-5` | Step Decay | $10^{-5}$ | — | Step $S=15$ epochs, $\gamma=0.5$ (control) |
| `cosine_1e-1` | Cosine Annealing | $10^{-1}$ | $10^{-5}$ | $E=100$ epochs period |
| `warmup_cosine_1e-1` | Warmup + Cosine | $10^{-1}$ | $10^{-5}$ | 5 epochs warmup ($10^{-5} \to 10^{-1}$) |

---

## Experimental Setup

- **Dataset**: `dataset/cnn_dataset_train.npz` (1,713 samples).
- **Cross-Validation**: 5-Fold Stratified `RandomKFold(k=5, shuffle=true, seed=42)`.
- **Training Epochs**: 100 epochs per fold.
- **Global Batch Size**: 64 (MPI distributed).
- **Selection Metric**: Sample-weighted physical-unit validation MSE ($\text{MSE}_{\text{val}}$).
- **Untouched Test Set Evaluation**: Evaluated strictly once on `dataset/cnn_dataset_test.npz` by refitting the winning configuration on the full training dataset.

---

## Compilation

### Option A: CMake (Recommended)
```bash
cmake -S CNN -B build/CNN -DCMAKE_BUILD_TYPE=Release
cmake --build build/CNN --target learning_rate_tuning --parallel
```

### Option B: Direct MPI Compiler (`mpicxx`)
```bash
mpicxx -std=c++20 -O3 -ICNN/src \
  CNN/experiments/learning_rate_tuning/main.cpp \
  CNN/src/core/*.cpp CNN/src/data/*.cpp CNN/src/layers/*.cpp \
  CNN/src/model/*.cpp CNN/src/optimizers/*.cpp \
  CNN/src/training/*.cpp CNN/src/tuning/*.cpp \
  -o build/CNN/experiments/learning_rate_tuning
```

---

## Running the Experiment

### 1. Sanity Smoke Test (2 folds, 2 epochs)
```bash
mpirun -n 2 ./build/CNN/experiments/learning_rate_tuning --smoke
```

### 2. Standard 5-Fold Tuning Run (100 epochs per fold)
```bash
mpirun -n 64 ./build/CNN/experiments/learning_rate_tuning \
  --mode tune \
  --folds 5 \
  --epochs 100 \
  --batch-size 64 \
  --seed 42 \
  --physics-weight 0.25 \
  --results-dir results/cross_validation/learning_rate_tuning
```

---

## Generated Artifacts

Results are written to the specified `--results-dir`:
- `fold_results.csv`: Per-candidate and per-fold training MSE, validation MSE, and SIMM objective.
- `training_history.csv`: History at every 10-epoch validation checkpoint per fold.
- `cv_summary.csv`: Summary statistics (mean, stddev, min, max validation MSE) across all candidates.
- `summary.txt`: Human-readable summary of cross-validation ranking and untouched test MSE.

---

## Results

The recorded experiment evaluated all 11 learning rate schedule candidates across 5 folds and 100 epochs per fold with batch size 64, physics weight 0.25, and seed 42.

### Cross-Validation Summary Table

| Rank | Candidate ID | Schedule Type | Base $\eta$ | Mean Validation MSE $\pm$ SD | Fold Range ($\text{MSE}_{\text{val}}$) | Status |
|:---:|:---|:---|:---:|:---:|:---:|:---:|
| **1** | **`const_1e-3`** | **Constant** | **$10^{-3}$** | **$0.003609 \pm 0.001335$** | **0.002092 – 0.006094** | **Selected Winner** |
| 2 | `step_s15_1e-3` | Step Decay ($S=15, \gamma=0.5$) | $10^{-3}$ | $0.004005 \pm 0.000704$ | 0.002826 – 0.004779 | Stable |
| 3 | `step_s30_1e-3` | Step Decay ($S=30, \gamma=0.1$) | $10^{-3}$ | $0.004072 \pm 0.000785$ | 0.002865 – 0.004955 | Stable |
| 4 | `const_1e-5` | Constant (Baseline) | $10^{-5}$ | $0.004419 \pm 0.000790$ | 0.002880 – 0.005001 | Stable |
| 5 | `step_s30_1e-5` | Step Decay ($S=30, \gamma=0.1$) | $10^{-5}$ | $0.005034 \pm 0.000901$ | 0.003434 – 0.005786 | Stable |
| 6 | `step_s15_1e-5` | Step Decay ($S=15, \gamma=0.5$) | $10^{-5}$ | $0.005094 \pm 0.000905$ | 0.003484 – 0.005861 | Stable |
| 7 | `cosine_1e-1` | Cosine Annealing | $10^{-1}$ | $5.58 \times 10^{8} \pm 2.25 \times 10^{8}$ | $2.09 \times 10^8$ – $7.95 \times 10^8$ | Diverged |
| 8 | `warmup_cosine_1e-1` | Warmup + Cosine | $10^{-1}$ | $9.05 \times 10^{8} \pm 8.87 \times 10^{8}$ | $1.21 \times 10^8$ – $2.51 \times 10^9$ | Diverged |
| 9 | `step_s15_1e-1` | Step Decay ($S=15, \gamma=0.5$) | $10^{-1}$ | $5.01 \times 10^{9} \pm 2.96 \times 10^{9}$ | $1.53 \times 10^9$ – $9.32 \times 10^9$ | Diverged |
| 10 | `step_s30_1e-1` | Step Decay ($S=30, \gamma=0.1$) | $10^{-1}$ | $9.45 \times 10^{9} \pm 4.73 \times 10^{9}$ | $3.20 \times 10^9$ – $1.63 \times 10^{10}$ | Diverged |
| 11 | `const_1e-1` | Constant | $10^{-1}$ | $2.13 \times 10^{17} \pm 3.12 \times 10^{17}$ | $1.84 \times 10^{16}$ – $8.31 \times 10^{17}$ | Diverged |

---

## Scientific Analysis & Model Selection

1. **Optimal Learning Rate Magnitude ($\eta = 10^{-3}$):**
   - Constant Adam at $\eta = 10^{-3}$ achieved the best overall cross-validation performance ($0.003609$), representing an **18.3% error reduction** over the initial baseline $\eta = 10^{-5}$ ($0.004419$).
   - The larger step size facilitates rapid descent across the non-convex SIMM physics-regularized loss surface without sacrificing convergence precision.

2. **Impact of Dynamic Schedules:**
   - For $\eta_0 = 10^{-3}$, decaying the learning rate too early (`step_s15` or `step_s30`) slightly increased validation error ($0.004005$ and $0.004072$). Within a 100-epoch budget, early rate drops prematurely stall parameter updates before reaching the optimal basin.
   - For $\eta_0 = 10^{-5}$, further decay worsened performance ($>0.0050$), as the step size became negligibly small ($\le 10^{-6}$).

3. **Numerical Instability at $\eta \ge 10^{-1}$:**
   - All schedules operating at or ramping to $10^{-1}$ suffered catastrophic gradient explosion within the first 1–6 epochs, causing validation MSE to exceed $10^8 - 10^{17}$.
   - Even linear warmup over 5 epochs failed to stabilize $\eta_{\text{peak}} = 10^{-1}$, confirming that $10^{-1}$ exceeds the maximum stable Lipschitz-bounded step size for this CNN architecture and batch size.

---

## Final Untouched Test Set Evaluation

Following cross-validation model selection, the winning configuration (**`const_1e-3`**) was refit on the full training set (1,713 samples) and evaluated strictly once on the untouched test dataset (`dataset/cnn_dataset_test.npz`):

- **Selected Schedule:** `const_1e-3` (Adam, $\eta = 10^{-3}$, $\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$)
- **Cross-Validation Validation MSE:** **`0.00360884` $\pm$ `0.00133457`**
- **Final Untouched Test Physical MSE:** **`0.00325384`**

---

## Diagnostic Plotting

To render diagnostic plots (effective learning rate schedules, gradient depth profiles, parameter update ratios, and activation heatmaps) for representative folds and epochs:

```bash
python3 CNN/analysis/plot_training_diagnostics.py \
  --input results/cross_validation/learning_rate_tuning/learning_rate_tuning/run \
  --output-dir results/cross_validation/learning_rate_tuning/plots \
  --fold 0 \
  --epoch 1 --epoch 10 --epoch 25 --epoch 50 --epoch 100
```

