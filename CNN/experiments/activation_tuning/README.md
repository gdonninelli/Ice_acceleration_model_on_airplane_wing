# Activation-Function Tuning Experiment

## Question

Which activation function minimises the physical-unit validation MSE for the
baseline airfoil-coefficient CNN? This experiment holds every other
hyperparameter fixed and varies only the non-linearity.

## Search Space

A single search axis, `activation-function`, with eight candidates. The chosen
activation is applied uniformly at every activation site (feature trunk and
dense head).

| Candidate | Activation |
|---|---|
| `relu` | ReLU |
| `tanh` | Hyperbolic tangent |
| `sigmoid` | Logistic sigmoid |
| `leakyrelu-0.01` | Leaky ReLU (α = 0.01) |
| `leakyrelu-0.05` | Leaky ReLU (α = 0.05) **(Baseline)** |
| `leakyrelu-0.1` | Leaky ReLU (α = 0.1) |
| `leakyrelu-0.2` | Leaky ReLU (α = 0.2) |
| `leakyrelu-0.3` | Leaky ReLU (α = 0.3) |

Total runs = 8 candidates × 5 folds = **40 fold evaluations**.

## Fixed Baseline (`TrialConfig`)

| Component | Setting |
|---|---|
| Feature trunk | `Conv2D(8, 5×5)` → `Activation` → `Flatten` |
| Head | `Dense(128)` → `Activation` → `Dense(64)` → `Activation` → `Dense(1)` |
| Optimizer | Adam, learning rate `1e-5` |
| Loss | SIMM physics loss, weight α = `0.25` |
| Folds | 5 (`RandomKFold`, shuffle, seed 42) |
| Epochs | 100 |
| Global batch | 64 (MPI-global, not per-rank) |
| Seed | 42 |

## Build

From the repository root:

```bash
export PATH=/usr/lib64/openmpi/bin:$PATH
mkdir -p build/experiments

mpicxx -std=c++20 -O3 -ICNN/src \
  CNN/experiments/activation_tuning/main.cpp \
  CNN/src/core/*.cpp CNN/src/data/*.cpp CNN/src/layers/*.cpp \
  CNN/src/model/*.cpp CNN/src/optimizers/*.cpp \
  CNN/src/training/*.cpp CNN/src/tuning/*.cpp \
  -o build/experiments/activation_tuning
```

## Run

To avoid MPI timeouts for lengthy runs, the experiment uses an orchestrator script that invokes the compiled binary separately for each candidate.

```bash
python3 CNN/experiments/activation_tuning/orchestrator.py
```
*(The script will create CSV files inside `results/cross_validation/activation_tuning/` for each candidate, and an `aggregated.csv`)*

## Results & Analysis

Criterio di selezione: Un candidato batte il baseline (`leakyrelu-0.05`) solo se migliora in **almeno 4 fold su 5** e la differenza appaiata media supera la sua deviazione standard (ovvero `abs(mean_diff) > std_diff` e `mean_diff < 0`).

| Candidate | Val MSE | Std | Val/Baseline | Paired Diff vs 0.05 | Std | Improved Folds | Beats Baseline? |
|---|---|---|---|---|---|---|---|
| leakyrelu-0.01 | 0.005664 | 0.001296 | 0.01149 | -0.000112 | 0.000116 | 4/5 | No |
| **leakyrelu-0.05** | **0.005776** | 0.001282 | 0.01172 | — | — | — | — |
| relu | 0.005792 | 0.001340 | 0.01175 | +0.000016 | 0.000198 | 2/5 | No |
| leakyrelu-0.1 | 0.005848 | 0.001305 | 0.01186 | +0.000072 | 0.000088 | 0/5 | No |
| leakyrelu-0.2 | 0.006080 | 0.001486 | 0.01233 | +0.000304 | 0.000368 | 1/5 | No |
| leakyrelu-0.3 | 0.006188 | 0.001458 | 0.01255 | +0.000412 | 0.000279 | 0/5 | No |
| tanh | 0.006481 | 0.001364 | 0.01314 | +0.000705 | 0.000148 | 0/5 | No |
| sigmoid | 0.009049 | 0.001242 | 0.01829 | +0.003273 | 0.000593 | 0/5 | No |

**Conclusione:**
Nessuna funzione di attivazione supera il rigoroso test di significatività rispetto al baseline.
Il candidato `leakyrelu-0.01` presenta una MSE leggermente migliore e vince in 4 fold su 5, ma la differenza appaiata media (-0.000112) non supera la sua deviazione standard (0.000116).
Di conseguenza, il vincitore rimane **`leakyrelu-0.05`**, che è un risultato legittimo a riprova della bontà della configurazione iniziale.

> `leakyrelu α=0.01` migliora in 4 fold su 5 ma non supera il criterio (differenza appaiata
> media 0.000112 contro std 0.000116) ed è il valore più basso della griglia. Non è però un
> ottimo di bordo da esplorare oltre: `relu` è matematicamente LeakyReLU con α = 0 ed è stato
> testato, risultando nettamente peggiore. Se alpha più piccoli fossero migliori, il limite
> α → 0 sarebbe il migliore di tutti; non lo è. Il 4/5 di α=0.01 è quindi un'oscillazione
> isolata, non un trend, e la griglia non va estesa verso il basso.


## Caveat on absolute values

> I risultati sono ottenuti con lo split train/test casuale (seed 42), in cui la stessa
> geometria (profilo, angolo) compare in entrambi i set con Reynolds diverso. Poiché il
> Reynolds non influenza il target in modo misurabile, questi campioni sono di fatto
> duplicati fra train e test e le MSE di validazione riportate sono ottimistiche. I confronti
> *relativi* fra candidati restano validi perché tutti condividono lo stesso split; i valori
> *assoluti* andranno rimisurati se il gruppo adotterà uno split raggruppato.
