# Plot Provenance

These PNGs are historical artifacts produced by the former Python workflow.
They were moved with the experiment when the larger topology became the
canonical `regularization_tuning` experiment; they are not a fresh run of the
current C++ executable.

Fresh plots should be generated from `sweep_l1.csv`, `sweep_l2.csv`, and the
diagnostics tree with:

```bash
python3 CNN/experiments/regularization_tuning/plot_results.py \
    --results-dir results/cross_validation/regularization_tuning
```
