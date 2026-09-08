#!/usr/bin/env python3
"""Plot unbalanced-batch instability: trial 1 vs trial 2 (ordinary training).

Reads epoch_metrics.csv, gradient_norms.csv and parameter_update_ratios.csv
from the two seed-42 runs with a 200-epoch budget and global batch size 64.
Writes two report figures showing that balancing the batches reduced recurrent
physical-MSE excursions.

Trial 1 (trial-1_slurm-56688186): original range-based batching, 24 batches
of 64 plus a final 6-sample tail.
Trial 2 (trial-2_slurm-56688187): balanced batching, 17 batches of 62 and
8 batches of 61, same 25 optimizer steps per epoch, stopping at epoch 191.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TRIAL1 = Path("results/ordinary-training/trial-1_slurm-56688186")
TRIAL2 = Path("results/ordinary-training/trial-2_slurm-56688187")
DEFAULT_OUT = Path("Report/Images/Chapter04/final_training")


def read_mse(path: Path) -> tuple[list[int], list[float], list[float]]:
    epochs: list[int] = []
    train: list[float] = []
    val: list[float] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                e = int(row["epoch"])
                t = float(row["training_physical_mse"])
                v = float(row["validation_physical_mse"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid row in {path}: {row}") from error
            if not all(math.isfinite(x) for x in (t, v)):
                raise ValueError(f"Non-finite row in {path}: {row}")
            epochs.append(e)
            train.append(t)
            val.append(v)
    if not epochs or epochs != list(range(1, len(epochs) + 1)):
        raise ValueError(f"Expected contiguous completed epochs in {path}")
    return epochs, train, val


def read_gradient_depth(path: Path, epoch: int) -> tuple[list[int], list[float], list[float]]:
    layers: list[int] = []
    rms: list[float] = []
    mx: list[float] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["parameter_scope"] != "all":
                continue
            if int(row["epoch"]) != epoch:
                continue
            layers.append(int(row["layer_index"]))
            rms.append(float(row["rms_norm"]))
            mx.append(float(row["maximum_norm"]))
    if not layers:
        raise ValueError(f"No scope=all rows for epoch {epoch} in {path}")
    order = sorted(range(len(layers)), key=lambda i: layers[i])
    return [layers[i] for i in order], [rms[i] for i in order], [mx[i] for i in order]


def read_update_layer(path: Path, layer_index: int) -> tuple[list[int], list[float]]:
    epochs: list[int] = []
    ratio: list[float] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["parameter_scope"] != "all":
                continue
            if int(row["layer_index"]) != layer_index:
                continue
            nz = int(row["near_zero_denominator_steps"])
            if nz != 0:
                raise ValueError(f"Unexpected near-zero denominator in {path}: {row}")
            epochs.append(int(row["epoch"]))
            ratio.append(float(row["mean_ratio"]))
    if not epochs or epochs != list(range(1, len(epochs) + 1)):
        raise ValueError(f"Expected contiguous update epochs in {path}")
    return epochs, ratio


def plot_mse_comparison(
    e1: list[int], t1: list[float], v1: list[float],
    e2: list[int], t2: list[float], v2: list[float],
    output: Path,
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True, sharey=True)
    figure.suptitle("Unbalanced tail versus balanced batches (global batch size 64)")
    for axis, epochs, train, val, title in zip(
        axes,
        (e1, e2),
        (t1, t2),
        (v1, v2),
        (
            "Trial 1: 24 batches of 64 plus one 6-sample tail",
            "Trial 2: 17 batches of 62 and 8 batches of 61",
        ),
    ):
        axis.plot(epochs, train, color="tab:blue", linewidth=1.1,
                  label="Training physical MSE")
        axis.plot(epochs, val, color="tab:orange", linewidth=1.1,
                  label="Validation physical MSE")
        axis.set_title(title, fontsize=10)
        axis.set_ylabel("Physical MSE")
        axis.set_ylim(bottom=0.0, top=1.15 * max(t1 + v1 + t2 + v2))
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False, loc="upper right", fontsize=8)
    # Annotate the representative spike present only in trial 1.
    idx84 = e1.index(84)
    axes[0].annotate(
        f"epoch 84: train {t1[idx84]:.4f}, val {v1[idx84]:.4f}",
        xy=(84, v1[idx84]), xytext=(0.48, 0.72), textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "0.3"},
        fontsize=8, color="0.2",
    )
    axes[1].set_xlabel("Completed epoch")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_gradient_update(
    layers1: list[int], rms1: list[float],
    layers2: list[int], rms2: list[float],
    ue1: list[int], u1: list[float], ue2: list[int], u2: list[float],
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.2))
    figure.suptitle("Gradient and update signature of the epoch-84 excursion")
    left = axes[0]
    left.plot(layers1, rms1, marker="o", color="tab:red", linewidth=1.4,
              label="Trial 1 unbalanced, epoch 84")
    left.plot(layers2, rms2, marker="s", color="tab:green", linewidth=1.4,
              label="Trial 2 balanced, epoch 84")
    left.set_yscale("log")
    left.set_xlabel("Stable model layer index")
    left.set_ylabel("RMS gradient norm (scope all, log scale)")
    left.set_title("Depth profile at epoch 84")
    left.set_xticks(layers1)
    left.grid(True, which="both", alpha=0.25)
    left.legend(frameon=False, fontsize=8)
    right = axes[1]
    right.plot(ue1, u1, color="tab:red", linewidth=1.3,
               label="Trial 1 unbalanced, dense layer 4")
    right.plot(ue2, u2, color="tab:green", linewidth=1.3,
               label="Trial 2 balanced, dense layer 4")
    right.set_yscale("log")
    right.set_xlabel("Completed epoch")
    right.set_ylabel("Actual mean update ratio (scope all, log scale)")
    right.set_title("First dense layer update trajectory")
    right.grid(True, which="both", alpha=0.25)
    right.legend(frameon=False, fontsize=8)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot trial-1 unbalanced-batch instability against trial-2 balanced control."
    )
    parser.add_argument("--trial1", type=Path, default=TRIAL1)
    parser.add_argument("--trial2", type=Path, default=TRIAL2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    e1, t1, v1 = read_mse(args.trial1 / "epoch_metrics.csv")
    e2, t2, v2 = read_mse(args.trial2 / "epoch_metrics.csv")
    layers1, rms1, _ = read_gradient_depth(args.trial1 / "gradient_norms.csv", 84)
    layers2, rms2, _ = read_gradient_depth(args.trial2 / "gradient_norms.csv", 84)
    # Stable trainable layer 4 is the first dense layer DenseLayer(7202, 1024).
    ue1, u1 = read_update_layer(args.trial1 / "parameter_update_ratios.csv", 4)
    ue2, u2 = read_update_layer(args.trial2 / "parameter_update_ratios.csv", 4)
    if ue1 != e1 or ue2 != e2:
        raise ValueError("Update-ratio epochs do not match each trial's MSE history")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mse_path = args.output_dir / "final_trial1_trial2_mse_comparison.png"
    diag_path = args.output_dir / "final_trial1_trial2_gradient_update.png"
    plot_mse_comparison(e1, t1, v1, e2, t2, v2, mse_path)
    plot_gradient_update(layers1, rms1, layers2, rms2, ue1, u1, ue2, u2, diag_path)
    print(f"Saved plot to: {mse_path}")
    print(f"Saved plot to: {diag_path}")


if __name__ == "__main__":
    main()
