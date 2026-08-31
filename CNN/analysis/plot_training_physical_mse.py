#!/usr/bin/env python3
"""Plot training and validation physical MSE on the same figure."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_epoch_metrics(csv_path: Path):
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    epochs = []
    train_mse = []
    val_mse = []

    for row in rows:
        try:
            epochs.append(int(row["epoch"]))
            train_mse.append(float(row["training_physical_mse"]))
            val_mse.append(float(row["validation_physical_mse"]))
        except (KeyError, TypeError, ValueError):
            continue

    if not epochs:
        raise ValueError(f"CSV file {csv_path} does not contain valid training/validation MSE columns.")

    return epochs, train_mse, val_mse


def main():
    parser = argparse.ArgumentParser(
        description="Plot the training and validation physical MSE curves on the same chart."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="results/ordinary-training/slurm-55367279/epoch_metrics.csv",
        help="Path to the epoch_metrics.csv file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Optional output image path. Defaults to the CSV directory with a plot name.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively after saving it.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    epochs, train_mse, val_mse = load_epoch_metrics(csv_path)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_mse, label="Training physical MSE", color="tab:blue", linewidth=2)
    plt.plot(epochs, val_mse, label="Validation physical MSE", color="tab:orange", linewidth=2)

    plt.title("Training vs Validation Physical MSE")
    plt.xlabel("Epoch")
    plt.ylabel("Physical MSE")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = Path(args.output) if args.output else csv_path.with_name("physical_mse_comparison.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")

    if args.show:
        plt.show()
    else:
        plt.close()

    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
