#!/usr/bin/env python3
"""
Generate result plots for the regularization_tuning_bigarch sweep.

Reads diagnostics from RESULTS_DIR (default: $WORK/results/sweep or
results/cross_validation/regularization_tuning_bigarch relative to the
repository root) and writes PNG files to OUTDIR (default: plots/ next to
this script).

Usage:
    python3 plot_results.py [--results-dir PATH] [--out-dir PATH]
"""

import argparse
import csv
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Candidate grids and display config
# ---------------------------------------------------------------------------

L1_LAMBDAS = [0.0, 6.75e-7, 2.13e-6, 6.75e-6, 2.13e-5, 6.75e-5, 2.13e-4, 6.75e-4]
L2_LAMBDAS = [0.0, 1e-4, 3.16e-4, 1e-3, 3.16e-3, 1e-2, 3.16e-2, 1e-1]

def fmt_lam(v):
    if v == 0.0:
        return "0"
    return f"{v:.2e}"

def label_of(axis, lam):
    s = "0" if lam == 0.0 else f"{lam:.0e}"
    return f"{axis}_{s}"

CONFIG_STR = (
    "conv5x5-dense-{1024,512,256,128}  Adam lr=1e-5  "
    "5 folds  100 epochs  seed=42"
)

FONT = {"fontsize": 13}
TITLE_FONT = {"fontsize": 11}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")

def std(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

def read_fold_csv(results_dir, label):
    path = os.path.join(results_dir, f"{label}.csv")
    with open(path) as f:
        return list(csv.DictReader(f))

def read_epoch_metrics(results_dir, label):
    """Return {epoch: [val_mse fold 0..4]} averaged over folds."""
    # for l2_0, diagnostics live under l1_0
    diag_label = "l1_0" if label == "l2_0" else label
    diag_base = os.path.join(
        results_dir, "regularization_tuning_bigarch", diag_label, "candidate_000"
    )
    epoch_vals = {}
    for fold in range(5):
        fpath = os.path.join(diag_base, f"fold_{fold:03d}", "epoch_metrics.csv")
        if not os.path.isfile(fpath):
            continue
        with open(fpath) as f:
            for row in csv.DictReader(f):
                v = row.get("validation_physical_mse", "").strip()
                if v:
                    ep = int(row["epoch"])
                    epoch_vals.setdefault(ep, []).append(float(v))
    return epoch_vals

def read_grad_norms(results_dir, label, scope="all"):
    """Return {epoch: mean_norm_across_folds_and_layers} for gradient_norms.csv."""
    diag_label = "l1_0" if label == "l2_0" else label
    diag_base = os.path.join(
        results_dir, "regularization_tuning_bigarch", diag_label, "candidate_000"
    )
    epoch_data = {}
    for fold in range(5):
        fpath = os.path.join(diag_base, f"fold_{fold:03d}", "gradient_norms.csv")
        if not os.path.isfile(fpath):
            continue
        with open(fpath) as f:
            for row in csv.DictReader(f):
                ep = int(row["epoch"])
                epoch_data.setdefault(ep, []).append(float(row["mean_norm"]))
    return {ep: mean(vs) for ep, vs in epoch_data.items()}

def read_update_ratios(results_dir, label, scope="weights"):
    """Return {epoch: mean_ratio_across_folds_and_layers} for given scope."""
    diag_label = "l1_0" if label == "l2_0" else label
    diag_base = os.path.join(
        results_dir, "regularization_tuning_bigarch", diag_label, "candidate_000"
    )
    epoch_data = {}
    for fold in range(5):
        fpath = os.path.join(
            diag_base, f"fold_{fold:03d}", "parameter_update_ratios.csv"
        )
        if not os.path.isfile(fpath):
            continue
        with open(fpath) as f:
            for row in csv.DictReader(f):
                if row.get("parameter_scope", "") != scope:
                    continue
                ep = int(row["epoch"])
                val = row.get("mean_ratio", "").strip()
                if val:
                    epoch_data.setdefault(ep, []).append(float(val))
    return {ep: mean(vs) for ep, vs in epoch_data.items()}

# ---------------------------------------------------------------------------
# Plot 1: Val MSE vs lambda (L1 and L2, two panels)
# ---------------------------------------------------------------------------

def plot_val_mse(results_dir, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Validation MSE vs regularization strength\n{CONFIG_STR}", **TITLE_FONT)

    for ax, axis, lambdas in zip(axes, ["l1", "l2"], [L1_LAMBDAS, L2_LAMBDAS]):
        ref_label = label_of(axis, 0.0)
        ref_rows = read_fold_csv(results_dir, ref_label)
        ref_mean = mean([float(r["val_mse"]) for r in ref_rows])

        means, stds, xs = [], [], []
        for lam in lambdas:
            lbl = label_of(axis, lam)
            rows = read_fold_csv(results_dir, lbl)
            vals = [float(r["val_mse"]) for r in rows]
            means.append(mean(vals))
            stds.append(std(vals))
            xs.append(lam if lam > 0 else lambdas[1] * 0.3)

        ax.axhline(ref_mean, color="gray", linestyle="--", linewidth=1,
                   label=f"λ=0 mean ({ref_mean:.5f})")
        ax.errorbar(xs[1:], means[1:], yerr=stds[1:], fmt="o-", capsize=4,
                    linewidth=1.5, markersize=5,
                    label=f"{axis.upper()} candidates")
        ax.plot(xs[:1], means[:1], "s", color="tab:orange", markersize=7,
                label=f"λ=0 ({means[0]:.5f})")
        ax.set_xscale("log")
        ax.set_xlabel(f"λ ({axis.upper()})", **FONT)
        ax.set_ylabel("Mean val MSE (physical units)", **FONT)
        ax.set_title(f"{axis.upper()} axis", **FONT)
        ax.legend(fontsize=10)
        ax.tick_params(labelsize=11)
        ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(out_dir, "plot1_val_mse_vs_lambda.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

# ---------------------------------------------------------------------------
# Plot 2: Paired difference vs lambda=0
# ---------------------------------------------------------------------------

def plot_paired_diff(results_dir, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Paired validation MSE difference vs λ=0 (positive = improvement)\n{CONFIG_STR}",
        **TITLE_FONT,
    )

    ref_rows = read_fold_csv(results_dir, "l1_0")
    ref_vals = [float(r["val_mse"]) for r in ref_rows]

    for ax, axis, lambdas in zip(axes, ["l1", "l2"], [L1_LAMBDAS, L2_LAMBDAS]):
        diff_means, diff_stds, xs = [], [], []
        for lam in lambdas[1:]:
            lbl = label_of(axis, lam)
            rows = read_fold_csv(results_dir, lbl)
            vals = [float(r["val_mse"]) for r in rows]
            diffs = [ref_vals[i] - vals[i] for i in range(5)]
            diff_means.append(mean(diffs))
            diff_stds.append(std(diffs))
            xs.append(lam)

        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.errorbar(xs, diff_means, yerr=diff_stds, fmt="o-", capsize=4,
                    linewidth=1.5, markersize=5, color="tab:blue")
        ax.fill_between(xs,
                        [m - s for m, s in zip(diff_means, diff_stds)],
                        [m + s for m, s in zip(diff_means, diff_stds)],
                        alpha=0.15, color="tab:blue")
        ax.set_xscale("log")
        ax.set_xlabel(f"λ ({axis.upper()})", **FONT)
        ax.set_ylabel("Δ val MSE (ref − λ)", **FONT)
        ax.set_title(f"{axis.upper()} axis — paired diff ± fold std", **FONT)
        ax.tick_params(labelsize=11)
        ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(out_dir, "plot2_paired_diff.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

# ---------------------------------------------------------------------------
# Plot 3: Validation loss curves per epoch
# ---------------------------------------------------------------------------

def plot_val_curves(results_dir, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Validation MSE per epoch (fold-average ± fold std)\n{CONFIG_STR}",
        **TITLE_FONT,
    )

    cmap_l1 = plt.cm.Blues
    cmap_l2 = plt.cm.Oranges

    for ax, axis, lambdas, cmap in zip(
        axes, ["l1", "l2"], [L1_LAMBDAS, L2_LAMBDAS], [cmap_l1, cmap_l2]
    ):
        n = len(lambdas)
        colors = [cmap(0.35 + 0.65 * i / max(n - 1, 1)) for i in range(n)]

        for i, lam in enumerate(lambdas):
            lbl = label_of(axis, lam)
            epoch_vals = read_epoch_metrics(results_dir, lbl)
            if not epoch_vals:
                continue
            epochs = sorted(epoch_vals.keys())
            fold_means = [mean(epoch_vals[ep]) for ep in epochs]
            fold_stds = [std(epoch_vals[ep]) for ep in epochs]
            ax.plot(epochs, fold_means, color=colors[i], linewidth=1.5,
                    label=f"λ={fmt_lam(lam)}")
            ax.fill_between(
                epochs,
                [m - s for m, s in zip(fold_means, fold_stds)],
                [m + s for m, s in zip(fold_means, fold_stds)],
                alpha=0.12, color=colors[i],
            )

        ax.set_xlabel("Epoch", **FONT)
        ax.set_ylabel("Val MSE (physical units)", **FONT)
        ax.set_title(f"{axis.upper()} axis", **FONT)
        ax.legend(fontsize=8, ncol=2)
        ax.tick_params(labelsize=11)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(out_dir, "plot3_val_curves_epoch.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

# ---------------------------------------------------------------------------
# Plot 4: Sum-of-squared weights (Sigma w^2) vs lambda
# ---------------------------------------------------------------------------

def plot_sum_w2(results_dir, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(
        f"Σw² (sum of squared weights at end of training) vs λ\n{CONFIG_STR}",
        **TITLE_FONT,
    )

    ref_rows = read_fold_csv(results_dir, "l1_0")
    ref_sw2 = mean([float(r["sum_w2"]) for r in ref_rows])

    for axis, lambdas, color, marker in [
        ("l1", L1_LAMBDAS, "tab:blue", "o"),
        ("l2", L2_LAMBDAS, "tab:orange", "s"),
    ]:
        xs, ys, errs = [], [], []
        for lam in lambdas[1:]:
            lbl = label_of(axis, lam)
            rows = read_fold_csv(results_dir, lbl)
            sw2s = [float(r["sum_w2"]) for r in rows]
            xs.append(lam)
            ys.append(mean(sw2s))
            errs.append(std(sw2s))
        ax.errorbar(xs, ys, yerr=errs, fmt=f"{marker}-", capsize=4,
                    linewidth=1.5, markersize=5, color=color, label=f"{axis.upper()}")

    ax.axhline(ref_sw2, color="gray", linestyle="--", linewidth=1,
               label=f"λ=0 reference ({ref_sw2:.0f})")
    ax.set_xscale("log")
    ax.set_xlabel("λ", **FONT)
    ax.set_ylabel("Mean Σw² (fold average)", **FONT)
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=11)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(out_dir, "plot4_sum_w2_vs_lambda.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

# ---------------------------------------------------------------------------
# Plot 5: Gradient norms and weight-update ratios over training epochs
# ---------------------------------------------------------------------------

def plot_diagnostics_over_epochs(results_dir, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Training diagnostics over epochs\n{CONFIG_STR}",
        fontsize=11,
    )

    cmap_l1 = plt.cm.Blues
    cmap_l2 = plt.cm.Oranges

    for col, (axis, lambdas, cmap) in enumerate(
        [("l1", L1_LAMBDAS, cmap_l1), ("l2", L2_LAMBDAS, cmap_l2)]
    ):
        n = len(lambdas)
        colors = [cmap(0.35 + 0.65 * i / max(n - 1, 1)) for i in range(n)]

        ax_grad = axes[0][col]
        ax_ratio = axes[1][col]

        for i, lam in enumerate(lambdas):
            lbl = label_of(axis, lam)
            grad_data = read_grad_norms(results_dir, lbl)
            ratio_data = read_update_ratios(results_dir, lbl, scope="weights")

            if grad_data:
                eps_g = sorted(grad_data.keys())
                ax_grad.plot(eps_g, [grad_data[e] for e in eps_g],
                             color=colors[i], linewidth=1.2,
                             label=f"λ={fmt_lam(lam)}")
            if ratio_data:
                eps_r = sorted(ratio_data.keys())
                ax_ratio.plot(eps_r, [ratio_data[e] for e in eps_r],
                              color=colors[i], linewidth=1.2,
                              label=f"λ={fmt_lam(lam)}")

        ax_grad.set_title(f"{axis.upper()} — gradient norm (mean, all layers)", **FONT)
        ax_grad.set_xlabel("Epoch", **FONT)
        ax_grad.set_ylabel("Mean grad norm", **FONT)
        ax_grad.legend(fontsize=7, ncol=2)
        ax_grad.tick_params(labelsize=10)
        ax_grad.grid(True, alpha=0.3)

        ax_ratio.set_title(f"{axis.upper()} — update ratio (weights scope)", **FONT)
        ax_ratio.set_xlabel("Epoch", **FONT)
        ax_ratio.set_ylabel("|Δw| / |w|  (weights scope)", **FONT)
        ax_ratio.legend(fontsize=7, ncol=2)
        ax_ratio.tick_params(labelsize=10)
        ax_ratio.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(out_dir, "plot5_diagnostics_over_epochs.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default=os.environ.get(
            "REG_RESULTS_DIR",
            os.path.join(os.environ.get("WORK", ""), "results", "sweep")
            if os.environ.get("WORK")
            else None,
        ),
        help=(
            "Directory containing the sweep CSVs and diagnostics sub-tree. "
            "Defaults to $WORK/results/sweep if $WORK is set, or $REG_RESULTS_DIR."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots"),
        help="Output directory for PNG files.",
    )
    args = parser.parse_args()

    if not args.results_dir or not os.path.isdir(args.results_dir):
        print(
            "ERROR: results dir not found or not specified.\n"
            "Pass --results-dir PATH or set $WORK or $REG_RESULTS_DIR.",
            file=sys.stderr,
        )
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Results: {args.results_dir}")
    print(f"Output:  {args.out_dir}")
    print()

    plot_val_mse(args.results_dir, args.out_dir)
    plot_paired_diff(args.results_dir, args.out_dir)
    plot_val_curves(args.results_dir, args.out_dir)
    plot_sum_w2(args.results_dir, args.out_dir)
    plot_diagnostics_over_epochs(args.results_dir, args.out_dir)

    print("\nDone.")

if __name__ == "__main__":
    main()
