#!/usr/bin/env python3
"""Reproducible topology-tuning figures from committed raw artifacts.

Reads only:
  results/cross_validation/layer_tuning/summary.csv
  results/cross_validation/layer_tuning/aggregated.csv
  results/cross_validation/layer_tuning/run_leonardo_51807287.log

Writes:
  Report/Images/Chapter04/topology/topology_ranking.png
  Report/Images/Chapter04/topology/topology_kernel_paired.png
  Report/Images/Chapter04/topology/topology_history.png

The log histories are public 10-epoch checkpoints (epoch 10..100) printed by
CNN/experiments/layer_tuning/main.cpp via FoldMetrics::history. No per-epoch
diagnostics (gradient norms, activation statistics, update ratios) exist for
this experiment, so none are plotted.
"""

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[4]
RESULTS = REPO / "results" / "cross_validation" / "layer_tuning"
OUTDIR = REPO / "Report" / "Images" / "Chapter04" / "topology"

SUMMARY = RESULTS / "summary.csv"
AGGREGATED = RESULTS / "aggregated.csv"
LOG = RESULTS / "run_leonardo_51807287.log"


def load_summary():
    rows = []
    with SUMMARY.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(
                {
                    "candidate": row["candidate"],
                    "mean": float(row["mean_val_mse"]),
                    "std": float(row["std_val_mse"]),
                }
            )
    return rows


def load_aggregated():
    per_candidate = {}
    per_candidate_weighted = {}
    with AGGREGATED.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            cand = row["candidate"]
            val = float(row["val_mse"])
            nval = int(row["val_samples"])
            per_candidate.setdefault(cand, []).append(val)
            if cand not in per_candidate_weighted:
                per_candidate_weighted[cand] = [0.0, 0]
            per_candidate_weighted[cand][0] += val * nval
            per_candidate_weighted[cand][1] += nval
    weighted_means = {
        cand: num / den for cand, (num, den) in per_candidate_weighted.items()
    }
    return per_candidate, weighted_means


def fold_weights():
    """Validation sample counts per zero-based fold from aggregated.csv."""
    weights = {}
    with AGGREGATED.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            weights[int(row["fold"])] = int(row["val_samples"])
    return weights


def parse_log_histories():
    """Return {candidate: {fold_idx: [(epoch, train_obj, val_mse)]}}.

    Candidate identity comes from log order lines
    'Candidate: ... [layer-architecture=<name>]' followed by five
    'Fold N (...)' blocks each with ten 'epoch=' lines.
    """
    text = LOG.read_text(encoding="utf-8")
    histories = {}
    current = None
    current_fold = None
    cand_re = re.compile(r"\[layer-architecture=([^\]]+)\]")
    fold_re = re.compile(r"^\s*Fold\s+(\d+)")
    epoch_re = re.compile(
        r"epoch=(\d+)\s+train_objective=([0-9.eE+-]+)\s+validation_mse=([0-9.eE+-]+)"
    )
    for line in text.splitlines():
        m = cand_re.search(line)
        if m and line.strip().startswith("Candidate:"):
            current = m.group(1)
            histories[current] = {}
            current_fold = None
            continue
        m = fold_re.match(line)
        if m and current is not None:
            current_fold = int(m.group(1)) - 1
            histories[current][current_fold] = []
            continue
        m = epoch_re.search(line)
        if m and current is not None and current_fold is not None:
            histories[current][current_fold].append(
                (int(m.group(1)), float(m.group(2)), float(m.group(3)))
            )
    return histories


def fold_mean(histories, candidate):
    """Sample-weighted fold-averaged validation MSE per checkpoint epoch."""
    weights = fold_weights()
    folds = histories[candidate]
    epochs = [e for e, _, _ in folds[0]]
    means = []
    for j in range(len(epochs)):
        num = sum(folds[f][j][2] * weights[f] for f in sorted(folds))
        den = sum(weights[f] for f in sorted(folds))
        means.append(num / den)
    return epochs, means


def plot_ranking(summary):
    ordered = list(summary)
    labels = [r["candidate"] for r in ordered]
    means = [r["mean"] for r in ordered]
    stds = [r["std"] for r in ordered]
    short = [l.replace("conv5x5-dense-", "5x5:").replace("conv3x3-dense-", "3x3:") for l in labels]

    fig, ax = plt.subplots(figsize=(12, 5.2))
    x = list(range(len(ordered)))
    colors = []
    for l in labels:
        if l == "conv5x5-dense-1024-512-256-128":
            colors.append("tab:blue")
        elif l == "conv5x5-dense-128-64":
            colors.append("tab:orange")
        elif l.startswith("conv3x3"):
            colors.append("tab:red")
        elif l in ("conv5x5-dense-1536-768", "conv5x5-dense-2048-1024", "conv5x5-dense-64-32"):
            colors.append("tab:gray")
        else:
            colors.append("tab:green")
    ax.errorbar(x, means, yerr=stds, fmt="o", ecolor="black",
                capsize=3, markersize=5, markerfacecolor="none",
                markeredgecolor="black", elinewidth=1)
    for xi, yi, c in zip(x, means, colors):
        ax.plot(xi, yi, "o", color=c, markersize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean validation MSE (physical units)")
    ax.set_xlabel("Candidate in ranked order (lowest mean on the left)")
    ax.set_title("Layer-tuning ranking: mean validation MSE over five folds")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out = OUTDIR / "topology_ranking.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_kernel_paired(per_candidate, weighted_means):
    pairs = ["128-64", "256-128", "512-256"]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = list(range(len(pairs)))
    width = 0.36
    for i, suffix in enumerate(pairs):
        c5 = f"conv5x5-dense-{suffix}"
        c3 = f"conv3x3-dense-{suffix}"
        v5 = per_candidate[c5]
        v3 = per_candidate[c3]
        m5 = weighted_means[c5]
        m3 = weighted_means[c3]
        ax.bar(i - width / 2, m5, width, label="5x5" if i == 0 else "", color="tab:blue")
        ax.bar(i + width / 2, m3, width, label="3x3" if i == 0 else "", color="tab:red")
        for v in v5:
            ax.plot(i - width / 2, v, "o", color="black", markersize=4)
        for v in v3:
            ax.plot(i + width / 2, v, "o", color="black", markersize=4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"head {s}" for s in pairs])
    ax.set_ylabel("Validation MSE (physical units)")
    ax.set_title("Paired kernel comparison at matched dense heads (dots are folds)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out = OUTDIR / "topology_kernel_paired.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    for suffix in pairs:
        c5 = f"conv5x5-dense-{suffix}"
        c3 = f"conv3x3-dense-{suffix}"
        m5 = weighted_means[c5]
        m3 = weighted_means[c3]
        print(f"{suffix}: 5x5 mean={m5:.8f} 3x3 mean={m3:.8f} diff={m3-m5:.8f}")


def plot_history(histories):
    curves = [
        ("conv5x5-dense-1024-512-256-128", "winner 1024-512-256-128 (depth 4)", "tab:blue"),
        ("conv5x5-dense-1024-512", "same entry width, depth 2", "tab:green"),
        ("conv5x5-dense-128-64", "baseline 128-64", "tab:orange"),
        ("conv3x3-dense-512-256", "3x3 mate at 512-256", "tab:red"),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for cand, label, color in curves:
        epochs, means = fold_mean(histories, cand)
        assert epochs == [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], epochs
        ax.plot(epochs, means, "-o", label=label, color=color, markersize=4)
    ax.set_xlabel("Epoch (public 10-epoch checkpoints)")
    ax.set_ylabel("Fold-averaged validation MSE (physical units)")
    ax.set_title("Validation convergence for selected topology candidates")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = OUTDIR / "topology_history.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    summary = load_summary()
    assert len(summary) == 18, f"expected 18 candidates, got {len(summary)}"
    per_candidate, weighted_means = load_aggregated()
    assert len(per_candidate) == 18
    histories = parse_log_histories()
    assert len(histories) == 18, f"expected 18 log candidates, got {len(histories)}"
    for cand, folds in histories.items():
        assert len(folds) == 5, cand
        for f, pts in folds.items():
            assert len(pts) == 10, (cand, f, len(pts))
    plot_ranking(summary)
    plot_kernel_paired(per_candidate, weighted_means)
    plot_history(histories)


if __name__ == "__main__":
    main()
