#!/usr/bin/env python3
"""Paired analysis of a regularization sweep.

The fold-to-fold spread (~0.0014) is larger than any effect the penalty is
expected to produce, so comparing `mean +/- std` between candidates cannot
conclude anything. Every candidate is evaluated on the SAME folds with the
SAME seeds, so the correct comparison is paired: for each fold, take
val_mse(lambda) - val_mse(0) and analyse the five differences.

Decision rule, fixed before looking at the data: a lambda beats 0 only if it
improves in at least 4 folds out of 5 AND the mean paired difference is larger
in magnitude than its own standard deviation. Otherwise lambda* = 0.

    python3 CNN/experiments/regularization_tuning/analyze.py \
        results/cross_validation/regularization_tuning/sweep_l2.csv l2
"""

import csv
import statistics as st
import sys

MIN_IMPROVING_FOLDS = 4
TOTAL_FOLDS = 5


def load(path, axis):
    key = "l2_weight" if axis == "l2" else "l1_weight"
    by_lambda = {}
    for row in csv.DictReader(open(path)):
        value = float(row[key])
        by_lambda.setdefault(value, {})[int(row["fold"])] = {
            "val": float(row["val_mse"]),
            "train": float(row["train_mse"]),
            "baseline": float(row["baseline_mse"]),
            "penalty": float(row[f"{axis}_penalty"]),
            "sum_w2": float(row["sum_w2"]),
            "change": float(row["weight_change_norm"]),
        }
    return by_lambda


def main():
    path, axis = sys.argv[1], sys.argv[2]
    data = load(path, axis)
    if 0.0 not in data:
        raise SystemExit("the sweep must include lambda = 0 as the reference")
    reference = data[0.0]
    lambdas = sorted(data)

    print(f"\n=== SWEEP {axis.upper()} — {path} ===")
    print(f"reference: lambda = 0, val MSE per fold "
          f"{[round(reference[f]['val'], 6) for f in sorted(reference)]}\n")

    print(f"{'lambda':>10} {'val MSE':>10} {'std':>9} {'ratio/base':>11} "
          f"{'paired diff':>13} {'std':>10} {'better':>7} {'penalty':>10} "
          f"{'sum_w2':>10} {'|dw|':>8}")
    print("-" * 116)

    rows = []
    for value in lambdas:
        folds = data[value]
        vals = [folds[f]["val"] for f in sorted(folds)]
        base = [folds[f]["baseline"] for f in sorted(folds)]
        diffs = [folds[f]["val"] - reference[f]["val"] for f in sorted(folds)]
        better = sum(1 for d in diffs if d < 0)
        mean_diff = st.mean(diffs)
        std_diff = st.stdev(diffs) if len(diffs) > 1 else 0.0
        penalty = st.mean(folds[f]["penalty"] for f in sorted(folds))
        sum_w2 = st.mean(folds[f]["sum_w2"] for f in sorted(folds))
        change = st.mean(folds[f]["change"] for f in sorted(folds))
        # The rule is applied here, not by eye.
        wins = (value != 0.0 and better >= MIN_IMPROVING_FOLDS
                and mean_diff < 0 and abs(mean_diff) > std_diff)
        rows.append((value, st.mean(vals), std_diff, mean_diff, better, wins,
                     penalty, sum_w2, change))
        print(f"{value:>10.3g} {st.mean(vals):>10.6f} {st.stdev(vals):>9.6f} "
              f"{st.mean(vals)/st.mean(base):>11.5f} {mean_diff:>+13.7f} "
              f"{std_diff:>10.7f} {better:>5}/5 {penalty:>10.4g} "
              f"{sum_w2:>10.2f} {change:>8.4f}"
              + ("   <-- BEATS 0" if wins else ""))

    print("\n--- verdict ---")
    winners = [r for r in rows if r[5]]
    if winners:
        best = min(winners, key=lambda r: r[3])
        print(f"lambda* = {best[0]:.3g}: improves in {best[4]}/5 folds, "
              f"paired diff {best[3]:+.7f} +/- {best[2]:.7f}")
    else:
        print(f"lambda* = 0. No value satisfies the rule "
              f"(>= {MIN_IMPROVING_FOLDS}/{TOTAL_FOLDS} folds improved and "
              f"|mean paired diff| > its std).")

    non_zero = [r for r in rows if r[0] != 0.0]
    ordered = sorted(non_zero, key=lambda r: r[3])[:3]
    print("\ntop 3 by paired difference (ordering only, see the verdict above):")
    for value, mean_val, std_diff, mean_diff, better, wins, *_ in ordered:
        print(f"  lambda = {value:<10.3g} paired diff {mean_diff:+.7f} "
              f"+/- {std_diff:.7f}  improved in {better}/5 folds")

    # Did lambda do anything at all? If the weights are identical across the
    # grid, the sweep measured the same run several times.
    w2 = [r[7] for r in rows]
    print(f"\nsum(w^2) across the grid: {min(w2):.2f} .. {max(w2):.2f} "
          f"(spread {100*(max(w2)-min(w2))/max(w2):.1f}%)")
    if (max(w2) - min(w2)) / max(w2) < 1e-4:
        print("  WARNING: the grid did not move the weights; the range is too low.")
    else:
        print("  the grid moved the weights, so the range is in the active region.")


if __name__ == "__main__":
    main()
