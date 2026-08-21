#!/usr/bin/env python3
"""Paired analysis of the dropout-rate sweep.

Every candidate is evaluated on the SAME folds with the SAME seeds and the
SAME initial weights (the dropout recipe is present at rate = 0 too, so the
layer-seed sequence is identical). The correct comparison is therefore
paired: for each fold, take val_mse(rate) - val_mse(0) and analyse the five
differences.

Decision rule, fixed before looking at the data (same as the other
experiments): a rate beats 0 only if it improves in at least 4 folds out of 5
AND the mean paired difference is larger in magnitude than its own standard
deviation. Otherwise rate* = 0.

The report also shows the train/validation gap per rate: dropout is an
overfitting remedy, so if the gap at rate = 0 is already inside the fold
noise there is nothing for it to fix and rate* = 0 is the expected verdict.

    python3 CNN/experiments/dropout_tuning/analyze.py \
        results/cross_validation/dropout_tuning/sweep_dropout.csv
"""

import csv
import statistics as st
import sys

MIN_IMPROVING_FOLDS = 4
TOTAL_FOLDS = 5


def load(path: str) -> dict:
    by_rate: dict = {}
    with open(path) as handle:
        for row in csv.DictReader(handle):
            value = float(row["dropout_rate"])
            by_rate.setdefault(value, {})[int(row["fold"])] = {
                "val": float(row["val_mse"]),
                "train": float(row["train_mse"]),
                "baseline": float(row["baseline_mse"]),
            }
    return by_rate


def main() -> None:
    path = sys.argv[1]
    data = load(path)
    if 0.0 not in data:
        raise SystemExit("the sweep must include rate = 0 as the reference")
    reference = data[0.0]
    rates = sorted(data)

    print(f"\n=== DROPOUT SWEEP — {path} ===")
    print(f"reference: rate = 0, val MSE per fold "
          f"{[round(reference[f]['val'], 6) for f in sorted(reference)]}\n")

    print(f"{'rate':>6} {'val MSE':>10} {'std':>9} {'ratio/base':>11} "
          f"{'paired diff':>13} {'std':>10} {'better':>7} "
          f"{'train MSE':>10} {'val-train':>10}")
    print("-" * 96)

    rows = []
    for value in rates:
        folds = data[value]
        vals = [folds[f]["val"] for f in sorted(folds)]
        base = [folds[f]["baseline"] for f in sorted(folds)]
        trains = [folds[f]["train"] for f in sorted(folds)]
        diffs = [folds[f]["val"] - reference[f]["val"] for f in sorted(folds)]
        better = sum(1 for d in diffs if d < 0)
        mean_diff = st.mean(diffs)
        std_diff = st.stdev(diffs) if len(diffs) > 1 else 0.0
        gap = st.mean(vals) - st.mean(trains)
        # The rule is applied here, not by eye.
        wins = (value != 0.0 and better >= MIN_IMPROVING_FOLDS
                and mean_diff < 0 and abs(mean_diff) > std_diff)
        rows.append((value, st.mean(vals), std_diff, mean_diff, better, wins))
        print(f"{value:>6.2g} {st.mean(vals):>10.6f} {st.stdev(vals):>9.6f} "
              f"{st.mean(vals)/st.mean(base):>11.5f} {mean_diff:>+13.7f} "
              f"{std_diff:>10.7f} {better:>5}/5 {st.mean(trains):>10.6f} "
              f"{gap:>+10.6f}" + ("   <-- BEATS 0" if wins else ""))

    print("\n--- verdict ---")
    winners = [r for r in rows if r[5]]
    if winners:
        best = min(winners, key=lambda r: r[3])
        print(f"rate* = {best[0]:.2g}: improves in {best[4]}/5 folds, "
              f"paired diff {best[3]:+.7f} +/- {best[2]:.7f}")
    else:
        print(f"rate* = 0. No value satisfies the rule "
              f"(>= {MIN_IMPROVING_FOLDS}/{TOTAL_FOLDS} folds improved and "
              f"|mean paired diff| > its std).")

    non_zero = [r for r in rows if r[0] != 0.0]
    ordered = sorted(non_zero, key=lambda r: r[3])[:3]
    print("\ntop 3 by paired difference (ordering only, see the verdict above):")
    for value, _mean_val, std_diff, mean_diff, better, _wins in ordered:
        print(f"  rate = {value:<6.2g} paired diff {mean_diff:+.7f} "
              f"+/- {std_diff:.7f}  improved in {better}/5 folds")


if __name__ == "__main__":
    main()
