#!/usr/bin/env python3
"""Paired analysis of the physics-weight (SIMM lambda) sweep.

Every candidate is evaluated on the SAME folds with the SAME seeds, so the
correct comparison is paired: for each fold, take val_mse(lambda) - val_mse(0)
and analyse the five differences. lambda = 0 is the pure data-driven
reference, so the paired difference directly measures the value of the
physical prior C_l = 2*pi*alpha.

Decision rule, fixed before looking at the data (same as the activation and
regularization experiments): a lambda beats the reference only if it improves
in at least 4 folds out of 5 AND the mean paired difference is larger in
magnitude than its own standard deviation. Otherwise lambda* = 0.

The report also splits the validation MSE into the small-angle population
(|alpha| <= 10 deg, where the prior is active) and the past-stall population
(|alpha| > 10 deg), because the physics term should only act inside the mask.

    python3 CNN/experiments/physics_weight_tuning/analyze.py \
        results/cross_validation/physics_weight_tuning/sweep_physics.csv
"""

import csv
import statistics as st
import sys

MIN_IMPROVING_FOLDS = 4
TOTAL_FOLDS = 5
PRODUCTION_DEFAULT = 0.25


def load(path: str) -> dict:
    by_weight: dict = {}
    with open(path) as handle:
        for row in csv.DictReader(handle):
            value = float(row["physics_weight"])
            by_weight.setdefault(value, {})[int(row["fold"])] = {
                "val": float(row["val_mse"]),
                "train": float(row["train_mse"]),
                "baseline": float(row["baseline_mse"]),
                "small": float(row["val_mse_small_angle"]),
                "large": float(row["val_mse_large_angle"]),
            }
    return by_weight


def main() -> None:
    path = sys.argv[1]
    data = load(path)
    if 0.0 not in data:
        raise SystemExit("the sweep must include lambda = 0 as the reference")
    reference = data[0.0]
    weights = sorted(data)

    print(f"\n=== PHYSICS-WEIGHT SWEEP — {path} ===")
    print(f"reference: lambda = 0 (pure data-driven), val MSE per fold "
          f"{[round(reference[f]['val'], 6) for f in sorted(reference)]}\n")

    print(f"{'lambda':>8} {'val MSE':>10} {'std':>9} {'ratio/base':>11} "
          f"{'paired diff':>13} {'std':>10} {'better':>7} "
          f"{'small-ang':>10} {'large-ang':>10} {'train':>9}")
    print("-" * 108)

    rows = []
    for value in weights:
        folds = data[value]
        vals = [folds[f]["val"] for f in sorted(folds)]
        base = [folds[f]["baseline"] for f in sorted(folds)]
        diffs = [folds[f]["val"] - reference[f]["val"] for f in sorted(folds)]
        better = sum(1 for d in diffs if d < 0)
        mean_diff = st.mean(diffs)
        std_diff = st.stdev(diffs) if len(diffs) > 1 else 0.0
        small = st.mean(folds[f]["small"] for f in sorted(folds))
        large = st.mean(folds[f]["large"] for f in sorted(folds))
        train = st.mean(folds[f]["train"] for f in sorted(folds))
        # The rule is applied here, not by eye.
        wins = (value != 0.0 and better >= MIN_IMPROVING_FOLDS
                and mean_diff < 0 and abs(mean_diff) > std_diff)
        rows.append((value, st.mean(vals), std_diff, mean_diff, better, wins))
        marker = "   <-- BEATS 0" if wins else ""
        if value == PRODUCTION_DEFAULT:
            marker += "   (production default)"
        print(f"{value:>8.3g} {st.mean(vals):>10.6f} {st.stdev(vals):>9.6f} "
              f"{st.mean(vals)/st.mean(base):>11.5f} {mean_diff:>+13.7f} "
              f"{std_diff:>10.7f} {better:>5}/5 {small:>10.6f} "
              f"{large:>10.6f} {train:>9.6f}" + marker)

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
    for value, _mean_val, std_diff, mean_diff, better, _wins in ordered:
        print(f"  lambda = {value:<8.3g} paired diff {mean_diff:+.7f} "
              f"+/- {std_diff:.7f}  improved in {better}/5 folds")


if __name__ == "__main__":
    main()
