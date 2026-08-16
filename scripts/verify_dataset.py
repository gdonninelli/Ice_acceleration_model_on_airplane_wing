#!/usr/bin/env python3
"""Acceptance checks for a CNN dataset .npz.

Verifies that the file contains airfoil signed distance fields paired with the
matching CFD targets, rather than noise or mismatched columns. Exits non-zero
if any check fails, so it can gate a regeneration.

    python3 scripts/verify_dataset.py dataset/cnn_dataset_train.npz
    python3 scripts/verify_dataset.py dataset/cnn_dataset_train.npz \
        --summaries dataset --paired dataset/cnn_dataset_test.npz

Checks (see results/data_audit_v2/ for the reasoning behind each threshold):
  1. lag-1 spatial autocorrelation > 0.9        - an SDF is 1-Lipschitz, hence smooth
  2. negative-cell fraction in [0.02, 0.20]     - an airfoil covers little of the grid
  3. |grad| coefficient of variation < 0.30     - |grad(SDF)| is nearly constant
  4. field values reject N(0,1) (KS p < 0.01)   - guards against the noise failure
  5. every target present in the summaries      - guards against decoupled columns
  6. both scalar columns have positive variance - guards against dead inputs
  7. no field shared across different angles    - geometry must encode the angle
  8. no train/test leakage                      - only with --paired
"""

import argparse
import glob
import hashlib
import os
import sys

import numpy as np
from scipy import stats

THRESHOLDS = {
    "min_autocorrelation": 0.9,
    "min_negative_fraction": 0.02,
    "max_negative_fraction": 0.20,
    "max_gradient_cv": 0.30,
    "max_normality_p": 0.01,
    "target_match_tolerance": 1e-5,
}


class Checks:
    def __init__(self):
        self.failed = 0
        self.passed = 0

    def report(self, ok, name, measured, expected):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<42} {measured:>22}   expected {expected}")
        if ok:
            self.passed += 1
        else:
            self.failed += 1


def field_stats(X):
    ac_h, ac_v, neg, gcv = [], [], [], []
    for s in X:
        ac_h.append(np.corrcoef(s[:, :-1].ravel(), s[:, 1:].ravel())[0, 1])
        ac_v.append(np.corrcoef(s[:-1, :].ravel(), s[1:, :].ravel())[0, 1])
        neg.append((s < 0).mean())
        gy, gx = np.gradient(s)
        g = np.sqrt(gx ** 2 + gy ** 2)
        # Scale-free: |grad(SDF)| is constant, whatever the grid spacing is, so
        # its coefficient of variation discriminates where its absolute value
        # cannot. Measured 0.131 on real fields against 0.536 on N(0,1) noise
        # (Rayleigh: sqrt(4/pi - 1) = 0.523), so 0.30 separates them with a
        # 2.2x margin above real data and 1.8x below noise.
        gcv.append(g.std() / g.mean() if g.mean() > 0 else np.inf)
    return map(np.array, (ac_h, ac_v, neg, gcv))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--summaries", default="dataset",
                    help="Directory holding summary_*.txt (default: dataset).")
    ap.add_argument("--paired", default=None,
                    help="Companion .npz to check for leakage against.")
    args = ap.parse_args()

    d = np.load(args.npz)
    X = d["X_sdf"].astype(np.float64)
    if X.ndim == 4:
        X = X[:, 0]
    Y = d["Y_cl"].astype(np.float64)
    S = d["X_scalars"].astype(np.float64)
    print(f"\n=== VERIFY {args.npz} ===")
    print(f"  {len(Y)} samples, fields {X.shape[1]}x{X.shape[2]}\n")

    c = Checks()
    ac_h, ac_v, neg, gcv = field_stats(X)

    c.report(ac_h.mean() > THRESHOLDS["min_autocorrelation"],
             "lag-1 autocorrelation (horizontal)", f"{ac_h.mean():.6f}",
             f"> {THRESHOLDS['min_autocorrelation']}")
    c.report(ac_v.mean() > THRESHOLDS["min_autocorrelation"],
             "lag-1 autocorrelation (vertical)", f"{ac_v.mean():.6f}",
             f"> {THRESHOLDS['min_autocorrelation']}")
    c.report(THRESHOLDS["min_negative_fraction"] <= neg.mean() <= THRESHOLDS["max_negative_fraction"],
             "negative-cell fraction", f"{neg.mean():.6f}",
             f"{THRESHOLDS['min_negative_fraction']} - {THRESHOLDS['max_negative_fraction']}")
    c.report(gcv.mean() < THRESHOLDS["max_gradient_cv"],
             "|grad| coefficient of variation", f"{gcv.mean():.6f}",
             f"< {THRESHOLDS['max_gradient_cv']}")

    z = (X[0].ravel() - X[0].mean()) / X[0].std()
    p = stats.kstest(z, "norm").pvalue
    c.report(p < THRESHOLDS["max_normality_p"], "field rejects N(0,1)",
             f"KS p={p:.3g}", f"p < {THRESHOLDS['max_normality_p']}")

    files = sorted(glob.glob(os.path.join(args.summaries, "summary_*.txt")))
    if files:
        cl = np.concatenate([
            np.genfromtxt(f, delimiter=",", skip_header=1, usecols=3) for f in files])
        matched = sum(1 for y in Y
                      if np.any(np.abs(cl - y) < THRESHOLDS["target_match_tolerance"]))
        c.report(matched == len(Y), "targets found in the summaries",
                 f"{matched}/{len(Y)}", "all")
    else:
        print(f"  [SKIP] no summary_*.txt under {args.summaries}")

    for i, name in enumerate(("Reynolds", "AoA")):
        c.report(S[:, i].std() > 0, f"scalar column varies [{name}]",
                 f"std={S[:, i].std():.6g}", "> 0")

    by_hash = {}
    for i in range(len(X)):
        by_hash.setdefault(hashlib.md5(X[i].tobytes()).hexdigest(), []).append(i)
    conflicts = sum(1 for g in by_hash.values()
                    if len(g) > 1 and len(set(S[g, 1])) > 1)
    c.report(conflicts == 0, "no field shared across different angles",
             f"{conflicts} conflicts", "0")

    if args.paired:
        companion = np.load(args.paired)
        other = companion["X_sdf"].astype(np.float64)
        if other.ndim == 4:
            other = other[:, 0]
        other_scalars = companion["X_scalars"].astype(np.float64)

        def sample_key(field, scalars):
            return hashlib.md5(field.tobytes() + scalars.tobytes()).hexdigest()

        def geometry_key(field):
            return hashlib.md5(field.tobytes()).hexdigest()

        mine = {sample_key(X[i], S[i]) for i in range(len(X))}
        theirs = {sample_key(other[i], other_scalars[i]) for i in range(len(other))}
        c.report(not (mine & theirs), "no duplicated sample in the companion set",
                 f"{len(mine & theirs)} shared", "0")

        # Reported, not failed: one geometry legitimately serves several
        # Reynolds numbers, so a uniformly random split necessarily puts the
        # same field on both sides. It is not a pipeline defect, but it does
        # mean the test set measures generalization to unseen (geometry,
        # Reynolds) pairs, not to unseen geometries. A group-wise split on
        # (profile, angle) would be needed for the stronger claim.
        my_geometry = {geometry_key(X[i]) for i in range(len(X))}
        their_geometry = {geometry_key(other[i]) for i in range(len(other))}
        overlap = len(my_geometry & their_geometry)
        print(f"  [INFO] geometries shared with the companion set        "
              f"{overlap}/{len(their_geometry)} of its fields")
        print(f"         expected: one geometry serves several Reynolds, so a random")
        print(f"         split shares fields. Test measures unseen (geometry, Re) pairs,")
        print(f"         not unseen geometries.")

    print(f"\n  {c.passed} passed, {c.failed} failed\n")
    return 1 if c.failed else 0


if __name__ == "__main__":
    sys.exit(main())
