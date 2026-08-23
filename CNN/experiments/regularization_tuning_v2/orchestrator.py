#!/usr/bin/env python3
"""Segmented driver for the L1/L2 regularization sweep (layer_tuning architecture, lr=1e-5).

One MPI invocation per (axis, lambda) candidate, so an interrupted sweep
loses at most the candidate that was running. Completed candidates are
recorded in a state file and skipped on the next run; a candidate counts as
complete only when its CSV exists and holds one row per fold. Mirrors
CNN/experiments/physics_weight_tuning_lr1e3/orchestrator.py.

The lambda=0 candidate is identical for both axes (l1=0, l2=0 is the same
TrialConfig), so it is run once and its CSV is copied to serve as both
l1_0.csv and l2_0.csv, instead of spending a second ~76-minute run on a
bit-identical result.

Usage:
    python3 CNN/experiments/regularization_tuning_v2/orchestrator.py [options]

Run with --help for the option list.
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time

L2_GRID = [0.0, 1e-4, 3.16e-4, 1e-3, 3.16e-3, 1e-2, 3.16e-2, 1e-1]
L1_GRID = [0.0, 6.75e-7, 2.13e-6, 6.75e-6, 2.13e-5, 6.75e-5, 2.13e-4, 6.75e-4]
DEFAULT_RESULTS_DIR = "results/cross_validation/regularization_tuning_v2"
# OpenMPI is not on PATH by default on every machine in the group.
OPENMPI_BIN = "/usr/lib64/openmpi/bin"


def format_lambda(value):
    # Matches format_lambda() in main.cpp: scientific, 0 decimal digits,
    # plain "0" for the reference candidate (e.g. 6.75e-4 -> "7e-04").
    if value == 0.0:
        return "0"
    return f"{value:.0e}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the L1/L2 regularization grids one candidate at a time.")
    parser.add_argument("--binary", default="build/CNN/experiments/regularization_tuning_v2",
                        help="Path to the executable.")
    parser.add_argument("--ranks", type=int, default=4,
                        help="MPI ranks per invocation (default: 4, matching "
                             "activation_tuning's measured optimum).")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Epochs per fold (default: 100).")
    parser.add_argument("--folds", type=int, default=5,
                        help="Number of CV folds (default: 5).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--diagnostics", action="store_true", default=True,
                        help="Write per-epoch diagnostics (default: on).")
    parser.add_argument("--no-diagnostics", dest="diagnostics",
                        action="store_false")
    parser.add_argument("--axis", choices=["l1", "l2"], default=None,
                        help="Restrict to one axis (default: both).")
    parser.add_argument("--only", nargs="*", type=float, default=None,
                        help="Restrict to these lambda values (within --axis, "
                             "or applied to both axes if --axis is omitted).")
    parser.add_argument("--force", action="store_true",
                        help="Re-run candidates already marked complete.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the commands without running them.")
    return parser.parse_args()


def state_path(results_dir):
    return os.path.join(results_dir, "orchestrator_state.json")


def read_state(results_dir):
    path = state_path(results_dir)
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        print(f"WARNING: {path} is unreadable, starting from an empty state.")
        return {}


def write_state(results_dir, state):
    os.makedirs(results_dir, exist_ok=True)
    with open(state_path(results_dir), "w") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)


def candidate_csv(results_dir, axis, label):
    return os.path.join(results_dir, f"{axis}_{label}.csv")


def csv_is_complete(path, folds):
    """A candidate is complete only if its CSV holds one row per fold."""
    if not os.path.exists(path):
        return False
    try:
        with open(path, newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        return False
    if len(rows) != folds:
        return False
    seen = {row.get("fold") for row in rows}
    return seen == {str(index) for index in range(folds)}


def build_command(args, axis, value):
    command = [
        "mpirun", "-n", str(args.ranks), "--oversubscribe", args.binary,
        "--mode", "cv",
        "--axis", axis,
        "--lambda", str(value),
        "--epochs", str(args.epochs),
        "--folds", str(args.folds),
        "--seed", str(args.seed),
        "--results-dir", args.results_dir,
    ]
    command.append("--diagnostics" if args.diagnostics else "--no-diagnostics")
    return command


def copy_zero_candidate(results_dir, from_axis, to_axis, folds):
    """Reuse the lambda=0 CSV (bit-identical run) across axes instead of
    re-running it. Rewrites the candidate/axis-coded columns are NOT
    touched: the row content (l1_weight=0, l2_weight=0 either way) is
    already axis-agnostic, only the file name differs."""
    src = candidate_csv(results_dir, from_axis, "0")
    dst = candidate_csv(results_dir, to_axis, "0")
    if csv_is_complete(dst, folds):
        return True
    if not csv_is_complete(src, folds):
        return False
    shutil.copyfile(src, dst)
    src_history = src.replace(".csv", "_history.csv")
    dst_history = dst.replace(".csv", "_history.csv")
    if os.path.exists(src_history):
        shutil.copyfile(src_history, dst_history)
    print(f"[{to_axis}_0] reused from [{from_axis}_0] (identical config, "
          "not re-run).")
    return True


def aggregate(results_dir, axis, labels, folds):
    """Concatenate one axis's per-lambda CSVs into sweep_<axis>_v2.csv,
    with the same schema as regularization_tuning's sweep_l1.csv/sweep_l2.csv,
    so analyze.py runs against it unmodified."""
    rows = []
    header = None
    for label in labels:
        path = candidate_csv(results_dir, axis, label)
        if not csv_is_complete(path, folds):
            continue
        with open(path, newline="") as handle:
            reader = csv.reader(handle)
            candidate_header = next(reader)
            header = header or candidate_header
            rows.extend(reader)
    if not rows:
        return None
    out_path = os.path.join(results_dir, f"sweep_{axis}_v2.csv")
    with open(out_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return out_path


def main():
    args = parse_args()
    axes = [args.axis] if args.axis else ["l1", "l2"]
    grids = {"l1": L1_GRID, "l2": L2_GRID}
    if args.only:
        unknown = {axis: [v for v in args.only if v not in grids[axis]]
                  for axis in axes}
        bad = {axis: vals for axis, vals in unknown.items() if vals}
        if bad:
            sys.exit(f"Unknown lambda value(s) for the given axis/grid: {bad}")

    if not os.path.exists(args.binary) and not args.dry_run:
        sys.exit(f"Executable not found: {args.binary}\n"
                 "Build it first (see the experiment README).")

    os.makedirs(args.results_dir, exist_ok=True)
    state = read_state(args.results_dir)

    env = os.environ.copy()
    if os.path.isdir(OPENMPI_BIN):
        env["PATH"] = OPENMPI_BIN + os.pathsep + env.get("PATH", "")

    print("===== Regularization tuning orchestrator =====")
    print(f"  axes       : {axes}")
    print(f"  ranks      : {args.ranks}")
    print(f"  epochs     : {args.epochs}   folds: {args.folds}")
    print(f"  diagnostics: {'on' if args.diagnostics else 'off'}")
    print(f"  results    : {args.results_dir}")
    print()

    failures = []
    zero_done_by = None  # first axis whose lambda=0 candidate completed
    for axis in axes:
        values = args.only if args.only else grids[axis]
        labels = [format_lambda(v) for v in values]
        for value, label in zip(values, labels):
            key = f"{axis}_{label}"
            path = candidate_csv(args.results_dir, axis, label)

            if value == 0.0 and zero_done_by is not None and zero_done_by != axis:
                if copy_zero_candidate(args.results_dir, zero_done_by, axis,
                                       args.folds):
                    state.setdefault(key, {})["status"] = "complete (reused)"
                    write_state(args.results_dir, state)
                    continue

            if not args.force and csv_is_complete(path, args.folds):
                print(f"[{key}] already complete, skipping.")
                state.setdefault(key, {})["status"] = "complete"
                if value == 0.0 and zero_done_by is None:
                    zero_done_by = axis
                continue

            command = build_command(args, axis, value)
            if args.dry_run:
                print(f"[{key}] {' '.join(command)}")
                continue

            print(f"[{key}] starting ({time.strftime('%H:%M:%S')})")
            started = time.time()
            result = subprocess.run(command, env=env)
            elapsed = time.time() - started

            if result.returncode != 0 and not csv_is_complete(path, args.folds):
                print(f"[{key}] FAILED with exit code {result.returncode} "
                      f"after {elapsed:.0f}s")
                state[key] = {"status": "failed",
                             "exit_code": result.returncode,
                             "seconds": round(elapsed, 1)}
                write_state(args.results_dir, state)
                failures.append(key)
                continue

            if not csv_is_complete(path, args.folds):
                print(f"[{key}] FAILED: exited cleanly but {path} is missing "
                      "or incomplete.")
                state[key] = {"status": "incomplete", "seconds": round(elapsed, 1)}
                write_state(args.results_dir, state)
                failures.append(key)
                continue

            print(f"[{key}] done in {elapsed / 60:.1f} min")
            state[key] = {"status": "complete", "seconds": round(elapsed, 1),
                         "epochs": args.epochs, "folds": args.folds}
            write_state(args.results_dir, state)
            if value == 0.0 and zero_done_by is None:
                zero_done_by = axis

    if args.dry_run:
        return 0

    print("\n===== Orchestrator finished =====")
    out_paths = []
    for axis in ["l1", "l2"]:
        labels = [format_lambda(v) for v in grids[axis]]
        complete = [l for l in labels
                   if csv_is_complete(candidate_csv(args.results_dir, axis, l),
                                      args.folds)]
        missing = [l for l in labels if l not in complete]
        print(f"{axis}: complete {len(complete)}/{len(labels)}"
              + (f"  missing: {missing}" if missing else ""))
        if not missing:
            out_path = aggregate(args.results_dir, axis, labels, args.folds)
            if out_path:
                out_paths.append(out_path)
                print(f"  aggregated CSV: {out_path}")

    if len(out_paths) == 2:
        print("\nNext step: analyse each grid with\n"
              f"  python3 CNN/experiments/regularization_tuning/analyze.py "
              f"{out_paths[0]} l1\n"
              f"  python3 CNN/experiments/regularization_tuning/analyze.py "
              f"{out_paths[1]} l2")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
