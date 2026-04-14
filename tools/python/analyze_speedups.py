#!/usr/bin/env python3
"""
Analyze SpMV speedups from benchmark_summary.tsv.

Baseline: cpu solver's spmv time per matrix.
Speedup = cpu_spmv / solver_spmv for each successful run.

For gpu: one speedup per matrix.
For mpi/mpi_gpu: one speedup per (matrix, imbalance, weight, seed) combo.

Usage:
    micromamba run -n octave python tools/python/analyze_speedups.py \
        --input data/results/benchmark_summary.tsv \
        --output data/results/speedup_analysis.tsv
"""

import argparse
import csv
import sys


def main():
    parser = argparse.ArgumentParser(description="Compute SpMV speedups relative to CPU baseline")
    parser.add_argument("--input", default="data/results/benchmark_summary.tsv",
                        help="Path to benchmark_summary.tsv")
    parser.add_argument("--output", default="data/results/speedup_analysis.tsv",
                        help="Output TSV path")
    args = parser.parse_args()

    # --- Parse input ---
    rows = []
    with open(args.input, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)

    # --- Build CPU baseline lookup: matrix -> spmv ---
    cpu_baseline = {}
    for row in rows:
        if row["solver_type"] == "cpu" and row["status"] == "success":
            matrix = row["matrix"]
            spmv = float(row["spmv"])
            cpu_baseline[matrix] = spmv

    # --- Compute speedups for all non-cpu successful runs ---
    results = []
    skipped_no_baseline = 0
    skipped_failed = 0

    for row in rows:
        if row["solver_type"] == "cpu":
            continue
        if row["status"] != "success":
            skipped_failed += 1
            continue

        matrix = row["matrix"]
        if matrix not in cpu_baseline:
            skipped_no_baseline += 1
            continue

        cpu_spmv = cpu_baseline[matrix]
        solver_spmv = float(row["spmv"])

        if solver_spmv <= 0:
            continue

        speedup = cpu_spmv / solver_spmv

        results.append({
            "matrix": matrix,
            "solver_type": row["solver_type"],
            "imbalance": row["imbalance"],
            "weight": row["weight"],
            "seed": row["seed"],
            "cpu_spmv": cpu_spmv,
            "solver_spmv": solver_spmv,
            "speedup": speedup,
        })

    # --- Sort: by solver_type, then matrix, then speedup descending ---
    results.sort(key=lambda r: (r["solver_type"], r["matrix"], -r["speedup"]))

    # --- Write output ---
    fieldnames = ["matrix", "solver_type", "imbalance", "weight", "seed",
                  "cpu_spmv", "solver_spmv", "speedup"]

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(results)

    # --- Print summary to stdout ---
    print(f"Input rows:              {len(rows)}")
    print(f"Skipped (failed):        {skipped_failed}")
    print(f"Skipped (no CPU baseline): {skipped_no_baseline}")
    print(f"Speedup rows written:    {len(results)}")
    print(f"Output: {args.output}")

    # --- Per-solver summary stats ---
    from collections import defaultdict
    by_solver = defaultdict(list)
    for r in results:
        by_solver[r["solver_type"]].append(r["speedup"])

    print("\n--- Speedup summary (spmv) ---")
    print(f"{'solver':<10} {'count':>6} {'min':>8} {'median':>8} {'mean':>8} {'max':>8}")
    for solver in ["gpu", "mpi", "mpi_gpu"]:
        vals = sorted(by_solver.get(solver, []))
        if not vals:
            continue
        n = len(vals)
        mn = vals[0]
        mx = vals[-1]
        mean = sum(vals) / n
        median = vals[n // 2] if n % 2 == 1 else (vals[n // 2 - 1] + vals[n // 2]) / 2
        print(f"{solver:<10} {n:>6} {mn:>8.2f} {median:>8.2f} {mean:>8.2f} {mx:>8.2f}")


if __name__ == "__main__":
    main()
