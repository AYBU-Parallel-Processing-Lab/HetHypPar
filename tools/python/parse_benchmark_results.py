#!/usr/bin/env python3
"""
Parse benchmark results from run_benchmarks.py output directory.

Walks through all .stdout files in data/results/, extracts metrics from
successful runs and error reasons from failed runs, and writes a summary TSV.
"""

import os
import re
import sys
from pathlib import Path

import pandas as pd


RESULTS_DIR = Path("data/results")
OUTPUT_TSV = RESULTS_DIR / "benchmark_summary.tsv"

# Patterns to extract metrics from stdout
METRIC_PATTERNS = {
    "n_iters": r"n_iters\s*:\s*(\d+)",
    "spmv": r"spmv\s*:\s*([\d.]+)",
    "file_read": r"file_read\s*:\s*([\d.]+)",
    "relative_residual": r"relative_residual\s*:\s*([\d.eE+-]+)",
    "everything_total": r"everything_total\s*:\s*([\d.]+)",
}


def parse_metrics(stdout_text):
    """Extract metrics from stdout. Returns dict or None if parsing fails."""
    metrics = {}
    for key, pattern in METRIC_PATTERNS.items():
        match = re.search(pattern, stdout_text)
        if match:
            metrics[key] = match.group(1)
        else:
            return None  # incomplete output
    return metrics


def extract_error_reason(stderr_path):
    """Extract a one-line error reason from stderr file."""
    if not stderr_path.exists():
        return ""
    text = stderr_path.read_text().strip()
    if not text:
        return ""

    # Look for ABORT lines first (our own error messages)
    abort_match = re.search(r"ABORT\s*\[.*?\]:\s*(.+)", text)
    if abort_match:
        return abort_match.group(1).strip()

    # Look for signal-based crashes
    if "signal 11" in text.lower() or "segmentation fault" in text.lower() or "exit 139" in text.lower():
        return "segfault (signal 11)"
    if "signal 9" in text.lower() or "killed" in text.lower():
        return "killed (signal 9 / OOM?)"

    # Look for MPI abort messages
    mpi_match = re.search(r"MPI_ABORT was invoked.*exit code\s*(\d+)", text)
    if mpi_match:
        return f"MPI_ABORT exit code {mpi_match.group(1)}"

    # Generic: take the last non-empty, non-binding line
    for line in reversed(text.splitlines()):
        line = line.strip()
        # Skip MPI binding report lines
        if line.startswith("[") and "bound to" in line:
            continue
        if line.startswith("---"):
            continue
        if line:
            return line[:200]  # truncate long lines

    return "unknown error (see stderr)"


def process_unpartitioned(results_dir):
    """Process cpu.stdout and gpu.stdout files in data/results/<matrix>/."""
    rows = []
    for matrix_dir in sorted(results_dir.iterdir()):
        if not matrix_dir.is_dir():
            continue
        if matrix_dir.name == "logs":
            continue

        matrix = matrix_dir.name

        for solver_type in ("cpu", "gpu"):
            stdout_path = matrix_dir / f"{solver_type}.stdout"
            stderr_path = matrix_dir / f"{solver_type}.stderr"

            if not stdout_path.exists():
                continue

            stdout_text = stdout_path.read_text()
            metrics = parse_metrics(stdout_text)

            row = {
                "matrix": matrix,
                "solver_type": solver_type,
                "imbalance": "",
                "weight": "",
                "seed": "",
            }

            if metrics:
                row.update(metrics)
                row["status"] = "success"
                row["error_reason"] = ""
            else:
                row.update({k: "" for k in METRIC_PATTERNS})
                row["status"] = "failed"
                row["error_reason"] = extract_error_reason(stderr_path)

            rows.append(row)

    return rows


def process_partitioned(logs_dir):
    """Process mpi.stdout and mpi_gpu.stdout under logs/imbalance_X/weight_A_B/matrix_seedS/."""
    rows = []

    if not logs_dir.exists():
        return rows

    for imbalance_dir in sorted(logs_dir.iterdir()):
        if not imbalance_dir.is_dir():
            continue
        # Extract imbalance value: imbalance_3 -> 3
        imb_match = re.match(r"imbalance_(\d+)", imbalance_dir.name)
        if not imb_match:
            continue
        imbalance = imb_match.group(1)

        for weight_dir in sorted(imbalance_dir.iterdir()):
            if not weight_dir.is_dir():
                continue
            # Extract weight: weight_90_10 -> 90_10
            wt_match = re.match(r"weight_(\d+_\d+)", weight_dir.name)
            if not wt_match:
                continue
            weight = wt_match.group(1)

            for run_dir in sorted(weight_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                # Extract matrix and seed: cage10_seed-1 -> cage10, -1
                seed_match = re.match(r"^(.+?)_seed(.+)$", run_dir.name)
                if not seed_match:
                    continue
                matrix = seed_match.group(1)
                seed = seed_match.group(2)

                for solver_type in ("mpi", "mpi_gpu"):
                    stdout_path = run_dir / f"{solver_type}.stdout"
                    stderr_path = run_dir / f"{solver_type}.stderr"

                    if not stdout_path.exists():
                        continue

                    stdout_text = stdout_path.read_text()
                    metrics = parse_metrics(stdout_text)

                    row = {
                        "matrix": matrix,
                        "solver_type": solver_type,
                        "imbalance": imbalance,
                        "weight": weight,
                        "seed": seed,
                    }

                    if metrics:
                        row.update(metrics)
                        row["status"] = "success"
                        row["error_reason"] = ""
                    else:
                        row.update({k: "" for k in METRIC_PATTERNS})
                        row["status"] = "failed"
                        row["error_reason"] = extract_error_reason(stderr_path)

                    rows.append(row)

    return rows


def main():
    if not RESULTS_DIR.exists():
        print(f"Error: results directory not found: {RESULTS_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {RESULTS_DIR} ...")

    rows = []
    rows.extend(process_unpartitioned(RESULTS_DIR))
    rows.extend(process_partitioned(RESULTS_DIR / "logs"))

    df = pd.DataFrame(rows)

    # Reorder columns
    columns = [
        "matrix",
        "solver_type",
        "imbalance",
        "weight",
        "seed",
        "n_iters",
        "spmv",
        "file_read",
        "relative_residual",
        "everything_total",
        "status",
        "error_reason",
    ]
    df = df[columns]

    # Summary
    total = len(df)
    success = (df["status"] == "success").sum()
    failed = (df["status"] == "failed").sum()

    print(f"\nTotal runs:  {total}")
    print(f"  Success:   {success}")
    print(f"  Failed:    {failed}")
    print(f"  Fail rate: {failed / total * 100:.1f}%")

    # Error breakdown
    if failed > 0:
        print(f"\nError reasons (top 10):")
        error_counts = df[df["status"] == "failed"]["error_reason"].value_counts().head(10)
        for reason, count in error_counts.items():
            print(f"  {count:5d}  {reason}")

    # Failed matrices breakdown by solver type
    if failed > 0:
        print(f"\nFailed runs by solver type:")
        for st in ("cpu", "gpu", "mpi", "mpi_gpu"):
            subset = df[(df["solver_type"] == st)]
            f = (subset["status"] == "failed").sum()
            t = len(subset)
            if t > 0:
                print(f"  {st:10s}: {f}/{t} failed ({f/t*100:.1f}%)")

    df.to_csv(OUTPUT_TSV, sep="\t", index=False)
    print(f"\nWritten to {OUTPUT_TSV}")


if __name__ == "__main__":
    main()
