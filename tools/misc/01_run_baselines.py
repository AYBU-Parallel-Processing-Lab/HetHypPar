#!/usr/bin/env python3
"""Phase 1: Run CPU and GPU baselines on cage13 at 200 iterations.

Outputs JSON results to tools/misc/results/baselines.json
"""

import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"
MATRIX = "cage13"
ITERATIONS = 200

METRIC_PATTERNS = {
    "n_iters": r"n_iters\s*:\s*(\d+)",
    "spmv": r"spmv\s*:\s*([\d.]+)",
    "file_read": r"file_read\s*:\s*([\d.]+)",
    "relative_residual": r"relative_residual\s*:\s*([\d.E+\-]+)",
    "everything_total": r"everything_total\s*:\s*([\d.]+)",
}


def parse_metrics(stdout: str) -> dict:
    metrics = {}
    for key, pattern in METRIC_PATTERNS.items():
        m = re.search(pattern, stdout)
        if m:
            val = m.group(1)
            metrics[key] = int(val) if key == "n_iters" else float(val)
    return metrics


def run_solver(binary_name: str, extra_args: list[str] = None) -> dict:
    binary = PROJECT_ROOT / "build" / binary_name
    if not binary.exists():
        return {"error": f"Binary not found: {binary}"}

    mat_dir = PROJECT_ROOT / "data" / "matrices" / MATRIX
    cmd = [
        str(binary),
        "-m", f"/matrices/{MATRIX}.mtx",
        "-n", str(ITERATIONS),
        "-o", str(mat_dir / "out" / f"X_{binary_name.replace('bicgstab-', '')}.txt"),
        "-x", str(mat_dir / "in" / "X_init.txt"),
        "-y", str(mat_dir / "in" / "B.txt"),
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"[run] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=PROJECT_ROOT)

    return {
        "command": " ".join(cmd),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "metrics": parse_metrics(result.stdout),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    print(f"=== Phase 1: Baselines ({MATRIX}, {ITERATIONS} iterations) ===\n")

    # CPU baseline
    print("--- CPU ---")
    results["cpu"] = run_solver("bicgstab-cpu")
    if results["cpu"].get("metrics"):
        print(f"  spmv: {results['cpu']['metrics'].get('spmv', 'N/A')}s")
        print(f"  total: {results['cpu']['metrics'].get('everything_total', 'N/A')}s")
    else:
        print(f"  ERROR: {results['cpu'].get('error', results['cpu'].get('stderr', ''))}")

    print()

    # GPU baseline
    print("--- GPU ---")
    results["gpu"] = run_solver("bicgstab-gpu")
    if results["gpu"].get("metrics"):
        print(f"  spmv: {results['gpu']['metrics'].get('spmv', 'N/A')}s")
        print(f"  total: {results['gpu']['metrics'].get('everything_total', 'N/A')}s")
    else:
        print(f"  ERROR: {results['gpu'].get('error', results['gpu'].get('stderr', ''))}")

    # Save results
    out_path = RESULTS_DIR / "baselines.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
