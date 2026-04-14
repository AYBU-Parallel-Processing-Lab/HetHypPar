#!/usr/bin/env python3
"""Phase 2: MPI-only sweep on P-cores for cage13 at 200 iterations.

Tests rank counts 2, 4, 6, 8 with uniform weights, pinned to P-cores via rankfiles.
Uses quick_mpi_test.run() for partitioning and execution.
Outputs JSON results to tools/misc/results/mpi_sweep.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
from quick_mpi_test import run as mpi_run

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"
MATRIX = "cage13"
ITERATIONS = 200

CONFIGS = [
    {"nranks": 2, "weight": "data/weights/experiment-pcore/equal_2.txt", "rankfile": "data/rankfile/experiment/2pcore.rankfile"},
    {"nranks": 4, "weight": "data/weights/experiment-pcore/equal_4.txt", "rankfile": "data/rankfile/experiment/4pcore.rankfile"},
    {"nranks": 6, "weight": "data/weights/experiment-pcore/equal_6.txt", "rankfile": "data/rankfile/experiment/6pcore.rankfile"},
    {"nranks": 8, "weight": "data/weights/experiment-pcore/equal_8.txt", "rankfile": "data/rankfile/experiment/8pcore.rankfile"},
]


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    print(f"=== Phase 2: MPI-only sweep ({MATRIX}, {ITERATIONS} iterations, P-cores) ===\n")

    for cfg in CONFIGS:
        nranks = cfg["nranks"]
        label = f"mpi_{nranks}rank"

        print(f"--- {nranks} ranks ---")
        result = mpi_run(
            weight_file=Path(cfg["weight"]),
            matrix=MATRIX,
            iterations=ITERATIONS,
            rankfile=PROJECT_ROOT / cfg["rankfile"],
        )
        results[label] = {
            "nranks": nranks,
            "weight_file": cfg["weight"],
            "rankfile": cfg["rankfile"],
            "part_file": str(result["part_path"].relative_to(PROJECT_ROOT)),
            "returncode": result["returncode"],
            "stdout": result["stdout"],
            "stderr": result["stderr"],
            "metrics": result["metrics"],
        }

        if result["metrics"]:
            print(f"  spmv: {result['metrics'].get('spmv', 'N/A')}s")
        else:
            print(f"  ERROR (rc={result['returncode']}): {result['stderr'][:200]}")
        print()

    # Summary
    print("=== MPI-only Summary (spmv seconds) ===")
    for label, r in results.items():
        spmv = r["metrics"].get("spmv", "N/A") if r["metrics"] else "FAIL"
        print(f"  {label}: {spmv}")

    out_path = RESULTS_DIR / "mpi_sweep.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
