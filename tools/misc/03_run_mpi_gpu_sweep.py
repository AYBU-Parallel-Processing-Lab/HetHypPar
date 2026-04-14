#!/usr/bin/env python3
"""Phase 3: MPI+GPU sweep on P-cores for cage13 at 200 iterations.

Tests CPU rank counts (1,2,3,4,7) + 1 GPU rank, with GPU weight ratios
(500,1000,1500,2000,2720) : 100 CPU weight.
Uses quick_mpi_gpu_test.run() for partitioning and execution.
Outputs JSON results to tools/misc/results/mpi_gpu_sweep.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
from quick_mpi_gpu_test import run as mpi_gpu_run

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"
MATRIX = "cage13"
ITERATIONS = 200

CPU_RANK_COUNTS = [1, 2, 3, 4, 7]
GPU_WEIGHTS = [500, 1000, 1500, 2000, 2720]


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    print(f"=== Phase 3: MPI+GPU sweep ({MATRIX}, {ITERATIONS} iterations, P-cores) ===\n")

    for ncpu in CPU_RANK_COUNTS:
        nranks = ncpu + 1
        gpu_rank = ncpu
        rankfile = PROJECT_ROOT / "data" / "rankfile" / "experiment" / f"{ncpu}pcore_1gpu.rankfile"

        for gw in GPU_WEIGHTS:
            label = f"{ncpu}cpu_1gpu_w{gw}"
            weight_file = Path(f"data/weights/experiment-pcore-gpu/{label}.txt")

            print(f"--- {label} ({nranks} ranks, GPU on rank {gpu_rank}) ---")
            result = mpi_gpu_run(
                weight_file=weight_file,
                matrix=MATRIX,
                iterations=ITERATIONS,
                gpu_rank=gpu_rank,
                rankfile=rankfile,
            )
            results[label] = {
                "ncpu": ncpu,
                "nranks": nranks,
                "gpu_rank": gpu_rank,
                "gpu_weight": gw,
                "weight_file": str(weight_file),
                "rankfile": str(rankfile.relative_to(PROJECT_ROOT)),
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
    print("=== MPI+GPU Summary (spmv seconds) ===")
    print(f"{'config':<25} {'spmv':>8}")
    print("-" * 35)
    for label, r in results.items():
        spmv = r["metrics"].get("spmv", "N/A") if r["metrics"] else "FAIL"
        print(f"  {label:<23} {spmv:>8}")

    out_path = RESULTS_DIR / "mpi_gpu_sweep.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
