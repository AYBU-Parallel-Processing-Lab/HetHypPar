#!/usr/bin/env python3
"""Phase 4: Analyze experiment results and produce summary table.

Reads JSON files from tools/misc/results/ and prints a comparison table.
"""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_json(name: str) -> dict:
    path = RESULTS_DIR / name
    if not path.exists():
        print(f"WARNING: {name} not found, skipping")
        return {}
    with open(path) as f:
        return json.load(f)


def main():
    baselines = load_json("baselines.json")
    mpi_sweep = load_json("mpi_sweep.json")
    mpi_gpu_sweep = load_json("mpi_gpu_sweep.json")

    # Extract GPU baseline spmv
    gpu_spmv = None
    cpu_spmv = None
    if baselines.get("gpu", {}).get("metrics"):
        gpu_spmv = baselines["gpu"]["metrics"]["spmv"]
    if baselines.get("cpu", {}).get("metrics"):
        cpu_spmv = baselines["cpu"]["metrics"]["spmv"]

    print("=" * 70)
    print("EXPERIMENT RESULTS: Beat GPU SpMV on cage13")
    print("=" * 70)

    # Baselines
    print("\n--- Baselines ---")
    print(f"  CPU (1 process):  spmv = {cpu_spmv}s" if cpu_spmv else "  CPU: FAILED")
    print(f"  GPU (1 process):  spmv = {gpu_spmv}s  <-- TARGET TO BEAT" if gpu_spmv else "  GPU: FAILED")

    # MPI-only results
    if mpi_sweep:
        print("\n--- MPI-only (P-cores, uniform weights) ---")
        print(f"  {'Config':<20} {'Ranks':>5} {'SpMV (s)':>10} {'vs GPU':>10}")
        print("  " + "-" * 47)
        for label, r in sorted(mpi_sweep.items(), key=lambda x: x[1].get("nranks", 0)):
            spmv = r.get("metrics", {}).get("spmv")
            if spmv is not None:
                ratio = f"{spmv / gpu_spmv:.2f}x" if gpu_spmv else "N/A"
                print(f"  {label:<20} {r['nranks']:>5} {spmv:>10.4f} {ratio:>10}")
            else:
                print(f"  {label:<20} {r.get('nranks', '?'):>5} {'FAIL':>10}")

    # MPI+GPU results
    if mpi_gpu_sweep:
        print("\n--- MPI+GPU (P-cores + RTX 3070) ---")
        print(f"  {'Config':<25} {'CPU+GPU':>7} {'GPU wt':>7} {'SpMV (s)':>10} {'vs GPU':>10} {'Beat?':>6}")
        print("  " + "-" * 67)

        best_label = None
        best_spmv = float("inf")

        for label, r in sorted(mpi_gpu_sweep.items(), key=lambda x: (x[1].get("ncpu", 0), x[1].get("gpu_weight", 0))):
            spmv = r.get("metrics", {}).get("spmv")
            ncpu = r.get("ncpu", "?")
            gw = r.get("gpu_weight", "?")
            if spmv is not None:
                ratio = f"{spmv / gpu_spmv:.2f}x" if gpu_spmv else "N/A"
                beats = "YES" if gpu_spmv and spmv < gpu_spmv else "no"
                print(f"  {label:<25} {ncpu}+1{'':<3} {gw:>7} {spmv:>10.4f} {ratio:>10} {beats:>6}")
                if spmv < best_spmv:
                    best_spmv = spmv
                    best_label = label
            else:
                print(f"  {label:<25} {ncpu}+1{'':<3} {gw:>7} {'FAIL':>10}")

        if best_label and gpu_spmv:
            print(f"\n  Best MPI+GPU: {best_label} = {best_spmv:.4f}s")
            speedup = gpu_spmv / best_spmv
            if best_spmv < gpu_spmv:
                print(f"  --> {speedup:.2f}x FASTER than GPU-only!")
            else:
                print(f"  --> {speedup:.2f}x of GPU-only (still slower)")

    # Overall summary
    if gpu_spmv:
        print("\n" + "=" * 70)
        print("OVERALL SUMMARY")
        print("=" * 70)
        print(f"  GPU-only SpMV:        {gpu_spmv:.4f}s")
        if mpi_sweep:
            best_mpi = min(
                (r["metrics"]["spmv"] for r in mpi_sweep.values() if r.get("metrics", {}).get("spmv")),
                default=None,
            )
            if best_mpi:
                print(f"  Best MPI-only SpMV:   {best_mpi:.4f}s ({gpu_spmv / best_mpi:.2f}x {'faster' if best_mpi < gpu_spmv else 'slower'} than GPU)")
        if mpi_gpu_sweep and best_label:
            print(f"  Best MPI+GPU SpMV:    {best_spmv:.4f}s ({gpu_spmv / best_spmv:.2f}x {'faster' if best_spmv < gpu_spmv else 'slower'} than GPU)")


if __name__ == "__main__":
    main()
