#!/usr/bin/env python3
"""Create weight files, rankfiles, and is_gpu files for the P-core experiment.

Weight files:
  - data/weights/experiment-pcore/equal_<N>.txt  (N uniform lines of 100.0)
  - data/weights/experiment-pcore-gpu/<C>cpu_1gpu_w<G>.txt  (C lines of 100.0 + 1 line of G)

Rankfiles:
  - data/rankfile/experiment/<N>pcore.rankfile         (N ranks on P-core CPUs)
  - data/rankfile/experiment/<C>pcore_1gpu.rankfile     (C CPU ranks + 1 GPU rank on P-cores)

is_gpu files:
  - Created via quick_mpi_gpu_test.get_is_gpu_file() convention

Hardware: i7-13700F P-cores are physical cores 0-7, with HT siblings on CPUs 0-15.
We pin one rank per physical P-core (use even CPU IDs: 0,2,4,6,8,10,12,14).
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WEIGHTS_MPI = PROJECT_ROOT / "data" / "weights" / "experiment-pcore"
WEIGHTS_GPU = PROJECT_ROOT / "data" / "weights" / "experiment-pcore-gpu"
RANKFILE_DIR = PROJECT_ROOT / "data" / "rankfile" / "experiment"
IS_GPU_DIR = PROJECT_ROOT / "data" / "is_gpu"

# P-core physical core -> CPU ID mapping (one thread per physical core)
# Core 0 -> CPU 0, Core 1 -> CPU 2, ..., Core 7 -> CPU 14
PCORE_CPUS = [0, 2, 4, 6, 8, 10, 12, 14]

# --- MPI-only configs ---
MPI_RANK_COUNTS = [2, 4, 6, 8]

# --- MPI+GPU configs ---
CPU_RANK_COUNTS = [1, 2, 3, 4, 7]  # + 1 GPU rank each
GPU_WEIGHTS = [500, 1000, 1500, 2000, 2720]


def create_weight_file(path: Path, weights: list[float]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f"{w:.1f}" for w in weights) + "\n")
    print(f"  {path.relative_to(PROJECT_ROOT)}  ({len(weights)} ranks)")


def create_rankfile(path: Path, lines: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"  {path.relative_to(PROJECT_ROOT)}")


def create_is_gpu_file(nranks: int, gpu_rank: int) -> Path:
    IS_GPU_DIR.mkdir(parents=True, exist_ok=True)
    path = IS_GPU_DIR / f"{nranks}r_gpu{gpu_rank}.txt"
    if path.exists():
        return path
    content = ["1" if r == gpu_rank else "0" for r in range(nranks)]
    path.write_text("\n".join(content) + "\n")
    print(f"  {path.relative_to(PROJECT_ROOT)}")
    return path


def main():
    print("=== MPI-only weight files (uniform, P-cores) ===")
    for n in MPI_RANK_COUNTS:
        create_weight_file(WEIGHTS_MPI / f"equal_{n}.txt", [100.0] * n)

    print("\n=== MPI-only rankfiles (P-cores) ===")
    for n in MPI_RANK_COUNTS:
        lines = [f"rank {i}=localhost slot={PCORE_CPUS[i]}" for i in range(n)]
        create_rankfile(RANKFILE_DIR / f"{n}pcore.rankfile", lines)

    print("\n=== MPI+GPU weight files (heterogeneous) ===")
    for ncpu in CPU_RANK_COUNTS:
        for gw in GPU_WEIGHTS:
            nranks = ncpu + 1
            weights = [100.0] * ncpu + [float(gw)]
            fname = f"{ncpu}cpu_1gpu_w{gw}.txt"
            create_weight_file(WEIGHTS_GPU / fname, weights)

    print("\n=== MPI+GPU rankfiles (CPU on P-cores, GPU on last P-core) ===")
    for ncpu in CPU_RANK_COUNTS:
        nranks = ncpu + 1
        gpu_rank = ncpu  # last rank
        lines = []
        for i in range(ncpu):
            lines.append(f"rank {i}=localhost slot={PCORE_CPUS[i]}")
        # GPU rank gets its own P-core (next available)
        lines.append(f"rank {gpu_rank}=localhost slot={PCORE_CPUS[ncpu]}")
        create_rankfile(RANKFILE_DIR / f"{ncpu}pcore_1gpu.rankfile", lines)

    print("\n=== is_gpu files ===")
    for ncpu in CPU_RANK_COUNTS:
        nranks = ncpu + 1
        gpu_rank = ncpu
        create_is_gpu_file(nranks, gpu_rank)

    print("\nDone.")


if __name__ == "__main__":
    main()
