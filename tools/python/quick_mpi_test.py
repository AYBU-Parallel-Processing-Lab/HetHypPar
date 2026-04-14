#!/usr/bin/env python3
"""Quick MPI test: partition a matrix with patpart, then run bicgstab-mpi.

Can be used as a CLI tool or imported as a library:
    from quick_mpi_test import run
    result = run(weight_file=Path("data/weights/cpu-p-e/w100_16.txt"))
"""

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PATPART_BINARY = Path("/home/bugra/.local/bin/patpart")
BICGSTAB_MPI_BINARY = PROJECT_ROOT / "build" / "bicgstab-mpi"
MATRICES_DIR = PROJECT_ROOT / "data" / "matrices"
MAX_RANKS = 16

METRIC_PATTERNS = {
    "n_iters": r"n_iters\s*:\s*(\d+)",
    "spmv": r"spmv\s*:\s*([\d.]+)",
    "file_read": r"file_read\s*:\s*([\d.]+)",
    "relative_residual": r"relative_residual\s*:\s*([\d.E+\-]+)",
    "everything_total": r"everything_total\s*:\s*([\d.]+)",
}


def count_ranks(weight_file: Path) -> int:
    return len(weight_file.read_text().split())


def compute_weight_hash(weight_file: Path) -> str:
    contents = weight_file.read_bytes()
    return hashlib.sha256(contents).hexdigest()[:8]


def build_partition_path(
    matrix_name: str, nranks: int, imbalance: int, hash8: str
) -> tuple[Path, Path]:
    stem = f"{nranks}r_i{imbalance}_{hash8}"
    part_dir = MATRICES_DIR / matrix_name / "in" / "part"
    return part_dir / f"{stem}.part", part_dir / f"{stem}.log"


def parse_metrics(stdout: str) -> dict:
    """Parse solver output metrics. Returns dict with float/int values for found metrics."""
    metrics = {}
    for key, pattern in METRIC_PATTERNS.items():
        m = re.search(pattern, stdout)
        if m:
            val = m.group(1)
            metrics[key] = int(val) if key == "n_iters" else float(val)
    return metrics


def run_patpart(
    matrix_name: str,
    nranks: int,
    weight_file: Path,
    imbalance: int,
    seed: int,
    part_path: Path,
    log_path: Path,
) -> None:
    part_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(PATPART_BINARY),
        f"/matrices/{matrix_name}.mtx",
        str(nranks),
        str(weight_file),
        str(imbalance),
        str(seed),
        str(part_path),
        str(log_path),
    ]
    print(f"[patpart] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        msg = f"patpart failed (exit {result.returncode})"
        if result.stderr:
            msg += f"\n{result.stderr}"
        if result.stdout:
            msg += f"\n{result.stdout}"
        raise RuntimeError(msg)


def run(
    weight_file: Path,
    matrix: str = "cage13",
    imbalance: int = 1,
    seed: int = 42,
    iterations: int = 20,
    rankfile: Path | None = None,
) -> dict:
    """Run MPI solver with the given weight file.

    Args:
        rankfile: Optional OpenMPI rankfile for CPU pinning. When provided,
                  replaces the default '-bind-to core' with '--rankfile <path>'.

    Returns dict with keys: stdout, stderr, returncode, metrics, part_path, nranks.
    Raises RuntimeError on patpart failure or validation errors.
    """
    weight_file = Path(weight_file)
    if not weight_file.is_absolute():
        weight_file = PROJECT_ROOT / weight_file
    if not weight_file.exists():
        raise FileNotFoundError(f"Weight file not found: {weight_file}")

    nranks = count_ranks(weight_file)
    print(f"Weight file: {weight_file} ({nranks} ranks)")

    if nranks < 1 or nranks > MAX_RANKS:
        raise ValueError(f"Rank count {nranks} out of range [1, {MAX_RANKS}]")

    if not BICGSTAB_MPI_BINARY.exists():
        raise FileNotFoundError(f"Solver binary not found: {BICGSTAB_MPI_BINARY}")

    hash8 = compute_weight_hash(weight_file)
    part_path, log_path = build_partition_path(matrix, nranks, imbalance, hash8)

    if part_path.exists():
        print(f"Partition file already exists, reusing: {part_path}")
    else:
        run_patpart(matrix, nranks, weight_file, imbalance, seed, part_path, log_path)
        print(f"Partition file created: {part_path}")

    mat_basedir = MATRICES_DIR / matrix
    (mat_basedir / "out").mkdir(parents=True, exist_ok=True)
    mpi_args = ["mpirun", "--report-bindings"]
    if rankfile is not None:
        mpi_args.extend(["--rankfile", str(rankfile)])
    else:
        mpi_args.extend(["-bind-to", "core"])
    mpi_args.extend(["-n", str(nranks)])
    cmd = [
        *mpi_args,
        str(BICGSTAB_MPI_BINARY),
        "-m", f"/matrices/{matrix}.mtx",
        "-n", str(iterations),
        "-p", str(part_path),
        "-o", str(mat_basedir / "out" / "X_mpi.txt"),
        "-x", str(mat_basedir / "in" / "X_init.txt"),
        "-y", str(mat_basedir / "in" / "B.txt"),
    ]
    print(f"[mpirun] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=PROJECT_ROOT)

    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
        "metrics": parse_metrics(result.stdout),
        "part_path": part_path,
        "nranks": nranks,
    }


def main():
    parser = argparse.ArgumentParser(description="Quick MPI test runner")
    parser.add_argument("-w", "--weight-file", type=Path, required=True)
    parser.add_argument("-m", "--matrix", default="cage13")
    parser.add_argument("-i", "--imbalance", type=int, default=1)
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-n", "--iterations", type=int, default=20)
    parser.add_argument("-r", "--rankfile", type=Path, default=None,
                        help="OpenMPI rankfile for CPU pinning")
    args = parser.parse_args()

    try:
        result = run(
            weight_file=args.weight_file,
            matrix=args.matrix,
            imbalance=args.imbalance,
            seed=args.seed,
            iterations=args.iterations,
            rankfile=args.rankfile,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(result["stdout"], end="")
    if result["stderr"]:
        print(result["stderr"], end="", file=sys.stderr)
    if result["returncode"] != 0:
        print(f"\nmpirun exited with code {result['returncode']}", file=sys.stderr)
        sys.exit(result["returncode"])


if __name__ == "__main__":
    main()
