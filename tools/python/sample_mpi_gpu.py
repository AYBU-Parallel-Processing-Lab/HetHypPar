#!/usr/bin/env python3

import subprocess
import re
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import hashlib

# Configuration
MATRICES_DIR = Path("data/matrices")
LOGS_DIR = Path("data/logs")
BUILD_DIR = Path("build")
BICGSTAB_MPI_GPU_BINARY = BUILD_DIR / "bicgstab-mpi-gpu"
BICGSTAB_MPI_BINARY = BUILD_DIR / "bicgstab-mpi"
BICGSTAB_CPU_BINARY = BUILD_DIR / "bicgstab-cpu"
BICGSTAB_GPU_BINARY = BUILD_DIR / "bicgstab-gpu"

def log_command(log_dir: Path, log_prefix: str, cmd: list[str], stdout: str, stderr: str, postfix="success") -> None:
    """Log command, stdout, and stderr to separate files."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Write command
    cmd_file = log_dir / f"{log_prefix}_cmd_{postfix}.txt"
    cmd_file.write_text(" ".join(cmd) + "\n")
    
    # Write stdout
    stdout_file = log_dir / f"{log_prefix}_stdout_{postfix}.txt"
    stdout_file.write_text(stdout)
    
    # Write stderr
    stderr_file = log_dir / f"{log_prefix}_stderr_{postfix}.txt"
    stderr_file.write_text(stderr)


def get_partition_files_hash(part_files: list[Path]) -> str:
    """Generate a hash from the sorted list of partition filenames, including cpu and gpu."""
    # Add 'cpu' and 'gpu' to the list of filenames
    filenames = ["cpu", "gpu"] + sorted([pf.name for pf in part_files])
    hash_str = "|".join(filenames)
    return hashlib.sha256(hash_str.encode()).hexdigest()[:16]


def check_existing_results(log_dir: Path, matrix_name: str, part_files_hash: str) -> bool:
    """Check if results already exist for this set of partition files."""
    if not log_dir.exists():
        return False

    # Look for TSV files matching the hash pattern
    pattern = f"benchmark_results_{matrix_name}_*_{part_files_hash}.tsv"
    existing_files = list(log_dir.glob(pattern))

    return len(existing_files) > 0


def parse_output(output: str) -> dict:
    """Parse the output from bicgstab binaries and extract metrics."""
    metrics = {}

    # Use regex to extract values
    patterns = {
        "n_iters": r"n_iters\s*:\s*(\d+)",
        "spmv": r"spmv\s*:\s*([\d.]+)",
        "file_read": r"file_read\s*:\s*([\d.]+)",
        "relative_residual": r"relative_residual\s*:\s*([\d.E+-]+)",
        "everything_total": r"everything_total\s*:\s*([\d.]+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            metrics[key] = match.group(1)
        else:
            raise ValueError(f"Could not find '{key}' in output")

    return metrics


def count_mpi_procs(part_file: Path) -> int:
    """Count unique partition IDs in the partition file."""
    with open(part_file, "r") as f:
        partitions = set(line.strip() for line in f if line.strip())
    return len(partitions)


def run_cpu_benchmark(matrix_name: str, iterations: int, pbar_part: tqdm) -> dict:
    """Run the CPU benchmark."""
    mat_basedir = Path("data/matrices") / matrix_name

    pbar_part.set_description(f"  [{matrix_name}] cpu (sequential)")

    # Build command
    cmd = [
        str(BICGSTAB_CPU_BINARY),
        "-m",
        f"/matrices/{matrix_name}.mtx",
        "-n",
        str(iterations),
        "-o",
        str(mat_basedir / "out" / "X_seq.txt"),
        "-x",
        str(mat_basedir / "in" / "X_init.txt"),
        "-y",
        str(mat_basedir / "in" / "B.txt"),
    ]

    # Run command and capture output
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    
    postfix = "success" if result.returncode == 0 else "failure"
    
    # Log command execution
    log_prefix = f"cpu_{matrix_name}"
    log_command(LOGS_DIR, log_prefix, cmd, result.stdout, result.stderr, postfix)
    
    # Check if command succeeded
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)

    # Parse output
    metrics = parse_output(result.stdout)
    metrics["partition_file"] = "cpu"
    metrics["mpi_procs"] = 1

    return metrics


def run_gpu_benchmark(matrix_name: str, iterations: int, pbar_part: tqdm) -> dict:
    """Run the GPU benchmark."""
    mat_basedir = Path("data/matrices") / matrix_name

    pbar_part.set_description(f"  [{matrix_name}] gpu")

    # Build command
    cmd = [
        str(BICGSTAB_GPU_BINARY),
        "-m",
        f"/matrices/{matrix_name}.mtx",
        "-n",
        str(iterations),
        "-o",
        str(mat_basedir / "out" / "X_gpu.txt"),
        "-x",
        str(mat_basedir / "in" / "X_init.txt"),
        "-y",
        str(mat_basedir / "in" / "B.txt"),
    ]

    # Run command and capture output
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    
    postfix = "success" if result.returncode == 0 else "failure"
    
    # Log command execution
    log_prefix = f"gpu_{matrix_name}"
    log_command(LOGS_DIR, log_prefix, cmd, result.stdout, result.stderr, postfix)
    
    # Check if command succeeded
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)

    # Parse output
    metrics = parse_output(result.stdout)
    metrics["partition_file"] = "gpu"
    metrics["mpi_procs"] = 1

    return metrics


def run_mpi_benchmark(
    matrix_name: str, part_file: Path, mpi_part_dir: Path, iterations: int, pbar_part: tqdm
) -> dict:
    """Run the MPI benchmark for a single partition file."""
    mat_basedir = Path("data/matrices") / matrix_name
    part_file_path = mat_basedir / mpi_part_dir / part_file.name

    # Verify partition file exists
    if not part_file_path.exists():
        raise FileNotFoundError(f"Partition file not found: {part_file_path}")

    # Count MPI processes
    mpi_procs = count_mpi_procs(part_file_path)
    pbar_part.set_description(f"  [{matrix_name}] {part_file.name} ({mpi_procs} procs)")

    # Build command
    cmd = [
        "mpirun",
        "--report-bindings",
        "-bind-to",
        "core",
        "-n",
        str(mpi_procs),
        str(BICGSTAB_MPI_BINARY),
        "-m",
        f"/matrices/{matrix_name}.mtx",
        "-n",
        str(iterations),
        "-p",
        str(part_file_path),
        "-o",
        str(mat_basedir / "out" / f"X_mpi_{part_file.stem}.txt"),
        "-x",
        str(mat_basedir / "in" / "X_init.txt"),
        "-y",
        str(mat_basedir / "in" / "B.txt"),
    ]

    # Run command and capture output
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    
    postfix = "success" if result.returncode == 0 else "failure"
    
    # Log command execution
    log_prefix = f"mpi_{matrix_name}_{part_file.stem}"
    log_command(LOGS_DIR, log_prefix, cmd, result.stdout, result.stderr, postfix)
    
    # Check if command succeeded
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)

    # Parse output
    metrics = parse_output(result.stdout)
    metrics["partition_file"] = part_file.name
    metrics["mpi_procs"] = mpi_procs

    return metrics

def run_mpi_gpu_benchmark(
        matrix_name: str, part_file: Path, mpi_gpu_part_dir: Path, is_gpu: Path, rankfile: Path, iterations: int, pbar_part: tqdm
) -> dict:
    """Run the MPI GPU benchmark for a single partition file."""
    mat_basedir = Path("data/matrices") / matrix_name
    part_file_path = mat_basedir / mpi_gpu_part_dir / part_file.name

    # Verify partition file exists
    if not part_file_path.exists():
        raise FileNotFoundError(f"Partition file not found: {part_file_path}")

    # Count MPI processes
    mpi_procs = count_mpi_procs(part_file_path)
    pbar_part.set_description(f"  [{matrix_name}] {part_file.name} ({mpi_procs} procs)")

    # Build command
    cmd = [
        "mpirun",
        "--report-bindings",
        "-bind-to", ":overload-allowed",
        "--rankfile", str(rankfile),
        str(BICGSTAB_MPI_GPU_BINARY),
        "-m", f"/matrices/{matrix_name}.mtx",
        "-n", str(iterations),
        "-p", str(part_file_path),
        "-o", str(mat_basedir / "out" / f"X_mpi_gpu_{part_file.stem}.txt"),
        "-x", str(mat_basedir / "in" / "X_init.txt"),
        "-y", str(mat_basedir / "in" / "B.txt"),
        "-g", str(is_gpu),
    ]

    # Run command and capture output
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    
    postfix = "success" if result.returncode == 0 else "failure"
    
    # Log command execution
    log_prefix = f"mpi_gpu_{matrix_name}_{part_file.stem}"
    log_command(LOGS_DIR, log_prefix, cmd, result.stdout, result.stderr,postfix)
    
    # Check if command succeeded
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)

    # Parse output
    metrics = parse_output(result.stdout)
    metrics["partition_file"] = part_file.name
    metrics["mpi_procs"] = mpi_procs

    return metrics




failed = 0


def process_matrix(
    matrix_name: str, 
    pbar_matrix: tqdm, 
    pbar_part: tqdm,
    iterations: int,
    mpi_part_dir: Path | None,
    mpi_gpu_part_dir: Path | None,
    rankfile: Path,
    is_gpu_file: Path
) -> pd.DataFrame:
    """Process CPU, GPU, and all partition files for a single matrix."""
    mat_basedir = MATRICES_DIR / matrix_name
    
    # NEW: Use unified directory
    log_dir = LOGS_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    global failed

    # Ensure output and log directories exist
    (mat_basedir / "out").mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Get all partition files (may not exist, which is ok)
    part_files = []
    if mpi_part_dir is not None:
        part_dir = mat_basedir / mpi_part_dir
        part_files = sorted(part_dir.glob("*.part")) if part_dir.exists() else []
    
    part_gpu_files = []
    if mpi_gpu_part_dir is not None:
        part_gpu_dir = mat_basedir / mpi_gpu_part_dir
        part_gpu_files = sorted(part_gpu_dir.glob("*.part")) if part_gpu_dir.exists() else []

    # Generate hash of partition files (includes cpu and gpu)
    part_files_hash = get_partition_files_hash(part_files)
    part_gpu_files_hash = get_partition_files_hash(part_gpu_files)

    # Check if results already exist
    if check_existing_results(log_dir, matrix_name, part_gpu_files_hash):
        pbar_matrix.write(
            f"⏭ Skipping {matrix_name}: Results already exist (hash: {part_gpu_files_hash})"
        )
        return pd.DataFrame()

    if check_existing_results(log_dir, matrix_name, part_files_hash):
        pbar_matrix.write(
            f"⏭ Skipping {matrix_name}: Results already exist (hash: {part_files_hash})"
        )
        return pd.DataFrame()

    # Set up partition progress bar (2 for cpu/gpu + number of partition files)
    total_runs = 2 + len(part_files) + len(part_gpu_files)
    pbar_part.reset(total=total_runs)
    pbar_part.set_description(f"  [{matrix_name}] Running benchmarks")

    # Run benchmarks and collect results
    results = []

    # Run CPU benchmark
    try:
        metrics = run_cpu_benchmark(matrix_name, iterations, pbar_part)
        metrics["matrix_name"] = matrix_name
        results.append(metrics)
        pbar_part.update(1)
    except Exception as e:
        failed += 1
        pbar_matrix.write(f"✗ {matrix_name}/cpu failed: {e}")
        pbar_part.update(1)

    # Run GPU benchmark
    try:
        metrics = run_gpu_benchmark(matrix_name, iterations, pbar_part)
        metrics["matrix_name"] = matrix_name
        results.append(metrics)
        pbar_part.update(1)
    except Exception as e:
        failed += 1
        pbar_matrix.write(f"✗ {matrix_name}/gpu failed: {e}")
        pbar_part.update(1)

    # Run MPI benchmarks for all partition files
    for part_file in part_files:
        try:
            metrics = run_mpi_benchmark(matrix_name, part_file, mpi_part_dir, iterations, pbar_part)
            metrics["matrix_name"] = matrix_name
            results.append(metrics)
            pbar_part.update(1)
        except Exception as e:
            failed += 1
            pbar_matrix.write(f"✗ {matrix_name}/{part_file.name} failed: {e}")
            pbar_part.update(1)
    
    for part_gpu_file in part_gpu_files:
        try:
            metrics = run_mpi_gpu_benchmark(matrix_name, part_gpu_file, mpi_gpu_part_dir, is_gpu_file, rankfile, iterations, pbar_part)
            metrics["matrix_name"] = matrix_name
            results.append(metrics)
            pbar_part.update(1)
        except Exception as e:
            failed += 1
            pbar_matrix.write(f"✗ {matrix_name}/{part_gpu_file.name} failed: {e}")
            pbar_part.update(1)


    # Convert to DataFrame
    df = pd.DataFrame(results)

    if df.empty:
        pbar_matrix.write(f"⚠ {matrix_name}: No successful benchmarks.")
        return df

    # Reorder columns
    columns = [
        "matrix_name",
        "partition_file",
        "mpi_procs",
        "n_iters",
        "spmv",
        "file_read",
        "relative_residual",
        "everything_total",
    ]
    df = df[columns]

    # Write results to TSV with hash in filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # NEW: Added {matrix_name} to filename so they don't overwrite each other
    tsv_file = log_dir / f"benchmark_results_{matrix_name}_{timestamp}_{part_files_hash}_{part_gpu_files_hash}.tsv"

    df.to_csv(tsv_file, sep="\t", index=False)

    pbar_matrix.write(f"✓ {matrix_name}: {len(results)} benchmarks → {tsv_file}")

    return df


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run BiCGSTAB benchmarks for CPU, GPU, MPI, and MPI-GPU configurations"
    )
    
    parser.add_argument(
        "--iterations",
        type=int,
        required=True,
        help="Number of iterations per benchmark"
    )
    
    parser.add_argument(
        "--mpi-part-dir",
        type=Path,
        default=None,
        help="Partition directory for MPI (relative to matrix directory). If not specified, MPI benchmarks are skipped."
    )
    
    parser.add_argument(
        "--mpi-gpu-part-dir",
        type=Path,
        required=True,
        help="Partition directory for MPI-GPU (relative to matrix directory)"
    )
    
    parser.add_argument(
        "--rankfile",
        type=Path,
        required=True,
        help="Path to rankfile for MPI-GPU runs"
    )
    
    parser.add_argument(
        "--is-gpu-file",
        type=Path,
        required=True,
        help="Path to is_gpu file for MPI-GPU runs"
    )
    
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Verify binaries exist
    for binary, name in [
        (BICGSTAB_CPU_BINARY, "CPU"),
        (BICGSTAB_GPU_BINARY, "GPU"),
        (BICGSTAB_MPI_BINARY, "MPI"),
        (BICGSTAB_MPI_GPU_BINARY, "MPI_GPU"),
    ]:
        if not binary.exists():
            raise FileNotFoundError(f"{name} binary not found: {binary}")

    # Verify rankfile and is_gpu_file exist
    if not args.rankfile.exists():
        raise FileNotFoundError(f"Rankfile not found: {args.rankfile}")
    
    if not args.is_gpu_file.exists():
        raise FileNotFoundError(f"is_gpu file not found: {args.is_gpu_file}")

    # Verify matrices directory exists
    if not MATRICES_DIR.exists():
        raise FileNotFoundError(f"Matrices directory not found: {MATRICES_DIR}")

    # Get all matrix directories
    matrix_dirs = sorted([d for d in MATRICES_DIR.iterdir() if d.is_dir()])

    if not matrix_dirs:
        raise FileNotFoundError(f"No matrix directories found in {MATRICES_DIR}")

    matrix_names = [d.name for d in matrix_dirs]

    print(f"Found {len(matrix_names)} matrices: {', '.join(matrix_names)}")
    print(f"Iterations per benchmark: {args.iterations}")
    if args.mpi_part_dir:
        print(f"MPI partition directory: {args.mpi_part_dir}")
    else:
        print("MPI benchmarks: SKIPPED (no --mpi-part-dir specified)")
    print(f"MPI-GPU partition directory: {args.mpi_gpu_part_dir}")
    print(f"Rankfile: {args.rankfile}")
    print(f"is_gpu file: {args.is_gpu_file}\n")

    # Create progress bars
    pbar_matrix = tqdm(
        total=len(matrix_names), desc="Overall Progress", position=0, leave=True
    )
    pbar_part = tqdm(total=0, desc="  Benchmark Runs", position=1, leave=True)

    total_benchmarks = 0
    all_results = []
    skipped = 0
    global failed
    try:
        for matrix_name in matrix_names:
            pbar_matrix.set_description(f"Processing {matrix_name}")
            df = process_matrix(
                matrix_name, 
                pbar_matrix, 
                pbar_part,
                args.iterations,
                args.mpi_part_dir,
                args.mpi_gpu_part_dir,
                args.rankfile,
                args.is_gpu_file
            )
            if not df.empty:
                all_results.append(df)
                total_benchmarks += len(df)
            else:
                skipped += 1
            pbar_matrix.update(1)

        # Final cleanup of progress bars
        pbar_part.close()
        pbar_matrix.close()

        print(f"\n{'=' * 60}")
        print(f"All benchmarks completed successfully!")
        print(f"Total matrices processed: {len(matrix_names)}")
        print(f"Total skipped: {skipped}")
        print(f"Total failed: {failed}")
        print(f"Total benchmarks run: {total_benchmarks}")
        print(f"{'=' * 60}")

    except Exception as e:
        # Ensure progress bars are cleaned up on error
        pbar_part.close()
        pbar_matrix.close()
        raise


if __name__ == "__main__":
    main()