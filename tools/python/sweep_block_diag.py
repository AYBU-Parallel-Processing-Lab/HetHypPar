#!/usr/bin/env python3
"""
Sweep block-diagonal configurations to find one where mpi_gpu beats gpu-only.

For each (base_matrix, cpu_blocks, gpu_blocks) configuration:
  1. Generate a block-diagonal matrix via setup_n_block_diag.py
  2. Run bicgstab-gpu on the generated matrix (baseline)
  3. Run bicgstab-mpi-gpu with the cutsize-0 partition
  4. Parse metrics and append to outdir/sweep_results.tsv

Generated matrices live under data/matrices/<base>_<c>c_<g>g/.
Solver stdout/stderr is kept under outdir/logs/ for inspection.

Usage:
  python tools/python/sweep_block_diag.py \\
    --base /matrices/bips07_1693.mtx /matrices/cage10.mtx \\
    --configs 1,1 1,3 1,9 1,15 \\
    --iters 200 \\
    --outdir data/block_sweep
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SETUP_SCRIPT = PROJECT_ROOT / "tools" / "python" / "setup_n_block_diag.py"
BICGSTAB_GPU = PROJECT_ROOT / "build" / "bicgstab-gpu"
BICGSTAB_MPI_GPU = PROJECT_ROOT / "build" / "bicgstab-mpi-gpu"

METRIC_PATTERNS = {
    "n_iters": r"n_iters\s*:\s*(\d+)",
    "spmv": r"spmv\s*:\s*([\d.eE+-]+)",
    "file_read": r"file_read\s*:\s*([\d.eE+-]+)",
    "relative_residual": r"relative_residual\s*:\s*([\d.eE+-NAan]+)",
    "everything_total": r"everything_total\s*:\s*([\d.eE+-]+)",
}


def parse_metrics(text):
    out = {}
    for k, pat in METRIC_PATTERNS.items():
        m = re.search(pat, text)
        out[k] = m.group(1) if m else ""
    return out


def parse_profile_accum(text):
    """Return {rank: {field: float}} from PROFILE_ACCUM lines."""
    accum = {}
    for line in text.splitlines():
        if not line.startswith("PROFILE_ACCUM "):
            continue
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        rank = int(parts[1])
        kvs = {}
        for tok in parts[2].split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                try:
                    kvs[k] = float(v)
                except ValueError:
                    pass
        accum[rank] = kvs
    return accum


def run_setup(base_mtx, base_name, cpu_blocks, gpu_blocks):
    """Generate the block-diagonal matrix and partition.

    setup_n_block_diag.py appends '_<c>c_<g>g' to the name we pass it,
    so we pass `base_name` and compute the final dir as base_name_<c>c_<g>g.
    """
    final_name = f"{base_name}_{cpu_blocks}c_{gpu_blocks}g"
    out_dir = PROJECT_ROOT / "data" / "matrices" / final_name
    if (out_dir / f"{final_name}.mtx").exists():
        print(f"  [skip setup] {out_dir} already exists")
        return out_dir, final_name

    print(f"  generating {final_name} ({cpu_blocks}c+{gpu_blocks}g) from {base_mtx}")
    result = subprocess.run(
        ["micromamba", "run", "-n", "octave", "python", str(SETUP_SCRIPT),
         base_name, str(base_mtx), "-c", str(cpu_blocks), "-g", str(gpu_blocks)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  setup FAILED: {result.stderr[-500:]}")
        return None, None
    return out_dir, final_name


def run_solver(cmd, stdout_path, stderr_path):
    """Run a solver, capture output. Returns (returncode, stdout_text)."""
    with open(stdout_path, "w") as fout, open(stderr_path, "w") as ferr:
        result = subprocess.run(cmd, stdout=fout, stderr=ferr, cwd=PROJECT_ROOT)
    stdout_text = Path(stdout_path).read_text()
    return result.returncode, stdout_text


def run_gpu(mtx_dir, name, iters, log_dir):
    cmd = [
        str(BICGSTAB_GPU),
        "-m", str(mtx_dir / f"{name}.mtx"),
        "-x", str(mtx_dir / "in" / "X_init.txt"),
        "-y", str(mtx_dir / "in" / "B.txt"),
        "-o", str(mtx_dir / "out" / "X_gpu.txt"),
        "-n", str(iters),
    ]
    return run_solver(
        cmd,
        log_dir / f"{name}_gpu.stdout",
        log_dir / f"{name}_gpu.stderr",
    )


def run_mpi_gpu(mtx_dir, name, iters, log_dir):
    cmd = [
        "mpirun", "-n", "2",
        str(BICGSTAB_MPI_GPU),
        "-m", str(mtx_dir / f"{name}.mtx"),
        "-x", str(mtx_dir / "in" / "X_init.txt"),
        "-y", str(mtx_dir / "in" / "B.txt"),
        "-o", str(mtx_dir / "out" / "X_mpi_gpu.txt"),
        "-p", str(mtx_dir / "in" / "part" / "workload.part"),
        "-g", str(PROJECT_ROOT / "data" / "is_gpu" / "g2_2_indep.txt"),
        "-n", str(iters),
    ]
    return run_solver(
        cmd,
        log_dir / f"{name}_mpi_gpu.stdout",
        log_dir / f"{name}_mpi_gpu.stderr",
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", nargs="+", required=True,
                   help="Base .mtx files (full paths)")
    p.add_argument("--configs", nargs="+", required=True,
                   help="cpu,gpu block counts (e.g. 1,1 1,3 1,9)")
    p.add_argument("--iters", type=int, default=200,
                   help="Solver iteration count (default 200)")
    p.add_argument("--outdir", default="data/block_sweep",
                   help="Output directory for logs and summary TSV")
    p.add_argument("--cleanup", action="store_true",
                   help="Delete generated matrices after each config")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = PROJECT_ROOT / outdir
    log_dir = outdir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Parse config strings into tuples
    configs = []
    for c in args.configs:
        try:
            cb, gb = map(int, c.split(","))
            configs.append((cb, gb))
        except ValueError:
            print(f"bad config '{c}', expected 'cpu,gpu' (e.g. 1,3)")
            sys.exit(1)

    out_tsv = outdir / "sweep_results.tsv"
    fieldnames = [
        "base_matrix", "cpu_blocks", "gpu_blocks", "total_blocks", "name",
        "iters",
        "gpu_status", "gpu_spmv", "gpu_residual",
        "mpi_gpu_status", "mpi_gpu_spmv", "mpi_gpu_residual",
        "speedup_ratio",  # gpu_spmv / mpi_gpu_spmv (>1 means mpi_gpu wins)
        "r0_local_spmv", "r0_shared_spmv", "r0_vecops",
        "r0_comm_wait", "r0_send_fill", "r0_send_wait",
        "r1_local_spmv", "r1_shared_spmv", "r1_vecops",
        "r1_comm_wait", "r1_send_fill", "r1_send_wait",
    ]

    # Open in append mode if file exists, else write header
    write_header = not out_tsv.exists()
    fout = open(out_tsv, "a", newline="")
    writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter="\t")
    if write_header:
        writer.writeheader()

    for base in args.base:
        base_path = Path(base)
        base_name = base_path.stem
        for cb, gb in configs:
            total = cb + gb
            print(f"\n=== {base_name} {cb}c+{gb}g (iters={args.iters}) ===")

            mtx_dir, name = run_setup(base_path, base_name, cb, gb)
            if mtx_dir is None:
                continue

            print(f"  running gpu baseline...")
            gpu_rc, gpu_out = run_gpu(mtx_dir, name, args.iters, log_dir)
            gpu_metrics = parse_metrics(gpu_out)

            print(f"  running mpi_gpu...")
            mg_rc, mg_out = run_mpi_gpu(mtx_dir, name, args.iters, log_dir)
            mg_metrics = parse_metrics(mg_out)
            prof = parse_profile_accum(mg_out)

            # Compute speedup if both succeeded
            speedup = ""
            try:
                if gpu_rc == 0 and mg_rc == 0:
                    g = float(gpu_metrics["spmv"])
                    m = float(mg_metrics["spmv"])
                    if m > 0:
                        speedup = f"{g/m:.4f}"
            except (ValueError, KeyError):
                pass

            row = {
                "base_matrix": base_name,
                "cpu_blocks": cb,
                "gpu_blocks": gb,
                "total_blocks": total,
                "name": name,
                "iters": args.iters,
                "gpu_status": "ok" if gpu_rc == 0 else f"fail({gpu_rc})",
                "gpu_spmv": gpu_metrics.get("spmv", ""),
                "gpu_residual": gpu_metrics.get("relative_residual", ""),
                "mpi_gpu_status": "ok" if mg_rc == 0 else f"fail({mg_rc})",
                "mpi_gpu_spmv": mg_metrics.get("spmv", ""),
                "mpi_gpu_residual": mg_metrics.get("relative_residual", ""),
                "speedup_ratio": speedup,
            }
            for rank in (0, 1):
                src = prof.get(rank, {})
                for field in ("local_spmv", "shared_spmv", "vecops",
                              "comm_wait", "send_fill", "send_wait"):
                    row[f"r{rank}_{field}"] = src.get(field, "")
            writer.writerow(row)
            fout.flush()

            tag = "WIN" if speedup and float(speedup) > 1.0 else "loss"
            print(f"  -> gpu_spmv={gpu_metrics.get('spmv','?')}  "
                  f"mpi_gpu_spmv={mg_metrics.get('spmv','?')}  "
                  f"ratio={speedup}  [{tag}]")

            if args.cleanup:
                print(f"  cleanup: removing {mtx_dir}")
                shutil.rmtree(mtx_dir)

    fout.close()
    print(f"\nWritten {out_tsv}")


if __name__ == "__main__":
    main()
