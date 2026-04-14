#!/usr/bin/env python3
"""
Run BiCGSTAB benchmark commands from commands.tsv.

Reads : commands.tsv
Writes: stdout/stderr logs per command into --outdir

Log layout:
  Partitioned (mpi, mpi_gpu):
    ${outdir}/logs/imbalance_X/weight_A_B/matrix_seedS/mpi_gpu.{stdout,stderr}
  Unpartitioned (cpu, gpu) — deduplicated per matrix:
    ${outdir}/${matrix}/cpu.{stdout,stderr}
"""

import argparse
import os
import re
import subprocess
import sys

import pandas as pd

COMMAND_TYPES = ["cpu", "gpu", "mpi", "mpi_gpu"]
COL_FOR_TYPE = {
    "cpu":     "cmd_cpu",
    "gpu":     "cmd_gpu",
    "mpi":     "cmd_mpi",
    "mpi_gpu": "cmd_mpi_gpu",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Run BiCGSTAB benchmarks from commands.tsv"
    )
    p.add_argument(
        "--parts-basedir", required=True,
        help="Absolute path to the results_medium directory "
             "(substituted for ${parts_basedir} in commands)",
    )
    p.add_argument(
        "--outdir", required=True,
        help="Root directory for benchmark logs",
    )
    p.add_argument(
        "--commands-tsv", default="commands.tsv",
        help="Path to commands.tsv (default: commands.tsv)",
    )
    p.add_argument(
        "--types", default=None,
        help="Comma-separated command types to run "
             f"(default: all — {','.join(COMMAND_TYPES)})",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing",
    )
    return p.parse_args()


def log_path_partitioned(outdir, partvec_path, cmd_type):
    """Log path for partition-dependent commands (mpi, mpi_gpu).

    partvec_path: logs/imbalance_3/weight_90_10/g7jac050sc_seed-1_partvec.txt
    result:       ${outdir}/logs/imbalance_3/weight_90_10/g7jac050sc_seed-1/mpi_gpu.stdout
    """
    # Strip _partvec.txt suffix to get the directory
    base = re.sub(r"_partvec\.txt$", "", partvec_path)
    return os.path.join(outdir, base, f"{cmd_type}.stdout")


def log_path_unpartitioned(outdir, matrix, cmd_type):
    """Log path for partition-independent commands (cpu, gpu).

    result: ${outdir}/${matrix}/cpu.stdout
    """
    return os.path.join(outdir, matrix, f"{cmd_type}.stdout")


def extract_matrix_name(partvec_path):
    """Extract matrix name from partvec_path.

    partvec_path: logs/imbalance_3/weight_90_10/g7jac050sc_seed-1_partvec.txt
    result:       g7jac050sc
    """
    filename = os.path.basename(partvec_path)  # g7jac050sc_seed-1_partvec.txt
    match = re.match(r"^(.+?)_seed", filename)
    if not match:
        raise ValueError(f"Cannot extract matrix name from {partvec_path}")
    return match.group(1)


def run_command(cmd, stdout_path, dry_run=False):
    """Run a shell command, writing stdout and stderr to separate files.

    Returns True if the command was executed, False if skipped.
    """
    stderr_path = re.sub(r"\.stdout$", ".stderr", stdout_path)

    if os.path.exists(stdout_path):
        return False

    if dry_run:
        print(f"  [DRY RUN] {cmd}")
        print(f"    -> {stdout_path}")
        return True

    os.makedirs(os.path.dirname(stdout_path), exist_ok=True)

    with open(stdout_path, "w") as fout, open(stderr_path, "w") as ferr:
        result = subprocess.run(
            cmd, shell=True, stdout=fout, stderr=ferr,
        )

    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode}): {cmd}", file=sys.stderr)
        print(f"    stderr: {stderr_path}", file=sys.stderr)
    return True


def main():
    args = parse_args()
    parts_basedir = args.parts_basedir.rstrip("/")
    outdir = args.outdir.rstrip("/")

    types_to_run = COMMAND_TYPES
    if args.types:
        types_to_run = [t.strip() for t in args.types.split(",")]
        invalid = set(types_to_run) - set(COMMAND_TYPES)
        if invalid:
            print(f"Error: unknown command types: {invalid}", file=sys.stderr)
            sys.exit(1)

    df = pd.read_csv(args.commands_tsv, sep="\t")
    print(f"Loaded {len(df)} rows from {args.commands_tsv}")

    # Track which matrices already ran cpu/gpu to deduplicate
    completed_unpartitioned = set()  # (matrix, cmd_type)

    total = 0
    skipped = 0
    executed = 0
    failed = 0

    for _, row in df.iterrows():
        partvec_path = row["partvec_path"]
        matrix = extract_matrix_name(partvec_path)

        for cmd_type in types_to_run:
            total += 1
            col = COL_FOR_TYPE[cmd_type]
            cmd = row[col]

            if cmd_type in ("cpu", "gpu"):
                # Deduplicate: same command for all weight/imbalance/seed combos
                if (matrix, cmd_type) in completed_unpartitioned:
                    skipped += 1
                    continue
                completed_unpartitioned.add((matrix, cmd_type))
                stdout_path = log_path_unpartitioned(outdir, matrix, cmd_type)
            else:
                # Substitute ${parts_basedir}
                cmd = cmd.replace("${parts_basedir}", parts_basedir)
                stdout_path = log_path_partitioned(outdir, partvec_path, cmd_type)

            if os.path.exists(stdout_path):
                skipped += 1
                continue

            print(f"[{executed + 1}] {cmd_type} | {matrix} -> {stdout_path}")
            was_run = run_command(cmd, stdout_path, dry_run=args.dry_run)
            if was_run:
                executed += 1

    print(f"\nDone. total={total}  executed={executed}  skipped={skipped}")


if __name__ == "__main__":
    main()
