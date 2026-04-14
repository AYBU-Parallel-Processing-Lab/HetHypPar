#!/usr/bin/env python3
"""Initialize matrix entries in data/matrices/ by calling Octave's process_matrixi.m.

Reads a newline-separated list of matrix names from a text file, checks whether
each matrix already has all required initialization files, and runs Octave to
generate any that are missing.
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "matrices"
REQUIRED_FILES = ["B.txt", "X_init.txt", "X_target.txt"]


def is_initialized(matrix_name: str) -> bool:
    """Check if all required init files exist for a matrix."""
    in_dir = DATA_DIR / matrix_name / "in"
    return all((in_dir / f).is_file() for f in REQUIRED_FILES)


def missing_files(matrix_name: str) -> list[str]:
    """Return list of missing init files for a matrix."""
    in_dir = DATA_DIR / matrix_name / "in"
    return [f for f in REQUIRED_FILES if not (in_dir / f).is_file()]


def init_matrix(matrix_name: str, matrices_dir: str) -> bool:
    """Run Octave to initialize a matrix. Returns True on success."""
    octave_cmd = (
        f"addpath('tools/scripts'); "
        f"process_matrixi('{matrix_name}', '{matrices_dir}')"
    )
    result = subprocess.run(
        ["octave", "--eval", octave_cmd],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  ERROR: Octave failed for '{matrix_name}'", file=sys.stderr)
        if result.stderr:
            print(f"  {result.stderr.strip()}", file=sys.stderr)
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Initialize matrix entries in data/matrices/."
    )
    parser.add_argument(
        "matrix_list",
        type=Path,
        help="Text file with newline-separated matrix names.",
    )
    parser.add_argument(
        "-s",
        "--source-dir",
        default="/matrices",
        help="Directory containing raw .mtx files (default: /matrices).",
    )
    args = parser.parse_args()

    if not args.matrix_list.is_file():
        print(f"Error: '{args.matrix_list}' not found.", file=sys.stderr)
        sys.exit(1)

    names = [
        line.strip()
        for line in args.matrix_list.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    if not names:
        print("No matrix names found in the input file.")
        sys.exit(0)

    skipped = 0
    initialized = 0
    failed = 0

    for name in names:
        mtx_file = Path(args.source_dir) / f"{name}.mtx"
        if not mtx_file.is_file():
            print(f"[SKIP]  {name} — source file not found: {mtx_file}")
            skipped += 1
            continue

        if is_initialized(name):
            print(f"[SKIP]  {name} — already initialized")
            skipped += 1
            continue

        missing = missing_files(name)
        print(f"[INIT]  {name} — missing: {', '.join(missing)}")
        if init_matrix(name, args.source_dir):
            initialized += 1
            print(f"        {name} — done")
        else:
            failed += 1

    print(f"\nSummary: {initialized} initialized, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
