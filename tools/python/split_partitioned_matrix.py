#!/usr/bin/env python3
"""
Batch split sparse matrices based on partition vectors.

Processes all matrices in /matrices and all partition files for each matrix,
writing split matrices to separate output directories.
"""

import argparse
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.io import mmread, mmwrite
from scipy.sparse import csr_matrix


def read_partition_vector(filepath: Path) -> np.ndarray:
    """Read partition vector from file (one integer per line)."""
    return np.loadtxt(filepath, dtype=np.int32)


def split_matrix_by_rows(
    matrix: csr_matrix, partvec: np.ndarray
) -> Dict[int, Tuple[csr_matrix, np.ndarray, np.ndarray]]:
    """
    Split a CSR matrix by rows according to partition vector.

    Args:
        matrix: Input CSR matrix (m x n)
        partvec: Partition vector of length n (which partition each row belongs to)

    Returns:
        Dictionary mapping partition_id -> (local_matrix, row_map, col_map)
        where:
        - local_matrix: The local partition in CSR format with renumbered indices
        - row_map: Global row indices for this partition's rows
        - col_map: Global column indices for this partition's columns
    """
    if matrix.shape[0] != len(partvec):
        raise ValueError(
            f"Matrix has {matrix.shape[0]} rows but partition vector has {len(partvec)} elements"
        )

    n_parts = np.max(partvec) + 1
    results = {}

    for part_id in range(n_parts):
        # Find which rows belong to this partition
        row_mask = partvec == part_id
        row_indices = np.where(row_mask)[0]

        if len(row_indices) == 0:
            print(f"    Warning: Partition {part_id} has no rows")
            continue

        # Extract rows for this partition
        part_rows = matrix[row_indices, :]

        # Find which columns are actually used in this partition
        col_indices_used = np.unique(part_rows.nonzero()[1])

        # Extract only the used columns to create compact local matrix
        part_local = part_rows[:, col_indices_used]

        # Convert back to CSR format
        part_local_csr = csr_matrix(part_local)

        results[part_id] = (part_local_csr, row_indices, col_indices_used)

    return results


def write_partition_matrices(
    partitions: Dict[int, Tuple[csr_matrix, np.ndarray, np.ndarray]],
    output_dir: Path,
    base_name: str,
    write_maps: bool = True,
):
    """
    Write partition matrices and optional mapping files.

    Args:
        partitions: Dictionary from split_matrix_by_rows
        output_dir: Directory to write files
        base_name: Base name for output files
        write_maps: Whether to write row/column mapping files
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for part_id, (local_matrix, row_map, col_map) in partitions.items():
        # Write matrix file
        matrix_file = output_dir / f"{base_name}_part{part_id}.mtx"
        mmwrite(matrix_file, local_matrix)

        if write_maps:
            # Write row mapping (local_row_idx -> global_row_idx)
            row_map_file = output_dir / f"{base_name}_part{part_id}_rowmap.txt"
            np.savetxt(row_map_file, row_map, fmt="%d")

            # Write column mapping (local_col_idx -> global_col_idx)
            col_map_file = output_dir / f"{base_name}_part{part_id}_colmap.txt"
            np.savetxt(col_map_file, col_map, fmt="%d")


def process_single_partition(
    matrix_path: Path,
    partition_path: Path,
    output_base_dir: Path,
    write_maps: bool,
) -> bool:
    """
    Process a single matrix with a single partition file.

    Returns:
        True if successful, False otherwise
    """
    matrix_name = matrix_path.stem
    partition_name = partition_path.stem

    # Determine final output directory
    final_output_dir = output_base_dir / matrix_name / partition_name

    # Skip if output already exists
    if final_output_dir.exists():
        print(f"  Skipping {matrix_name}/{partition_name} (output already exists)")
        return True

    print(f"  Processing {matrix_name}/{partition_name}")

    try:
        # Read matrix
        matrix = mmread(matrix_path)
        if not isinstance(matrix, csr_matrix):
            matrix = csr_matrix(matrix)

        # Read partition vector
        partvec = read_partition_vector(partition_path)

        # Split matrix
        partitions = split_matrix_by_rows(matrix, partvec)

        # Create temporary directory in CWD
        with tempfile.TemporaryDirectory(dir=Path.cwd(), prefix=f"tmp_split_{matrix_name}_{partition_name}_") as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Write to temporary directory
            write_partition_matrices(partitions, tmp_path, matrix_name, write_maps)

            # Ensure parent directory exists
            final_output_dir.parent.mkdir(parents=True, exist_ok=True)

            # Atomic move from temp to final location
            shutil.move(str(tmp_path), str(final_output_dir))

        print(f"    Success: {len(partitions)} partitions written to {final_output_dir}")
        return True

    except Exception as e:
        print(f"    ERROR processing {matrix_name}/{partition_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Batch split sparse matrices based on partition vectors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all matrices with partitions in specific directory
  %(prog)s -p my_partitions -o ./split_output

  # Without mapping files
  %(prog)s -p my_partitions -o ./split_output --no-maps
        """,
    )

    parser.add_argument(
        "-p",
        "--part-dir",
        type=str,
        required=True,
        help="Partition directory name under data/matrices/{matrix}/in/part/",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        required=True,
        help="Base output directory for split matrices",
    )
    parser.add_argument(
        "--no-maps",
        action="store_true",
        help="Do not write row/column mapping files",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data/matrices"),
        help="Base data directory (default: ./data/matrices)",
    )
    parser.add_argument(
        "--matrices-dir",
        type=Path,
        default=Path("/matrices"),
        help="Directory containing .mtx files (default: /matrices)",
    )

    args = parser.parse_args()

    # Find all matrix files
    matrices_dir = args.matrices_dir
    if not matrices_dir.exists():
        parser.error(f"Matrices directory not found: {matrices_dir}")

    matrix_files = sorted(matrices_dir.glob("*.mtx"))
    if not matrix_files:
        parser.error(f"No .mtx files found in {matrices_dir}")

    print(f"Found {len(matrix_files)} matrix files in {matrices_dir}")
    print(f"Output will be written to: {args.out_dir}")
    print()

    total_tasks = 0
    successful_tasks = 0
    skipped_tasks = 0

    # Process each matrix
    for matrix_path in matrix_files:
        matrix_name = matrix_path.stem
        print(f"Processing matrix: {matrix_name}")

        # Find partition directory for this matrix
        partition_dir = args.data_dir / matrix_name / "in" / "part" / args.part_dir

        if not partition_dir.exists():
            print(f"  No partition directory found at {partition_dir}")
            print()
            continue

        # Find all partition files
        partition_files = sorted(partition_dir.glob("*.part"))

        if not partition_files:
            print(f"  No .part files found in {partition_dir}")
            print()
            continue

        print(f"  Found {len(partition_files)} partition files")

        # Process each partition file
        for partition_path in partition_files:
            total_tasks += 1

            # Check if we're skipping
            matrix_name_check = matrix_path.stem
            partition_name_check = partition_path.stem
            final_output_dir_check = args.out_dir / matrix_name_check / partition_name_check

            if final_output_dir_check.exists():
                skipped_tasks += 1

            success = process_single_partition(
                matrix_path,
                partition_path,
                args.out_dir,
                not args.no_maps,
            )

            if success and not final_output_dir_check.exists():
                # Only count as successful if we actually did work
                successful_tasks += 1

        print()

    print("=" * 60)
    print(f"Batch processing complete!")
    print(f"Total tasks: {total_tasks}")
    print(f"Skipped (already exist): {skipped_tasks}")
    print(f"Successfully processed: {successful_tasks}")
    print(f"Failed: {total_tasks - skipped_tasks - successful_tasks}")


if __name__ == "__main__":
    main()
