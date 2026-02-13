import os
import sys
import subprocess
import numpy as np
from scipy import sparse, io
from pathlib import Path
from parallelbar import progress_map
import shutil
from typing import List, Tuple, Optional

# =============================================================================
# Configuration & Constants
# =============================================================================

# System Architecture Configuration
# ---------------------------------
# CPU: Intel 16 Cores (8 P-Cores, 8 E-Cores)
NUM_P_CORES = 8
NUM_E_CORES = 8
TOTAL_CPU_CORES = NUM_P_CORES + NUM_E_CORES

# Weighting Ratios (Performance Modeling)
# ---------------------------------------
# Ratio of aggregate GPU throughput to aggregate CPU throughput.
# Derived from Memory Bandwidth (900 GB/s vs 80 GB/s) + Latency Penalty
GPU_TO_CPU_PERF_RATIO = 10.0

# Ratio of P-Core throughput to E-Core throughput.
# Derived from IPC and Clock Speed differences (~5.0GHz vs ~3.9GHz, Wider Decode)
P_CORE_TO_E_CORE_PERF_RATIO = 2.0

# PaToH Configuration
PATOH_BINARY_PATH = "/usr/local/bin/patoh" # ADJUST THIS PATH AS NEEDED
SEED = 42

# =============================================================================
# Helper Utilities
# =============================================================================

def check_mtx(path_mtx: Path) -> bool:
    """Validates if the Matrix Market file is suitable for processing."""
    try:
        numrows, numcols, nnz, format_type, field, symmetry = mminfo(path_mtx)
        result = True
        result &= (numrows == numcols)        # Square matrices typically for solvers
        result &= (numrows > 0)
        result &= (nnz > 0)
        result &= (format_type == "coordinate")
        result &= (field == "real")
        return result
    except Exception as e:
        print(f"Error checking {path_mtx}: {e}")
        return False

def rmrf(path: Path) -> None:
    """Recursively delete a directory and all its contents."""
    path = Path(path)
    if path.is_file() or path.is_symlink():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)

# =============================================================================
# Hierarchical Partitioner Logic
# =============================================================================

class HierarchicalPartitioner:
    def __init__(self, patoh_bin: str, temp_dir: Path):
        self.patoh_bin = patoh_bin
        self.temp_dir = temp_dir
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def _matrix_to_patoh_hypergraph(self, matrix: sparse.csr_matrix, filename: Path):
        """
        Converts a Scipy CSR matrix to PaToH hypergraph format.
        We utilize the Column-Net model:
        - Vertices (Cells) = Rows of the matrix
        - Nets (Hyperedges) = Columns of the matrix
        """
        rows, cols = matrix.shape
        # Convert to CSC for efficient column access (Nets definition)
        mat_csc = matrix.tocsc()

        with open(filename, 'w') as f:
            # Header: Base-0, #Cells, #Nets, #Pins
            f.write(f"0 {rows} {cols} {mat_csc.nnz}\n")

            # Write Net List (Pins for each column)
            # Indptr array in CSC tells us where each column starts/ends
            for j in range(cols):
                start = mat_csc.indptr[j]
                end = mat_csc.indptr[j+1]
                if start == end:
                    # Empty column (Net with no pins).
                    # PaToH might expect an empty line or handle it gracefully.
                    f.write("\n")
                else:
                    pins = mat_csc.indices[start:end]
                    # Join pins with spaces
                    line = " ".join(map(str, pins))
                    f.write(f"{line}\n")

    def _run_patoh_binary(self, hg_file: Path, k: int, imbalance: float) -> np.ndarray:
        """Invokes the PaToH binary and parses the output partition vector."""
        if not Path(self.patoh_bin).exists():
            # Fallback or error if binary not found.
            # For this report simulation, we will raise an error.
            raise FileNotFoundError(f"PaToH binary not found at {self.patoh_bin}")

        # Command arguments for PaToH (assuming standard interface)
        # patoh <hg_file> <num_parts> IB=<imbalance> SD=<seed>
        cmd =

        # Check for output file existence before running to clear old runs
        expected_out = Path(f"{hg_file}.part.{k}")
        if expected_out.exists():
            expected_out.unlink()

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"PaToH execution failed for {hg_file}") from e

        if not expected_out.exists():
            raise FileNotFoundError(f"Output file {expected_out} not generated.")

        # Parse output: One integer per line
        with open(expected_out, 'r') as f:
            lines = f.read().split()

        return np.array([int(x) for x in lines], dtype=np.int32)

    def partition(self, matrix_path: Path, output_path: Path, imbalance: int) -> int:
        """
        Main driver for the two-step partitioning.
        Returns 0 on success, non-zero on failure.
        """
        try:
            # Load Matrix
            m = io.mmread(matrix_path)
            if not sparse.isspmatrix_csr(m):
                m = m.tocsr()

            total_rows = m.shape

            # -----------------------------------------------------------------
            # STEP 1: GPU vs CPU Partitioning
            # -----------------------------------------------------------------
            # Strategy: Partition into K virtual parts.
            # GPU weight = 10, CPU weight = 1. Total = 11.
            # We partition into 11 parts. Assign 10 to GPU, 1 to CPU.

            k_step1 = int(GPU_TO_CPU_PERF_RATIO + 1) # e.g., 11
            hg_step1 = self.temp_dir / f"{matrix_path.stem}_step1.hg"

            self._matrix_to_patoh_hypergraph(m, hg_step1)

            # Using slightly higher imbalance (e.g. 5% or user defined) for top cut
            # to prioritize cut minimization over perfect load balance.
            part_vec_s1 = self._run_patoh_binary(hg_step1, k_step1, imbalance/100.0)

            # Map Partitions:
            # Parts 0 to (k-2) -> GPU (ID: -1)
            # Part (k-1)       -> CPU (ID: Temporary 99)
            cpu_virtual_id = k_step1 - 1

            is_cpu_row = (part_vec_s1 == cpu_virtual_id)
            cpu_row_indices = np.where(is_cpu_row)
            gpu_row_indices = np.where(~is_cpu_row)

            if len(cpu_row_indices) == 0:
                # Fallback: If CPU gets nothing, force minimal assignment or warn
                # For this report, we accept it might happen on tiny matrices
                pass

            # -----------------------------------------------------------------
            # STEP 2: CPU Internal Partitioning (P-Core vs E-Core)
            # -----------------------------------------------------------------
            # Extract CPU Sub-matrix
            cpu_submatrix = m[cpu_row_indices, :]

            # Strategy: Partition CPU chunk into K virtual parts for P/E balance.
            # 8 P-Cores (Weight 2) + 8 E-Cores (Weight 1) = 16 + 8 = 24 Virtual Parts.

            k_step2 = int((NUM_P_CORES * P_CORE_TO_E_CORE_PERF_RATIO) + NUM_E_CORES) # 24
            hg_step2 = self.temp_dir / f"{matrix_path.stem}_step2.hg"

            self._matrix_to_patoh_hypergraph(cpu_submatrix, hg_step2)

            part_vec_s2_raw = self._run_patoh_binary(hg_step2, k_step2, imbalance/100.0)

            # Aggregate Virtual Parts into Physical Core IDs (0 to 15)
            # P-Cores (0-7): Each takes 2 virtual parts.
            # E-Cores (8-15): Each takes 1 virtual part.

            final_cpu_mapping = np.zeros_like(part_vec_s2_raw)

            p_core_count = 0
            # Assign P-Cores (consume 2 parts each)
            # Virtual IDs 0,1 -> P0; 2,3 -> P1... 14,15 -> P7
            limit_p = NUM_P_CORES * int(P_CORE_TO_E_CORE_PERF_RATIO)

            for v_id in range(0, limit_p, 2):
                mask = (part_vec_s2_raw == v_id) | (part_vec_s2_raw == v_id+1)
                final_cpu_mapping[mask] = p_core_count
                p_core_count += 1

            # Assign E-Cores (consume 1 part each)
            # Virtual IDs 16 -> E0... 23 -> E7
            # Physical ID starts at 8
            e_core_count = 8
            for v_id in range(limit_p, k_step2):
                mask = (part_vec_s2_raw == v_id)
                final_cpu_mapping[mask] = e_core_count
                e_core_count += 1

            # -----------------------------------------------------------------
            # Merge & Write Output
            # -----------------------------------------------------------------
            # Initialize full vector with -1 (GPU)
            final_partition_vector = np.full(total_rows, -1, dtype=np.int32)

            # Fill in CPU slots
            final_partition_vector[cpu_row_indices] = final_cpu_mapping

            # Write to output path (integer per line)
            np.savetxt(output_path, final_partition_vector, fmt='%d')

            # Write Log (Simple Summary)
            log_path = output_path.with_suffix(".log")
            with open(log_path, 'w') as log:
                log.write(f"Original Rows: {total_rows}\n")
                log.write(f"GPU Rows: {len(gpu_row_indices)}\n")
                log.write(f"CPU Rows: {len(cpu_row_indices)}\n")
                # Add distribution stats here if needed

            return 0

        except Exception as e:
            # Log failure to stderr so progress_map catches it
            sys.stderr.write(f"Failed {matrix_path.name}: {e}\n")
            return 1
        finally:
            # Cleanup Temps for this matrix
            # rmrf(self.temp_dir) # Optional: Keep for debugging or delete
            pass

# =============================================================================
# Main Execution Flow
# =============================================================================

def process_matrix_task(task_tuple):
    """Wrapper function for parallelbar."""
    mtx_file, output_base, imbal = task_tuple

    # Setup thread-local or process-local partitioner
    # We use a unique temp dir per process to avoid collision
    pid = os.getpid()
    temp_dir = Path(f"./temp_patoh_{pid}")

    partitioner = HierarchicalPartitioner(PATOH_BINARY_PATH, temp_dir)

    # Define output filename structure: <Name>_i<Imbal>.part
    out_file = output_base / f"{mtx_file.stem}_i{int(imbal*100)}.part"

    result = partitioner.partition(mtx_file, out_file, imbal)

    rmrf(temp_dir)
    return result

def main():
    # 1. Setup Directories
    path_data_dir = Path("./data/matrices")
    path_output_dir = Path("./data/partitions")
    path_output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Filter Matrices
    matrices_dir = Path("/matrices")
    all_mtx = sorted(matrices_dir.glob("*.mtx"))
    valid_mtx = [m for m in all_mtx if check_mtx(m)]

    print(f"Found {len(valid_mtx)} valid matrices out of {len(all_mtx)}.")

    # 3. Define Tasks
    # We iterate over matrices and imbalance ratios.
    # We NO LONGER need the weight file iterator from the original script
    # because the weights are now derived from the code's hierarchical constants.

    imbals = (0.10, 0.01, 0.03) # 10%, 1%, 3%

    tasks =
    for mtx in valid_mtx:
        for imb in imbals:
            # Create a dedicated subdir for each matrix to keep it clean
            matrix_out_dir = path_output_dir / mtx.stem
            matrix_out_dir.mkdir(exist_ok=True)
            tasks.append((mtx, matrix_out_dir, imb))

    # 4. Execute with Progress Map
    print(f"Starting partitioning for {len(tasks)} tasks...")
    # Using cpu_count for parallelism, but note that PaToH itself is single-threaded.
    # Running multiple PaToH instances in parallel is efficient.
    results = progress_map(process_matrix_task, tasks, n_cpu=os.cpu_count())

    # 5. Error Reporting
    failures = [t for t, r in zip(tasks, results) if r!= 0]
    if failures:
        print(f"Failed {len(failures)} tasks.")
        for f in failures:
            print(f"  - {f.name} (Imbal {f})")
    else:
        print("All tasks completed successfully.")

if __name__ == "__main__":
    main()
