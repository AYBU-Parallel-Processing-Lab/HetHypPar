# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

HetHypPar — a heterogeneous hybrid parallel sparse matrix solver implementing BiCGStab (Bi-Conjugate Gradient Stabilized) for solving `Ax = b` where A is a sparse matrix. Supports four execution modes: CPU-only, GPU-only, MPI-distributed, and hybrid MPI+GPU.

## Python Environment

All Python scripts must be run via micromamba: `micromamba run -n octave python <script>`

## Build Commands

```bash
# Configure (requires MPI, Intel MKL, CUDA Toolkit, OpenMP)
cmake -S src -B build -G "Ninja"

# Build all targets
cmake --build build

# Run tests
cd build && ctest
```

## Executable Targets & Usage

Four executables, all accept: `-m <matrix> -x <x_vec> -y <b_vec> -o <output> -n <iters>`
MPI variants also accept: `-p <partition_vector>`
MPI+GPU variant also accepts: `-g <is_gpu_file>`

| Target | Entry Point | Description |
|--------|-------------|-------------|
| `bicgstab-cpu` | `src/entry/bicgstab_cpu.c` | Single-process CPU solver |
| `bicgstab-gpu` | `src/entry/bicgstab_gpu.c` | Single-process GPU solver (cuSPARSE/cuBLAS) |
| `bicgstab-mpi` | `src/entry/bicgstab_mpi.c` | MPI-distributed CPU solver |
| `bicgstab-mpi-gpu` | `src/entry/bicgstab_mpi_gpu.c` | Hybrid MPI+GPU solver |

## Solver Output Format

All four solvers print the same metrics to stdout:
`n_iters`, `spmv`, `file_read`, `relative_residual`, `everything_total` (format: `key : value`).

## Architecture

All source lives under `src/`. The static library `inner_lib` is built from `src/util/*.c` and linked by all four executables.

**Core data structures** (`src/include/hhp_common.h`):
- `COO`/`CSR`/`CSC` — sparse matrix formats. CSR and CSC are typedef'd from COO (same struct, different index semantics: CSR's `.I` has `m+1` entries, CSC's `.I` has `n+1`).
- `SHARD_CSR`/`SHARD_CSC` — distributed sparse matrices split into `loc` (local) and `shr` (shared/needs communication) parts, with `COMM` structures defining MPI send/recv patterns in CSR-like format.
- `Device_*` variants — GPU counterparts with cuSPARSE descriptors.

**Module breakdown** (`src/util/`):
- `hhp_matrix.c` — Matrix Market I/O, COO/CSR/CSC conversions, matrix distribution across MPI ranks, partition vector handling.
- `hhp_cpu.c` — Vector operations (init, clone, arithmetic, dot products) and CSR SpMV for CPU.
- `hhp_cuda.c` — Device memory management, cuBLAS vector operations, cuSPARSE SpMV.
- `hhp_wrap_bora.c` — Integration with BORA-SpMxV library for matrix distribution and PaToH partitioning.

**Dependencies** (`src/dependencies/`): `bora_spmxv` (custom SpMV), `mmio` (Matrix Market I/O), `patoh` (hypergraph partitioner).

## Testing

Uses Unity v2.6.1 (fetched via CMake FetchContent). Tests are in `src/tests/test_cpu.c`. New tests are added via `add_unit_test(test_name test_file)` in `src/tests/CMakeLists.txt`.

To run MPI solvers locally (fewer cores than ranks): `mpirun --oversubscribe -bind-to none -n <N> ./build/bicgstab-mpi ...`

## Notes

- **Shell copy-paste:** When providing terminal commands for the user to run in tmux/terminal, prefer single-line commands over multi-line with `\` continuations — backslashes often get lost when pasting.
- **Weight file parsing:** Count ranks via `.read_text().split()`, not line count — some weight files lack a trailing newline, so `wc -l` undercounts.

## Known Issues & Fixes

- **MPI send buffer use-after-free (fixed):** `internal_setup_communication` in `hhp_matrix.c` had a send buffer freed before `MPI_Isend` completed. Fixed by tracking send requests and calling `MPI_Waitall` before freeing. See commit `0814a13`.
- **"Vector of size 0" errors (open):** 876 failures across matrices like `std1_Jac2_db`, `shyy161`, `ex35`. Caused by partitions that assign zero rows to a rank. Separate from the segfault bug.

## Error Checking Macros

`CHECK_CUDA()`, `CHECK_CUSPARSE()`, `CHECK_CUBLAS()` — wrap CUDA API calls with file/line error reporting. Defined in `src/include/hhp_cuda.h`. `MPI_CHECK()` — wraps MPI calls with abort on failure. Defined in `src/include/hhp_util.h`.

## Sample Data

`data/sample320/` contains an early test matrix with vectors and partition files. Its layout is outdated (predates the current `data/matrices/` structure) but still works for unit tests.

## Adding a New Matrix to `data/matrices/`

Raw `.mtx` files live in `/matrices/` on the shared university server. The workflow below creates the input vectors and partition files needed to run the solvers on a new matrix.

### Step 1: Generate input vectors (Octave/MATLAB)

`tools/scripts/process_matrixi.m` reads a matrix from `/matrices/<name>.mtx`, computes `B = A * X_target` (where `X_target = [1, 2, ..., n]'`), and writes the files to `data/matrices/<name>/in/`.

```matlab
% From the project root
addpath('tools/scripts')
process_matrixi('cage13')   % creates data/matrices/cage13/in/{X_target.txt, X_init.txt, B.txt}
```

`process_matrix.m` does the same for all subdirectories in `data/matrices/` at once.

Alternatively, use the Python wrapper which skips already-initialized matrices:

```bash
python tools/python/init_matrices.py --matrix-list <names.txt>
```

### Step 2: Generate partition vectors

`tools/python/matrix_partition.py` runs the `patpart` binary (PaToH wrapper) to produce row partition files. It reads weight files from a directory (`-w`) and writes partition files into `data/matrices/<name>/in/part/<outdir>/`.

`patpart` CLI: `patpart <matrix_path> <npart> <weight_path> <imbal_percent> <seed> <output.part> <output.log>` (binary at `~/.local/bin/patpart`, source at `/home/bugra/Workspace/C/HetHypPar`)

```bash
python tools/python/matrix_partition.py -w data/weights -o <outdir_name>
```

This iterates over all matrices in `data/matrices/`, all weight files in `-w`, and the imbalance ratios defined in the script (default: 1%). Output per matrix: `data/matrices/<name>/in/part/<outdir>/<weight_name>_i<imbal>.part`.

### Step 3: Run the solvers

`tools/scripts/test.sh` runs all four solver variants on a matrix. It auto-detects MPI process count from the partition file.

```bash
tools/scripts/test.sh -m cage13 -g data/is_gpu/g2_2.txt -p <weight_name>_i1.part [-i 200]
```

The `-g` flag takes an `is_gpu` file from `data/is_gpu/`. Files are named `g<gpu_rank_index>_<total_ranks>.txt` — e.g., `g17_17.txt` means 17 total MPI ranks with the GPU on rank 17 (1-indexed). Each line is `0` (CPU) or `1` (GPU) for that rank. Note: some files (e.g., `g2_2.txt`) were modified manually and may not follow the naming convention.

`tools/scripts/test-mpi.sh` runs only the MPI variant.

### Expected directory layout

```
data/matrices/<name>/
├── in/
│   ├── B.txt            # RHS vector (A * X_target)
│   ├── X_init.txt       # Initial guess (all ones)
│   ├── X_target.txt     # Known solution [1..n]'
│   └── part/
│       └── <outdir>/
│           ├── <weight>_i<imbal>.part   # Partition vector
│           └── <weight>_i<imbal>.log    # Partition log
└── out/
    ├── X_seq.txt         # CPU solver output
    ├── X_gpu.txt         # GPU solver output
    ├── X_mpi.txt         # MPI solver output
    └── X_mpi_gpu.txt     # Hybrid solver output
```

## MPI Rank Pinning

`data/rankfile/` contains OpenMPI rankfiles for heterogeneous CPU+GPU runs. Files are named `<cpu_count>cpu_<gpu_count>gpu.rankfile` (e.g., `8cpu_1gpu.rankfile`). Pass to `mpirun` via `--rankfile data/rankfile/<file>` to pin GPU ranks to CUDA-capable sockets and CPU ranks to the remaining cores.

## Batch Benchmarking

`tools/python/run_benchmarks.py` runs all four solver variants from a `commands.tsv` file (generated by `tools/python/sample_mpi_gpu.py`):

```bash
python tools/python/run_benchmarks.py --outdir data/results/
```

Logs are written per-command under `--outdir`; partitioned runs (`mpi`, `mpi_gpu`) are organized by imbalance ratio, weight file, and matrix name.

### Benchmark result layout (`data/results/`)

```
data/results/
├── <matrix_name>/                                          # One dir per matrix
│   ├── cpu.{stdout,stderr}                                 # CPU solver output
│   └── gpu.{stdout,stderr}                                 # GPU solver output
└── logs/imbalance_<X>/weight_<A>_<B>/<matrix>_seed<S>/     # Partitioned runs
    ├── mpi.{stdout,stderr}                                 # MPI solver output
    └── mpi_gpu.{stdout,stderr}                             # MPI+GPU solver output
```

`tools/python/parse_benchmark_results.py` walks this directory, parses metrics from successful runs (and error reasons from stderr for failures), and writes `data/results/benchmark_summary.tsv`.

### Two benchmark pipelines

There are two independent benchmark pipelines — do not confuse them:

1. **`run_benchmarks.py` pipeline** (used for the main analysis):
   - `gen_commands.py` (in `~/Templates/results_medium/`) generates `commands.tsv` with iteration count baked in (`ITERATIONS` constant)
   - `run_benchmarks.py` executes commands from TSV → writes to `data/results/`
   - `parse_benchmark_results.py` parses `data/results/` → `data/results/benchmark_summary.tsv`
   - `commands.tsv` and partition vectors live in `~/Templates/results_medium/`

2. **`sample_mpi_gpu.py` pipeline** (self-contained, writes to `data/logs/`):
   - Runs CPU, GPU, and MPI+GPU directly with `--iterations` CLI arg
   - Writes per-matrix TSV files to `data/logs/`
   - Has hash-based deduplication (skips matrices with existing results)
   - Not consumed by `parse_benchmark_results.py`
   - Auto-detects MPI rank count from partition files (counts unique partition IDs)
   - Preferred for multi-rank benchmarks since `gen_commands.py` is hardcoded to 2 ranks
   - **Known issue:** `parse_output` raises `ValueError` on NAN/- residuals, excluding successful runs from TSV. Affects 134 matrices (including cage13, cage14, Hamrle3). Timing data is valid; only residual is unparseable.

`benchmark_summary.tsv` columns: `matrix, solver_type, imbalance, weight, seed, n_iters, spmv, file_read, relative_residual, everything_total, status, error_reason`. File is large (~2MB+); use `awk`/`cut` for analysis, not direct reads.

### Other Python tools

- `tools/python/quick_mpi_test.py` — Quick single-matrix MPI test: partitions via patpart + runs bicgstab-mpi in one command. Importable: `from quick_mpi_test import run; result = run(weight_file=Path("data/weights/cpu-p-e/w100_16.txt"))`. Returns dict with stdout, stderr, returncode, parsed metrics, part_path, nranks. All relative paths resolve to project root (safe to call from notebooks in any directory).
- `tools/python/quick_mpi_gpu_test.py` — Same as above for MPI+GPU (bicgstab-mpi-gpu). Extra `gpu_rank` param (default: last rank). Auto-creates is_gpu files in `data/is_gpu/` named `<nranks>r_gpu<rank>.txt`.
- Quick-test scripts use partition naming: `<nranks>r_i<imbal>_<hash8>.part` where hash8 = first 8 chars of SHA256 of weight file contents. Saved directly in `data/matrices/<name>/in/part/`.
- `tools/python/sample_mpi.py` — Batch benchmarking script for CPU, GPU, and MPI solvers (no MPI+GPU). Runs all matrices in `data/matrices/`, skips already-completed runs via hash-based deduplication, writes per-matrix TSV results to `data/matrices/<name>/log/`.
- `tools/python/split_partitioned_matrix.py` — Splits sparse matrices by rows according to partition vectors. Reads `.mtx` files and `.part` files, outputs per-partition sub-matrices with row/column mappings. Usage: `python tools/python/split_partitioned_matrix.py -p <part_dir> -o <output_dir>`.

### Other partitioning tools (inactive)

- `tools/python/gpu_matrix_partition.py` — Hierarchical GPU/CPU partitioner using PaToH directly. Designed to do a two-step partition (GPU vs CPU, then P-core vs E-core) but is not currently functional.
- `tools/notebook/kahypar.py` and `tools/notebook/kahypar-gpu.ipynb` — mt-KaHyPar-based partitioning. Abandoned because KaHyPar does not support heterogeneous partitioning with different block weights.

## Multi-Rank Benchmark Configurations

The number of MPI ranks is determined by the **weight file** (one line per rank) → **partition file** (one partition ID per rank) chain. Rankfile and is_gpu file must match.

| Config | Rankfile | is_gpu | Weights dir | Partition dir | Status |
|--------|----------|--------|-------------|---------------|--------|
| 2-rank (1C+1G) | `1cpu_1gpu.rankfile` | `g2_2.txt` | `weights/gpu-cpu/` | `in/part/gpu-cpu/` | Ready |
| 16-rank (16C, MPI-only) | — | — | `weights/cpu-p-e/` | `in/part/w*_16_*.part` (directly in `in/part/`) | Ready |
| 17-rank (16C+1G) | `16cpu_1gpu.rankfile` | `g17_17.txt` | `weights/hybrid-cpu-gpu/` | `in/part/hybrid-cpu-gpu/` | Ready |
| 9-rank (8C+1G) | `8cpu_1gpu.rankfile` | — | — | — | Needs setup |
