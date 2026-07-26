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
# RTX 3070 = sm_86; set native arch or the default JITs inside the timed loop:
#   cmake -S src -B build -G Ninja -DCMAKE_CUDA_ARCHITECTURES=86

# Build all targets
cmake --build build

# Build one target (fast iteration when editing a single solver)
cmake --build build --target bicgstab-hybrid-async-dp

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
| `spmv-cpu` | `src/entry/spmv_cpu.c` | SpMV-only CPU benchmark (no solver) |
| `spmv-gpu` | `src/entry/spmv_gpu.c` | SpMV-only GPU benchmark (no solver) |

### Device-pointer / pipelined / hybrid variants (single-process CPU+GPU, no MPI)

Beyond the 4 base solvers, the build has dp (cuBLAS device-pointer-mode dots), pipelined (Cools & Vanroose 2017), and hybrid (CPU+GPU split) families: `bicgstab-gpu-dp`, `bicgstab-gpu-pipelined`, `bicgstab-hybrid-async-dp` (SpMV split, dots stay on GPU), `bicgstab-hybrid-dist-dp` (fully distributed), `bicgstab-hybrid-dist-pipelined`. Hybrid variants take `-p <part>` and `-g <is_gpu>` like the MPI solvers.

- **Key perf finding:** distribute the SpMV but keep dots GPU-resident. `hybrid-async-dp` ≈ 1.3× over `bicgstab-gpu`; distributing the dots too (`dist-dp` ≈ 1.1×) costs more CPU↔GPU sync than it saves; pipelining loses single-node (<1.0×) — it only pays multi-node. See `docs/dot-product-profiling.md`, `docs/ca-sgd-relation.md`.
- **Env vars** (dp/pipelined solvers): `HHP_REPLACE=k` = residual-replacement period (recompute true residual every k iters; needed for pipelined accuracy). `HHP_PROF` = unified profiling (see below).
- **Hybrid OMP threads:** tune `OMP_NUM_THREADS` to 1–4 — the default (24) is pathological for the small CPU row-slices.

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
- **Vector file format:** `vector_read` reads raw whitespace-separated floats with NO header line. Generators must be header-less (`process_matrixi.m`, `prepare_spmv_input.m` use bare `dlmwrite` — correct). Note: `setup_n_block_diag.py` writes a `<rows> 1` header that `vector_read` misreads as data (latent bug; fine for timing-only tests where residual is ignored).
- **Locale / benchmark parsing:** solver stdout prints decimals with the locale separator (e.g. `spmv : 0,947686` with a comma), which breaks `awk` numeric compares and bash `printf`. Always run benchmark loops with `LC_ALL=C` so decimals are dots.
- **octave env extras:** `pyarrow` (zstd parquet), `matplotlib`, and `cairosvg` are pip-installed into the `octave` micromamba env for the benchmark analysis/plot scripts (`cairosvg.svg2png(...)` to preview docs/*.svg). `micromamba` is a binary on PATH even in detached tmux.
- **Octave `mmread` fails on `integer`-field `.mtx`** (e.g. `atmosmodm`) — `process_matrixi.m` errors out. Select `real`-field matrices for benchmarks.
- **Commits:** do NOT add any AI/Claude attribution or `Co-Authored-By` trailer to commit messages (project convention).

## Known Issues & Fixes

- **MPI send buffer use-after-free (fixed):** `internal_setup_communication` in `hhp_matrix.c` had a send buffer freed before `MPI_Isend` completed. Fixed by tracking send requests and calling `MPI_Waitall` before freeing. See commit `0814a13`.
- **"Vector of size 0" from cutsize-0 partitions (fixed):** `internal_setup_communication` called `ivector_init(result->shr.n)` unconditionally — crashes when `shr.n == 0` (no shared columns). Fixed by guarding the allocation. GPU path also needed guards around `device_csc_create`/`device_vector_init`/`device_buffer_spmv_create` for empty shared matrices (`bicgstab_mpi_gpu.c`, `hhp_cuda.c`).
- **"Vector of size 0" from zero-row partitions (open):** Remaining failures across matrices like `std1_Jac2_db`, `shyy161`, `ex35`. Caused by partitions that assign zero rows to a rank. The solver should either reject these partitions at startup or handle empty ranks gracefully.
- **`patpart` segfaults on matrices >~2.1M rows (workaround in place):** `CalcPartVec` in the patpart source (`~/Workspace/C/HetHypPar/src/partition.c`) uses a stack VLA `int nweights[n]` that overflows the 8 MB stack. Raise the child stack before calling it — `prep_big_matrix.py` sets `RLIMIT_STACK`=1 GiB; manually `ulimit -s 262144`.

## SpMV-Only Benchmarking

`spmv-cpu` / `spmv-gpu` isolate matrix-vector multiply from BiCGStab (no dot products / MPI Allreduce). Args: `-m <matrix> -x <input_x> -o <output_y> -n <repetitions>`. They warm up once, time `n` repeated `Y=A*X`, and report `spmv_per_iter` and `max_abs_err_vs_1ton` (correctness check).

Prepare inputs with `tools/scripts/prepare_spmv_input.m`: solves `A*x=[1..n]` so the SpMV output is trivially `[1,2,...,n]`. Defaults to GMRES+ILU(0) (`'iterative'`); pass `'direct'` for small matrices only — direct `A\b` OOMs above ~400K rows.

```bash
micromamba run -n octave octave --no-gui --eval "addpath('tools/scripts'); prepare_spmv_input('/matrices/cage14.mtx', 'data/matrices/cage14/in')"
./build/spmv-gpu -m /matrices/cage14.mtx -x data/matrices/cage14/in/X_spmv.txt -o /tmp/Y.txt -n 500
```

- Pure-SpMV GPU/CPU speedup scales with matrix nnz: ~5x at 0.5M nnz, ~12x at 27M nnz.
- `spmv-mpi` (one rank/core) + `data/rankfile/8P_8E.rankfile` (P=cores 0-7, E=8-15) test heterogeneous P/E partitioning via `data/weights/cpu-p-e/w*_16` weights. Pin with `mpirun --rankfile ...` (this OpenMPI wants `--rankfile`, not `--map-by rankfile:file=`).
- Findings written up in `docs/spmv-bandwidth-analysis.md`: SpMV is memory-bandwidth-bound, best CPU (OpenMP 8T) is ~6.7x slower than GPU, OpenMP≈MKL>MPI on a single node, P/E partitioning tops out ~1.31x vs 1 core.

## Profiling

### Unified `HHP_PROF` profiling (`hhp_prof.h`/`.c`)

All 12 BiCGStab solvers share one optional profiler (`src/util/hhp_prof.c`, built into `inner_lib`). Off by default — every `prof_tick`/`prof_lap` early-returns, so a normal run has zero overhead. Env `HHP_PROF`:
- unset/`0` → off
- `1` → per-section wall-time summary to **stderr** (`spmv`/`dot`/`vecops`/`comm`/`allreduce`/`sync`, with total_s, µs/iter, %loop, plus a one-off `preprocess` time)
- `2` → also dumps a per-iteration TSV to `data/prof/<solver>__<matrix>.tsv` (one row per iteration, one column per section)

Device-agnostic: GPU solvers pass a sync callback (`cudaStreamSynchronize(compute_s)` for stream solvers, `cudaDeviceSynchronize()` for default-stream ones); CPU/MPI pass `NULL`. Sections are attributed via `prof_tick` (arm) + `prof_lap(&P, PF_*)` (attribute+re-arm) at region boundaries, `prof_iter_end` per iteration, `prof_report(&P, "<solver>", matrix_path)` at the end. **Profiling perturbs the absolute loop time** (it inserts syncs) — take the headline `spmv`/speedup number from a clean `HHP_PROF`-unset run; the bucket *ratios* stay faithful.

- **MPI solvers** (`mpi`, `mpi-gpu`, `mpi-gpu-mt`, `mpi-gpu-pipelined`): report per-rank with a rank-suffixed label (`mpi-r0`, `mpi-gpu-r1`, …) so per-iteration TSVs don't collide, and their existing `PROFILE_ITER`/`PROFILE_ACCUM` stdout lines are **untouched** (still consumed by `parse_benchmark_results.py`). For distributed dots the `dot` bucket includes each `MPI_Allreduce`; `mpi-gpu-pipelined` has an explicit `PF_ALLREDUCE` bucket around its `MPI_Iallreduce` regions.
- **`gpu` and `hybrid-async`** additionally keep their legacy `PROFILE_*` stdout lines (now driven from the same `Prof` accumulators) — the old `HHP_PROFILE` env name is gone; use `HHP_PROF`.
- **`hybrid-dist-dp`/`hybrid-dist-pipelined`**: the `sync` bucket (`PF_SYNC`) isolates the mapped-host spin-wait CPU↔GPU rendezvous — the extra cost of distributing the reduction (tiny for pipelined, ~10% for `dist-dp`).
- **Profiler var naming:** several solvers declare a `Device_Vector P`, so the `Prof` struct must be named `PROF` (gpu-dp, hybrid-async-dp) or file-static `g_P` — never `P`, or it shadows the vector and the build fails cryptically.
- **`bicgstab_gpu.c` has TWO BiCGStab implementations:** an unused reference `gpu_BiCGStab()` function (never called) and the *actual* solver as an inline loop in `main()`. Instrument/edit the loop in `main()`, not the dead function.

### Legacy MPI SpMV profiling structs

`SpMV_Profile` and `Iter_Profile` structs defined in `hhp_common.h`. Both MPI SpMV functions (`hhp_cpu.c`, `hhp_cuda.c`) accept an optional `SpMV_Profile *prof` parameter (NULL skips profiling). GPU path uses `cudaDeviceSynchronize()` before timing boundaries only when `prof != NULL`.

Solver output includes `PROFILE_ITER` (per-rank per-iteration) and `PROFILE_ACCUM` (per-rank totals) lines. `parse_benchmark_results.py` extracts these into per-rank accumulated columns (`r0_spmv`, `r1_comm_wait`, etc.) and a `profile_iterations` JSON column.

- **Iteration 0 is a startup outlier** -- the first iteration's `vecops` can be 100-1000x larger than subsequent iterations due to MPI/CUDA initialization (one rank blocks in `MPI_Allreduce` while the other does first-time CUDA kernel launches). Analysis should exclude iter 0.
- **The top-level `spmv` column is rank 0's loop time only**, not a max across ranks. For the slower rank's perspective, use per-rank profile fields.
- `tools/python/plot_profile_boxplots.py` generates per-component box plots normalized to single-GPU per-iteration baseline.
- **Don't run `nsys` on the spin-wait solvers** (dp/dist/pipelined use a mapped-host `while(*flag<seq){}` busy-loop) — it hangs and piles up stuck processes. Use `HHP_PROF=1` for their section timings instead.
- **The top-level `spmv` line is the full solver LOOP time, not the isolated SpMV.** For SpMV-isolated timing (e.g. partition-quality comparison) use `spmv-hybrid-async` (reports `hybrid_per_iter`); a ones-vector (`X_init.txt`) suffices since only timing matters (its `max_abs_err` is then meaningless).

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
   - 2-rank partition vectors: `~/Templates/results_medium/logs/imbalance_<X>/weight_<W>/<matrix>_seed<S>_partvec.txt`
   - Partition metadata (cutsize, partition sizes): `~/Templates/results_medium/master_summary_clean.csv`

2. **`sample_mpi_gpu.py` pipeline** (self-contained, writes to `data/logs/`):
   - Runs CPU, GPU, and MPI+GPU directly with `--iterations` CLI arg
   - Writes per-matrix TSV files to `data/logs/`
   - Has hash-based deduplication (skips matrices with existing results)
   - Not consumed by `parse_benchmark_results.py`
   - Auto-detects MPI rank count from partition files (counts unique partition IDs)
   - Preferred for multi-rank benchmarks since `gen_commands.py` is hardcoded to 2 ranks
   - **Known issue:** `parse_output` raises `ValueError` on NAN/- residuals, excluding successful runs from TSV. Affects 134 matrices (including cage13, cage14, Hamrle3). Timing data is valid; only residual is unparseable.

`benchmark_summary.tsv` columns: `matrix, solver_type, imbalance, weight, seed, n_iters, spmv, file_read, relative_residual, everything_total, status, error_reason`, plus per-rank profile columns (`r0_spmv`, `r0_vecops`, `r0_send_fill`, `r0_local_spmv`, `r0_comm_wait`, `r0_shared_spmv`, `r0_send_wait`, same for `r1_`), and `profile_iterations` (JSON). File is large; use `awk`/`cut` for analysis, not direct reads.

- File is ~275 MB with profiling enabled -- load with `pd.read_csv(..., sep='\t', low_memory=False)` and `usecols=[...]` for selective columns.
- Joining with `master_summary_clean.csv` requires explicit `.astype(int)` on `imbalance`/`seed` columns on both sides (TSV stores them as float).

### Benchmark result backups (`data/`)

- `results_backup_1000iter` -- 1000-iteration run, pre-cutsize-0-fix (has segfaults/vector-size-0 errors)
- `results_backup_20iter_noprofile` -- 20-iteration run, post-cutsize-0-fix, no profiling instrumentation

### Other Python tools

- `tools/python/quick_mpi_test.py` — Quick single-matrix MPI test: partitions via patpart + runs bicgstab-mpi in one command. Importable: `from quick_mpi_test import run; result = run(weight_file=Path("data/weights/cpu-p-e/w100_16.txt"))`. Returns dict with stdout, stderr, returncode, parsed metrics, part_path, nranks. All relative paths resolve to project root (safe to call from notebooks in any directory).
- `tools/python/quick_mpi_gpu_test.py` — Same as above for MPI+GPU (bicgstab-mpi-gpu). Extra `gpu_rank` param (default: last rank). Auto-creates is_gpu files in `data/is_gpu/` named `<nranks>r_gpu<rank>.txt`.
- Quick-test scripts use partition naming: `<nranks>r_i<imbal>_<hash8>.part` where hash8 = first 8 chars of SHA256 of weight file contents. Saved directly in `data/matrices/<name>/in/part/`.
- `tools/python/sample_mpi.py` — Batch benchmarking script for CPU, GPU, and MPI solvers (no MPI+GPU). Runs all matrices in `data/matrices/`, skips already-completed runs via hash-based deduplication, writes per-matrix TSV results to `data/matrices/<name>/log/`.
- `tools/python/split_partitioned_matrix.py` — Splits sparse matrices by rows according to partition vectors. Reads `.mtx` files and `.part` files, outputs per-partition sub-matrices with row/column mappings. Usage: `python tools/python/split_partitioned_matrix.py -p <part_dir> -o <output_dir>`.

### Other partitioning tools (inactive)

- `tools/python/gpu_matrix_partition.py` — Hierarchical GPU/CPU partitioner using PaToH directly. Designed to do a two-step partition (GPU vs CPU, then P-core vs E-core) but is not currently functional.
- `tools/notebook/kahypar.py` and `tools/notebook/kahypar-gpu.ipynb` — mt-KaHyPar-based partitioning. Abandoned because KaHyPar does not support heterogeneous partitioning with different block weights.

## Large-Matrix Hybrid Benchmark Suite

Pipeline for the GPU+CPU hybrid study (50+ SuiteSparse matrices). Run long sweeps under **tmux** (survives SSH/VPN drops), output `tee`'d to a logfile.

- `survey_matrices.py`/`select_matrices.py` — pool survey (`data/matrix_survey.tsv`) + stratified selection.
- `prep_big_matrix.py [--patoh-only]` — vectors + **2-rank** PaToH (`in/part/gpu-cpu/`) + naive nnz-balance (`in/part/gpu-cpu-naive/`, via `gen_naive_partition.py`, the cut-blind baseline). Both compared at the same weight.
- `big_benchmark_sweep.py` (full solver; `WEIGHTS`/`SOLVERS`/`PARTSRCS`/`OUT` env) and `spmv_patoh_sweep.py` (isolated SpMV) → TSVs.
- `results_to_parquet.py` (zstd parquet; `PARQ`/`SUMMARY_OUT` env on analysis scripts), `big_benchmark_summary.py`, `perf_profile.py` (Dolan-Moré profiles), `weight_runtime_plot.py`, `matrix_features.py`.
- **Findings** (`docs/big-benchmark-findings.md`): `hybrid-async-dp` wins most, `gpu-dp` (no CPU) wins ~1/3; **matrix density (nnz/row) predicts if the CPU split helps** (threshold ~8-10, Spearman 0.91); **PaToH ≈ naive** — not worth it for the 2-way split.

## Running on TRUBA (multi-node HPC)

`docs/truba-hpc-guide.md` — connecting, storage/quota rules, module system, queue selection (`orfoz` for CPU-only MPI scaling, `akya-cuda` for multi-node MPI+GPU hybrid), and SLURM templates (`tools/slurm/`). Local `/matrices/` tops out at 111M nnz (finishes in ~1s) — too small to exercise real multi-node communication cost; the guide identifies larger matrices (up to 6.7B nnz) pulled directly from `sparse.tamu.edu` via `tools/scripts/fetch_suitesparse_matrix.sh`.

## Multi-Rank Benchmark Configurations

The number of MPI ranks is determined by the **weight file** (one line per rank) → **partition file** (one partition ID per rank) chain. Rankfile and is_gpu file must match.

| Config | Rankfile | is_gpu | Weights dir | Partition dir | Status |
|--------|----------|--------|-------------|---------------|--------|
| 2-rank (1C+1G) | `1cpu_1gpu.rankfile` | `g2_2.txt` | `weights/gpu-cpu/` | `in/part/gpu-cpu/` | Ready |
| 16-rank (16C, MPI-only) | — | — | `weights/cpu-p-e/` | `in/part/w*_16_*.part` (directly in `in/part/`) | Ready |
| 17-rank (16C+1G) | `16cpu_1gpu.rankfile` | `g17_17.txt` | `weights/hybrid-cpu-gpu/` | `in/part/hybrid-cpu-gpu/` | Ready |
| 9-rank (8C+1G) | `8cpu_1gpu.rankfile` | — | — | — | Needs setup |
