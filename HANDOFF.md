# HetHypPar Project Handoff

Snapshot of project state for transferring to another host.

## What's in this archive

- `src/` — full source (build with `cmake -S src -B build -G Ninja && cmake --build build`)
- `tools/` — Python tooling (benchmarking, parsing, plotting, matrix generation)
- `CLAUDE.md` — full project documentation (READ THIS FIRST for project context)
- `README.md`, `docs/`
- `data/is_gpu/`, `data/rankfile/`, `data/weights/` — small config files
- `data/sample/`, `data/sample320/` — small test fixtures (used by unit tests)
- `data/matrices/{bips07_1693,rma10,cage11,cage12}/` — sample matrices for smoke testing. Each includes the raw `.mtx` file (inside the matrix dir, not under `/matrices/`) and `in/` vector files (B, X_init, X_target). Partition files (`in/part/`) are NOT included — regenerate with `tools/python/matrix_partition.py` or pull from the original host.
- `data/block_sweep/` — block-diagonal sweep TSV from the most recent analysis
- `data/results/benchmark_summary.tsv.gz` — full benchmark+profile data (decompress with `gunzip`)
- `data/results/profile_boxplot_*.png` — analysis plots
- `data/results/sweep_summary.tsv` (if present) — sweep run summary
- `.git/` — full git history

## What's NOT in this archive (and why)

| Path | Size | Why excluded | How to recover |
|------|------|--------------|----------------|
| `data/matrices/<base>/` | ~10–80 GB | Generated from raw `.mtx` files | Raw matrices live at `/matrices/<name>.mtx` on the shared university server. Regenerate inputs with `tools/python/init_matrices.py` (Octave-based) or `tools/scripts/process_matrix.m`. |
| `data/matrices/<name>_<c>c_<g>g/` | ~30 GB | Block-diagonal generated matrices | Regenerate with `python tools/python/setup_n_block_diag.py <name> <mtx_path> -c N -g M` |
| `data/matrices/<name>/in/part/` | ~22 MB each | Partition vectors | Regenerate with `python tools/python/matrix_partition.py` (needs `patpart` binary) or pull pre-computed ones from `~/Templates/results_medium/logs/imbalance_*/weight_*/*_partvec.txt` |
| `data/results/logs/` | 435 MB | Per-run stdout/stderr from the 31,816-run profiled benchmark | The aggregated summary is `benchmark_summary.tsv.gz`. To regenerate raw logs, re-run `tools/python/run_benchmarks.py` |
| `data/results_backup_*` | 950 MB | Old benchmark backups (1000iter, 20iter_noprofile) | Documented in CLAUDE.md "Benchmark result backups" section. Lost if not preserved separately. |
| `data/logs*`, `data/cage13`, `data/circuit5M_dc`, `data/nv2` | ~880 MB | Stale or experimental dirs | Not needed for current work. |
| `build/` | 22 MB | Build artifacts | `cmake -S src -B build -G Ninja && cmake --build build` |

## Where we left off

### Most recent commits (head: `59576ee` "extra timing")

```
59576ee extra timing                                        ← profiling instrumentation
505b646 post cut size 0 fix                                 ← GPU-path guards for empty shr
7ebfc47 pre cut size 0 case fix                             ← internal_setup_communication fix
0814a13 fix: MPI send buffer use-after-free segfaults
```

### Uncommitted changes (to commit before/after transfer)

- `CLAUDE.md` — updated with cutsize-0 fix details, profiling section, benchmark backups, file size warnings, etc.
- `tools/python/setup_n_block_diag.py` — block-diagonal matrix generator (GPU on rank 0)
- `tools/python/sweep_block_diag.py` — sweep harness for testing block-diag configurations
- `data/is_gpu/g2_2_indep.txt` — `1\n0\n` (GPU on rank 0) for block-diag tests
- `data/block_sweep/` — sweep results

### Last analysis conclusion

The fundamental result: **mpi_gpu cannot beat single-GPU on this hardware**, regardless of partition quality.

Measured CPU-vs-GPU per-block compute ratios:
- cage11: CPU is **11.9×** slower per block → theoretical max speedup 1.08× (CPU should do 7.8% of work)
- rma10: CPU is **17.4×** slower per block → theoretical max speedup 1.06× (CPU should do 5.4% of work)
- cage12: CPU is **15.7×** slower per block → theoretical max speedup 1.06× (CPU should do 6.0% of work)

MPI Allreduce overhead (5 dot products per BiCGStab iteration × ~20 µs each) typically erases this 5-8% theoretical gain. Best ratios actually achieved:
- rma10 1c+15g: **0.954×** (5% slower than single-GPU)
- cage11 1c+9g: **0.733×**
- cage12 1c+9g: **0.697×**

See full analysis in conversation history; key data in `data/block_sweep/sweep_results.tsv` and `data/results/benchmark_summary.tsv`.

### Suggested next steps for further investigation

1. **Pipelined / merged BiCGStab variants** that reduce the number of Allreduces per iteration from 5 to 2-3. This is the highest-impact direction since MPI Allreduce overhead is the binding constraint.
2. **Non-blocking Allreduce (`MPI_Iallreduce`) with compute overlap** — let SpMV overlap with dot-product reductions.
3. **Multi-GPU MPI** — partition across multiple GPUs (one per rank). The CPU rank's contribution is too small to matter; remove it entirely and add more GPUs.
4. **Memory-bound matrix benchmarks** — find matrices where the GPU's compute advantage shrinks (e.g., very sparse, irregular access patterns) so the CPU's contribution is relatively larger.

## Quick start on the new host

```bash
# 1. Extract
unzip hethyppar.zip && cd HetHypPar_rewrite

# 2. Decompress big TSV
gunzip data/results/benchmark_summary.tsv.gz

# 3. Build
cmake -S src -B build -G Ninja
cmake --build build

# 4. Run unit tests
cd build && ctest
cd ..

# 5. Smoke-test with the included bips07_1693 (no partition needed for CPU/GPU only)
./build/bicgstab-cpu \
  -m data/matrices/bips07_1693/bips07_1693.mtx \
  -x data/matrices/bips07_1693/in/X_init.txt \
  -y data/matrices/bips07_1693/in/B.txt \
  -o /tmp/X_cpu.txt -n 50

# 6. For MPI runs you need partition files. Either:
#    a) Regenerate with `python tools/python/matrix_partition.py` (needs patpart binary)
#    b) Copy specific partitions from the original host (~/Templates/results_medium/)
#    c) Block-diagonal sweeps generate their own — see tools/python/sweep_block_diag.py

# 7. For matrices not in this archive: raw .mtx files live at /matrices/ on the
#    original university server. Regenerate input vectors with:
micromamba run -n octave python tools/python/init_matrices.py --matrix-list <names.txt>
```

## Things to double-check on the new host

- **MPI implementation**: original was OpenMPI; `mpirun -n 2` should "just work" without `--oversubscribe` for 2 ranks.
- **CUDA toolkit**: cuSPARSE/cuBLAS required; tested on a single NVIDIA GPU.
- **Intel MKL**: linked into the CPU SpMV path (see `src/util/hhp_cpu.c`).
- **`micromamba` env `octave`**: provides Python + pandas + numpy + matplotlib + octave for input-vector generation.
- **PaToH binary**: `~/.local/bin/patpart` for partitioning (source at `/home/bugra/Workspace/C/HetHypPar` on original host).
- **Shared matrix server**: raw `.mtx` files at `/matrices/` are NOT in this archive.

## Files to read first on the new host

1. `HANDOFF.md` (this file) — what was done, what was excluded
2. `CLAUDE.md` — full project documentation, known issues, build/run instructions
3. `data/block_sweep/sweep_results.tsv` — latest analysis input
4. `docs/` — earlier analysis writeups
