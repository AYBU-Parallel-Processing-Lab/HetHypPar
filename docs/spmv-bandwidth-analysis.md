# SpMV Bandwidth Analysis: Why CPU Can't Catch the GPU

**Date:** 2026-05-24
**Scope:** Isolated sparse matrix-vector multiply (SpMV), no BiCGStab solver, no MPI dot-product Allreduces.
**Hardware:** Intel i7-13700F (8 P-cores @ up to 5.2 GHz + 8 E-cores @ 4.1 GHz, 16 physical cores / 24 threads, dual-channel DDR), NVIDIA RTX 3070 (8 GB GDDR6).
**Goal:** Determine the best achievable CPU SpMV throughput and how it compares to the GPU, to explain why heterogeneous CPU+GPU and CPU-only P/E partitioning cannot beat a single GPU.

## Motivation

Earlier BiCGStab benchmarks showed `mpi_gpu` (hybrid CPU+GPU) never beats single-GPU. To isolate the cause from solver overhead (dot products require `MPI_Allreduce` every iteration), we built SpMV-only entry points that time pure `Y = A*X` with no reductions:

| Target | Description |
|--------|-------------|
| `spmv-cpu` | Single-threaded CSR SpMV (`CSR_spmxv_seq`) |
| `spmv-cpu-mt` | OpenMP CSR SpMV (`CSR_spmxv_omp`, `guided` schedule), `OMP_NUM_THREADS` |
| `spmv-cpu-mkl` | MKL inspector-executor (`mkl_sparse_d_mv`), `MKL_NUM_THREADS` |
| `spmv-mpi` | Distributed CSR SpMV across ranks (one rank/core), partition-driven |
| `spmv-gpu` | cuSPARSE CSR SpMV |

Inputs are prepared with `tools/scripts/prepare_spmv_input.m`, which solves `A*x = [1..n]` (GMRES+ILU(0)) so the SpMV output is the easily-verified vector `[1, 2, ..., n]`.

## Headline Result

On **cage14** (1,505,785 rows, 27,130,349 nnz, ~18 nnz/row), 500 SpMV repetitions, iteration-0 warm-up excluded:

| Method | Best config | ms/iter | vs 1 core | vs GPU |
|--------|-------------|--------:|----------:|-------:|
| GPU (cuSPARSE) | — | **1.17** | 12.1× | 1.0× |
| OpenMP | 8 threads | 7.80 | 1.81× | 6.7× slower |
| MKL | 8 threads | 8.11 | 1.74× | 6.9× slower |
| OpenMP | 16 threads | 8.31 | 1.70× | 7.1× slower |
| MKL | 16 threads | 9.79 | 1.44× | 8.4× slower |
| MPI (P/E pinned) | 16 ranks, 250:100 | 10.80 | 1.31× | 9.2× slower |
| CPU | 1 core | 14.14 | 1.00× | 12.1× slower |

**The GPU is ~6.7× faster than the best CPU configuration.**

## Three Conclusions

### 1. SpMV is memory-bandwidth-bound; bandwidth is the wall

Every CPU method peaks at ~8 threads/cores (~1.7–1.8×) and then *regresses*. SpMV does almost no arithmetic per byte streamed from memory, so once the DRAM controller saturates (~8 cores on this dual-channel part), extra cores add only contention. The GPU's ~6.7× advantage equals its memory-bandwidth advantage (GDDR6 vs dual-channel DDR) — it is a hardware gap, not a software one.

Density does not change this. Re-running on **ML_Laplace** (377,002 rows, 27.7M nnz, ~73 nnz/row — 4× denser) gave the same shape: CPU plateaus ~7 ms at 8 threads, GPU 1.12 ms, ratio ~6.4×. More nnz/row means more bytes per output element for both devices, so the ratio holds.

### 2. Implementation barely matters: OpenMP ≈ MKL > MPI

- **MKL does not beat naive OpenMP.** MKL's inspector-executor helps single-thread efficiency (better vectorization) but cannot manufacture bandwidth. At 8 threads both land at ~8 ms. On the regular ML_Laplace operator MKL was actually *slower* than the naive loop at every thread count.
- **MPI is the slowest CPU approach**, even with P/E-aware partitioning, because on a single node it adds halo exchange (`MPI_Isend`/`Irecv` + send-buffer packing) and forfeits the shared address space that lets threads read all of X directly. MPI's value is *multi-node* aggregate bandwidth; on one node it only adds overhead.

### 3. P/E heterogeneous partitioning helps MPI, but can't break the ceiling

Sweeping the P-core:E-core weight ratio (E fixed at 100) with 16 ranks pinned one-per-physical-core (`data/rankfile/8P_8E.rankfile`, ranks 0-7 = P-cores, 8-15 = E-cores):

| Weight P:E | ms/iter | Speedup vs 1 core |
|-----------:|--------:|------------------:|
| 100:100 (uniform) | 11.58 | 1.22× |
| 150:100 | 11.12 | 1.27× |
| 200:100 | 10.93 | 1.29× |
| 219:100 | 10.83 | 1.31× |
| 250:100 | 10.77 | 1.31× |
| 300:100 | 10.82 | 1.31× |

Optimal ratio is **~2.2–2.5:1**, implying a P-core delivers ~2.3× the effective SpMV throughput of an E-core (higher clock + more usable bandwidth). The weighting recovers the best *MPI* result (1.31× vs 1.22× uniform), but MPI's floor sits above OpenMP's, so it still loses to shared-memory threading.

## Implications for the Hybrid Solver

This is the root cause behind the BiCGStab `mpi_gpu` results:

- A CPU rank contributes throughput proportional to its memory bandwidth, which is ~1/6.7 of the GPU's. The theoretical best a CPU co-processor can add is ~6-8% (matching the earlier 1.06–1.08× hybrid ceiling).
- That marginal gain is erased by MPI Allreduce latency in the solver and halo-exchange overhead in distributed SpMV.
- No CPU SpMV implementation (OpenMP, MKL, MPI) closes the bandwidth gap, so no partitioning scheme makes CPU+GPU beat GPU-alone on this hardware.

The negative result is robust and explained from first principles: **wall-clock SpMV performance tracks memory bandwidth, and the GPU has ~6.7× more of it.**

## Reproduction

```bash
# Prepare input (output of A*x will be [1..n])
micromamba run -n octave octave --no-gui --eval \
  "addpath('tools/scripts'); prepare_spmv_input('/matrices/cage14.mtx', 'data/matrices/cage14/in')"

X=data/matrices/cage14/in/X_spmv.txt; M=/matrices/cage14.mtx

# GPU
./build/spmv-gpu -m $M -x $X -o /tmp/Y.txt -n 500
# OpenMP (best at 8 threads)
OMP_NUM_THREADS=8 ./build/spmv-cpu-mt -m $M -x $X -o /tmp/Y.txt -n 500
# MKL
MKL_NUM_THREADS=8 ./build/spmv-cpu-mkl -m $M -x $X -o /tmp/Y.txt -n 500
# MPI P/E (needs partition from data/weights/cpu-p-e/, e.g. w250_16)
OMP_NUM_THREADS=1 mpirun --rankfile data/rankfile/8P_8E.rankfile -np 16 \
  ./build/spmv-mpi -m $M -x $X -p data/matrices/cage14/in/part/w250_16_i1.part -o /tmp/Y.txt -n 500
```

All SpMV binaries report `spmv_per_iter` and `max_abs_err_vs_1ton` (correctness check vs `[1..n]`).
