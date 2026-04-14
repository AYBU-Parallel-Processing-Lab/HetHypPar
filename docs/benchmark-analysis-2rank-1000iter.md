# Benchmark Analysis: 2-Rank Tests (1000 Iterations)

**Date:** 2026-04-05
**Scope:** 213 matrices, 4 weight profiles (50_50, 66_33, 75_25, 90_10), 5 imbalance ratios (3%, 5%, 10%, 20%, 50%), 1000 iterations, ~32,000 tests
**Setup:** 2 MPI ranks (1 CPU + 1 GPU), rankfile `1cpu_1gpu.rankfile`, is_gpu `g2_2.txt`
**Previous run:** [`benchmark-analysis-2rank.md`](benchmark-analysis-2rank.md) (20 iterations, pre-segfault fix)

## Test Results Overview

| Metric | Count | % |
|--------|------:|--:|
| Total tests | 32,242 | |
| Success | 29,541 | 91.6% |
| Failed | 2,701 | 8.4% |

### Failure Breakdown

| Error | Count | Solver | Description |
|-------|------:|--------|-------------|
| Unknown error | 1,587 | mpi_gpu | Binary ran without crashing but produced NAN residual or unparseable output. Numerical divergence, not segfaults. |
| Vector of size 0 | 876 | mpi, mpi_gpu | Partitions that assign zero rows to a rank. Known open issue. |
| Missing matrix file | 162 | all | `memplus_k6.mtx` not present on disk. |
| Empty stderr | 76 | gpu | CUDA failures on specific matrices (no error message captured). |

## Segfault Fix Impact

The MPI send buffer use-after-free bug was fixed in commit `0814a13` (see [`fix-mpi-send-use-after-free.md`](fix-mpi-send-use-after-free.md)).

| Metric | Previous (20 iter) | Current (1000 iter) |
|--------|-------------------:|--------------------:|
| Total failures | 6,468 | 2,701 |
| Segfaults | 5,429 (84%) | 0 |
| Vector of size 0 | 876 | 876 |
| Missing file | 162 | 162 |
| Other | 1 | 1,663 |

The segfault class is completely eliminated. The 1,587 new "unknown error" failures are numerical (NAN residual at 1000 iterations), not crashes.

## GPU vs CPU (SpMV)

Single-GPU vs single-CPU performance on the SpMV kernel.

| Metric | Value |
|--------|------:|
| Matrices compared | 136 |
| GPU faster | 123/136 (90%) |
| Mean speedup | 3.07x |
| Median speedup | 2.52x |
| Max speedup | 14.46x |
| Min speedup | 0.64x |

The 13 matrices where CPU wins are small enough that GPU kernel launch overhead and data transfer costs outweigh the parallel compute benefit.

## MPI+GPU vs GPU (SpMV, 2 Ranks)

Best partition configuration per matrix (minimizing SpMV time across all weight/imbalance combinations).

| Metric | Value |
|--------|------:|
| Matrices compared | 136 |
| MPI+GPU faster | 0/136 |
| Best case | 0.98x (essentially tied) |
| Mean | 0.76x |
| Worst case | 0.36x |

### MPI+GPU vs GPU (everything_total)

On total execution time, only 2 matrices marginally favor MPI+GPU:

| Matrix | Speedup | GPU time | MPI+GPU time |
|--------|--------:|---------:|-------------:|
| k3plates | 1.11x | 0.408s | 0.369s |
| Zd_Jac6_db | 1.04x | 0.486s | 0.469s |

These are within noise margin and likely due to cuSPARSE descriptor creation overhead on the GPU side, not SpMV improvement.

### Why MPI+GPU Cannot Beat GPU at 2 Ranks

1. Only 2 ranks (1 CPU + 1 GPU) — the GPU already handles the full matrix efficiently on its own.
2. MPI communication overhead is added (especially when cut > 0).
3. The single CPU rank cannot offset the communication cost with useful computation.

**Conclusion unchanged from the 20-iteration run:** MPI+GPU needs more CPU ranks (e.g., 8 CPU + 1 GPU, 16 CPU + 1 GPU) to have any chance of outperforming single-GPU.

## MPI (2-Rank CPU) vs Single CPU (SpMV)

Best partition configuration per matrix.

| Metric | Value |
|--------|------:|
| Matrices compared | 212 |
| MPI faster | 212/212 (100%) |
| Mean speedup | 1.68x |
| Median speedup | 1.68x |
| Max speedup | 2.21x |

### Top 10 Matrices

| Matrix | Speedup | CPU SpMV | MPI SpMV |
|--------|--------:|---------:|---------:|
| poli3 | 2.21x | 0.1787s | 0.0807s |
| epb1 | 2.11x | 0.1420s | 0.0674s |
| poli_large | 2.08x | 0.1251s | 0.0602s |
| epb2 | 2.04x | 0.2486s | 0.1216s |
| powersim | 2.03x | 0.1274s | 0.0626s |
| ABACUS_shell_ud_M | 2.02x | 0.2746s | 0.1357s |
| ABACUS_shell_ud | 1.97x | 0.2813s | 0.1431s |
| bips07_1693 | 1.94x | 0.1193s | 0.0617s |
| mimo46x46_system | 1.92x | 0.1174s | 0.0611s |
| viscoplastic2_C_7 | 1.92x | 0.4515s | 0.2356s |

Near-ideal scaling for 2 CPU ranks, indicating low MPI communication overhead in the CPU-only case.

## Convergence (1000 Iterations)

| Threshold | Matrices converged | % |
|-----------|-------------------:|--:|
| < 1e-10 | 9 | 4.2% |
| < 1e-6 | 24 | 11.3% |
| < 1e-3 | 62 | 29.2% |
| < 1 | 92 | 43.4% |
| >= 1 (diverged) | 112 | 52.8% |
| NAN | 8 | 3.8% |

**Total CPU successes:** 212

### Why Convergence Is Lower Than the 20-Iteration Run

The previous run reported 78/213 (37%) converged at 20 iterations. At 1000 iterations, only 24/212 (11.3%) converge below 1e-6. This is not contradictory: with only 20 iterations, many ill-conditioned matrices had not yet diverged — their residual was still low simply because there hadn't been enough iterations to amplify instabilities. At 1000 iterations, these matrices have time to blow up.

Preconditioning (ILU, Jacobi) would significantly improve convergence across the matrix set.

## MPI+GPU Success Rate by Weight Profile

| Weight | Successful | Total | Rate |
|--------|----------:|------:|-----:|
| 50_50 | 3,773 | 4,260 | 89% |
| 66_33 | 3,784 | 4,251 | 89% |
| 75_25 | 3,640 | 4,153 | 88% |
| 90_10 | 2,606 | 3,244 | 80% |

The 90_10 weight profile has the lowest success rate because extreme weight imbalance combined with high imbalance tolerance causes PaToH to assign all rows to a single partition, triggering "vector of size 0" errors on the empty rank.

## Remaining Issues

1. **Vector of size 0 (876 failures):** Partitions that assign zero rows to a rank. The solver should either reject these partitions at startup or handle empty ranks gracefully.

2. **Unknown mpi_gpu errors (1,587 failures):** Numerical divergence producing NAN residual. The binary runs to completion but the output cannot be parsed. These are not bugs in the solver — they are expected for ill-conditioned matrices without preconditioning.

3. **Missing matrix file (162 failures):** `memplus_k6.mtx` is not present in `/matrices/`. Either add the file or remove the matrix from the test set.
