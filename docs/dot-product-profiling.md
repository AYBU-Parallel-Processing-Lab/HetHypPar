# Dot-Product Contribution in BiCGStab (hybrid vs pure-GPU)

**Question (from the professor):** *"Can't we get any speedup on dot-product
computations? Dot products are simpler — just a reduction."*

This measures how much of each BiCGStab iteration is actually spent in dot
products, on the newest single-process solvers.

## Setup / reproducing

```bash
cmake --build build --target bicgstab-hybrid-async bicgstab-gpu spmv-hybrid-async
micromamba run -n octave python tools/python/profile_dot_breakdown.py \
    --iters 1000 --repeats 3
# -> data/profile_dot/breakdown.tsv
```

Hardware: RTX 3070 (8 GB) + 24-core host. Matrices from `/matrices/`.
Partitions: `data/matrices/<m>/in/part/gpu-cpu/<weight>_2_i1.part`, `g2_2.txt`
(rank0=GPU, rank1=CPU). Weight `wK_2` = GPU:CPU load `K:100`.

### Measurement method

Each solver runs **twice** per configuration:
- **clean** (`HHP_PROFILE` unset) → true loop time / speedup, no perturbation.
- **profiled** (`HHP_PROFILE=1`) → the loop drains `compute_s` at every category
  boundary and accumulates host wall-clock into **dot / spmv / vec** buckets.

The sync barriers perturb the async overlap, so the *total* from a profiled run
is inflated — totals/speedups below come from the clean run, the **percentages**
from the profiled run. Validation: profiled `sum` of buckets accounts for ~99%
of the profiled loop time, so attribution is clean.

The 5 dots/iter are: `R·R₀`, `R₀·V`, `T·S`, `T·T`, `S·S`. The 2 SpMVs/iter are
`V=A·P`, `T=A·S`. "vec" = all `axpy`/`scal`/device-to-device copies.

## Results (1000 iters, median of 3)

### Per-iteration time split (profiled %)

| Matrix | Solver | **dot %** | spmv % | vec % |
|--------|--------|----------:|-------:|------:|
| cage11 | gpu        | **40.6** | 42.8 | 16.4 |
| cage11 | hybrid w2720 | **40.3** | 43.7 | 16.0 |
| cage12 | gpu        | **29.6** | 56.0 | 14.4 |
| cage12 | hybrid w800  | **30.2** | 54.6 | 15.1 |
| rma10  | gpu        | **22.6** | 68.1 |  9.3 |
| rma10  | hybrid w800  | **24.0** | 66.2 |  9.8 |

### Best hybrid speedup vs pure-GPU (clean loop time)

Sweep extended down to w400/w600 (more CPU rows). The BiCGStab optimum sits at
*lower* weight (more CPU) than the original w800–w2720 range suggested:

| Matrix | Best weight | hybrid total | gpu total | **speedup** |
|--------|-------------|-------------:|----------:|------------:|
| cage11 | w2720 | 191.3 ms | 184.7 ms | **0.965×** (never wins) |
| cage12 | w800  | 354.2 ms | 378.0 ms | **1.067×** |
| rma10  | w400  | 299.6 ms | 339.9 ms | **1.135×** (w100 ≈ 1.140×, flat optimum) |

Note the BiCGStab optimum differs from the SpMV-only optimum: SpMV-only always
prefers more CPU (rma10 w400 **1.189×**, cage12 w400 **1.175×**), because the
extra dot/vec work in BiCGStab runs full-size on the GPU regardless and the CPU
rows only help the SpMV share. cage11 never wins — it is the sparsest, so its
SpMV share (the only thing the CPU offloads) is smallest. See `breakdown.tsv`.

## Takeaways

1. **Dot products are a large, first-order cost: 23–41% of per-iteration
   compute time**, second only to SpMV. The professor's intuition is correct —
   they are very much worth attacking.

2. **The dot fraction scales inversely with row density.** Sparser matrices
   (cage11, ~few nnz/row) spend ~40% in dots; denser rma10 spends ~23% because
   SpMV dominates. So dot optimization pays off *most* on the sparse matrices
   where the hybrid currently loses (cage11).

3. **Why 5 "simple reductions" cost so much: synchronization, not FLOPs.**
   `cublasDdot` runs in **host-pointer mode**, so each of the 5 dots blocks the
   host on a device→host copy of one scalar. That is 5 full GPU stalls per
   iteration. The reduction arithmetic is cheap; the serialized launch + D→H
   round-trips are the cost.

4. **Hybrid doesn't change the split** — dot/spmv/vec percentages are nearly
   identical hybrid vs pure-GPU, because the dots run full-size on the GPU in
   both. The hybrid only offloads SpMV rows, so its ceiling is bounded by the
   SpMV fraction (a smaller share on sparse matrices → cage11 can't win).

## Suggested next steps to actually speed up the dots

- **Device-pointer mode + on-device scalar math.** Set
  `cublasSetPointerMode(CUBLAS_POINTER_MODE_DEVICE)`, keep `rho/alpha/omega` on
  the device, and compute `beta = (rho/rho_prev)*(alpha/omega)` etc. in a tiny
  device kernel. This removes the 5 blocking D→H syncs per iteration and lets
  the whole iteration pipeline on the GPU — likely the single biggest win.
- **Fuse adjacent reductions.** `T·S` and `T·T` share the operand `T`; compute
  both in one pass (one kernel / batched dot) to halve that launch pair.
- **Reduce reduction count structurally.** Pipelined / communication-avoiding
  BiCGStab variants cut the number of distinct reduction points per iteration;
  this is the high-impact direction once the sync round-trips are the binding
  cost (which the data above shows they are).
- The CPU has spare cycles during the GPU dots — but a CPU-side partial dot
  needs the vector on the host every iteration (a D→H transfer), which on these
  matrices costs more than the dot itself. Keep the dots on the GPU; attack the
  *sync latency*, not the device placement.

## Design: device-pointer-mode dots (the top recommendation, in detail)

### Why the dots stall today

Every dot uses `cublasDdot` in cuBLAS's **default host-pointer mode**
(`device_vector_dot` → `cublasDdot(..., double *out)` with `out` on the host).
In this mode cuBLAS launches the reduction kernel and then **blocks the host
until the scalar result has been copied device→host**. The host cannot queue
the next op until that round-trip lands. With 5 dots/iter that is **5 full
host↔device stalls per iteration** — and the profiling shows that stall, not the
reduction FLOPs, is what makes dots 23–41% of the loop.

```
queue dot ─▶ [HOST STALL on D→H scalar] ─▶ host computes beta ─▶ queue ops ─▶ next dot ─▶ [STALL] ...
```

### The change

```c
cublasSetPointerMode(bh, CUBLAS_POINTER_MODE_DEVICE);
```

In device-pointer mode `cublasDdot` writes its result to a **device** pointer and
returns immediately (no host stall); the kernel just sits queued on the stream.
The scalar now lives on the GPU, so the host can no longer compute
`beta = (rho/rho_prev)*(alpha/omega)`. That arithmetic moves into a **one-thread
device kernel** that reads the device-resident `rho/alpha/omega` and writes
`beta`, `-alpha`, `-omega` back to device scalars. cuBLAS `axpy`/`scal` then read
*their* scale factors from those same device pointers (in device mode the `alpha`
argument is also a device pointer). The whole iteration becomes one
uninterrupted stream of GPU kernels with **zero host syncs**.

```c
// BEFORE (host mode): 5 stalls/iter
device_vector_dot(bh, R, R_0, &tmp_rho);                 // STALL
beta = (tmp_rho/rho)*(alpha_s/omega);                    // host arithmetic
...

// AFTER (device mode): 0 stalls/iter
cublasDdot(bh, n, R.vals,1, R_0.vals,1, d_rho);          // async, result on device
bicgstab_update_beta<<<1,1,0,s>>>(d_beta, d_rho, d_rho_prev, d_alpha, d_omega);
cublasDdot(bh, n, R_0.vals,1, V.vals,1, d_rv);           // async
bicgstab_update_alpha<<<1,1,0,s>>>(d_alpha, d_rho, d_rv);
cublasDaxpy(bh, n, d_neg_alpha, V.vals,1, S.vals,1);     // reads scale from device
// ... host syncs only every k iters to read the residual for the stop test
```

### Why it's faster: the per-iteration timeline

The CPU drives the GPU by pushing commands into a stream and normally runs ahead
without waiting. A host-pointer dot breaks that: the scalar result must land in
CPU memory, so cuBLAS launches the reduction, **waits for the GPU**, and copies
one number back — the CPU call blocks and the GPU drains. BiCGStab has 5 dots,
so 5 such stalls per iteration:

```
baseline (host-pointer, 5 stalls/iter):
  dot🛑 βcalc  vec  SpMV  dot🛑  vec  SpMV  dot🛑 dot🛑  vec  dot🛑
     └ CPU frozen, GPU idle on a 1-number round-trip ┘   (×5 every iteration)

prototype (device-pointer, 0 stalls/iter):
  dot✅ βkern✅ vec✅ SpMV✅ dot✅ αkern✅ vec✅ SpMV✅ dot✅ dot✅ ωkern✅ vec✅ dot✅
     └──── all queued back-to-back; GPU streams through; CPU syncs ONCE after the loop ────┘
```

The dot *results* (and the `beta/alpha/omega` derived from them) stay on the GPU;
the tiny scalar kernels do the arithmetic the CPU used to do, and the following
`scal`/`axpy` read their scale factors straight from device memory. Same math,
no round-trips.

### Scope / risk

- Device storage for ~8 scalars (`rho, rho_prev, alpha, omega, beta, -alpha,
  -omega`, plus the dot results) and 3 trivial `<<<1,1>>>` scalar-update kernels.
- All cuBLAS constant scale args (`1.0`, `-1.0`) must also become device pointers.
- The per-iteration `printf` residual goes away (reading it needs a stall);
  the convergence check moves to every-*k*-iterations. Fine for fixed-iteration
  timing runs.
- Complementary to fusing `T·S`/`T·T` and to pipelined BiCGStab — do this first
  because the data shows the binding cost is sync latency, which this removes.

## Prototype results: device-pointer-mode dots (pure GPU)

Implemented and measured. The prototype is `bicgstab-gpu-dp`
(`src/entry/bicgstab_gpu_dp.c`) with on-device scalar kernels in
`src/util/hhp_dp_kernels.cu`; the baseline is the unchanged `bicgstab-gpu`.

### Reproduce

```bash
# Build the kernels for the GPU's NATIVE arch (see gotcha below)
cmake -S src -B build -G Ninja -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --target bicgstab-gpu bicgstab-gpu-dp
micromamba run -n octave python tools/python/compare_devptr.py --iters 1000 --repeats 4
```

### Speedup (best-of-4, 1000 iters, RTX 3070)

| Matrix | dot % (baseline) | baseline | devptr | **speedup** |
|--------|-----------------:|---------:|-------:|------------:|
| cage11 | 40.6 | 0.184 s | 0.152 s | **1.21×** |
| cage12 | 29.6 | 0.377 s | 0.331 s | **1.13×** |
| rma10  | 22.6 | 0.339 s | 0.308 s | **1.10×** |

The speedup tracks the dot fraction exactly as predicted: the matrix that spent
the most time in dots (cage11, 40.6%) gains the most. Removing the 5 host syncs
cut 10–17% off the whole BiCGStab loop.

### Correctness

The device-pointer math is **identical** to the baseline — verified bit-for-bit
where values stay finite:
- cage11 converges to residual 2.63e-16 by iter 30; baseline and devptr match to
  the last digit, `max|X_base − X_dp| = 0.0`.
- bips07_1693 (ill-conditioned, residual grows): identical residual at 50/200
  iters, `max|dX| = 0.0`.

(The fixed 1000-iteration timing runs deliberately iterate past convergence for
stable per-iteration averages; cage11/cage12/rma10 then break down to NaN — a
0/0 in the scalar updates, identical in both binaries and harmless to timing,
since per-iteration GPU cost is value-independent. Use a stopping criterion for
production solves.)

### Gotcha: compile kernels for the native arch

The build default was `CMAKE_CUDA_ARCHITECTURES=75` but the RTX 3070 is sm_86.
With sm_75 kernels the *first* launch JIT-compiles inside the timed loop and
adds ~0.4 s — which made cage11 (smallest, fastest loop) look 3× *slower* before
rebuilding with `-DCMAKE_CUDA_ARCHITECTURES=86`. cuBLAS/cuSPARSE ship native
fat binaries so the baseline never paid this; only the custom kernels did.

## Prototype results: device-pointer-mode dots in the HYBRID solver

The same change ported onto the CPU+GPU hybrid (`bicgstab-hybrid-async-dp`,
`src/entry/bicgstab_hybrid_async_dp.c`), reusing the identical scalar kernels.
The hybrid keeps its 2 unavoidable host syncs/iter inside `hybrid_spmv` (the CPU
must read each SpMV input) — device-pointer mode only removes the 5 dot syncs —
so the gain is expected to be smaller than pure-GPU, but it stacks on top of the
SpMV overlap.

### Reproduce

```bash
cmake -S src -B build -G Ninja -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --target bicgstab-gpu bicgstab-hybrid-async bicgstab-hybrid-async-dp
micromamba run -n octave python tools/python/compare_devptr_hybrid.py --iters 1000 --repeats 3
```

### Results (best-of-3, 1000 iters, RTX 3070, seconds)

| Matrix | weight | pure-GPU | baseline hybrid | **hybrid-dp** | dp / baseline-hybrid | **hybrid-dp / pure-GPU** |
|--------|--------|---------:|----------------:|--------------:|:--------------------:|:------------------------:|
| cage11 | w2720 | 0.1843 | 0.1879 | **0.1594** | 1.18× | **1.16×** |
| cage12 | w800  | 0.3770 | 0.3519 | **0.3109** | 1.13× | **1.21×** |
| rma10† | w400  | 0.3398 | 0.2991 | **0.2639** | 1.13× | **1.29×** |

† **rma10 does NOT converge** — unpreconditioned BiCGStab is unstable on it; the
residual grows and pure-GPU goes NaN by ~iter 80. Its numbers are *per-iteration
throughput only* (the per-iteration GPU work is value-independent), not a solved
system. `hybrid_dp` stays bit-identical to the baseline hybrid even there
(`max|dX|=0`), but pure-GPU vs hybrid trajectories diverge because an unstable
iteration amplifies the legitimate rounding difference between the whole-matrix
cuSPARSE SpMV and the permuted split SpMV. Treat rma10 as a timing probe, not a
correctness/speedup result.

Correctness (converging matrices, bit-identical to baseline hybrid AND pure-GPU):
| Matrix | iters | residual (all three) | max\|dX\| hyb-vs-dp | max\|dX\| gpu-vs-dp |
|--------|------:|----------------------|--------------------:|--------------------:|
| cage11 | 40 | 2.63e-16 | 0.0 | 0.0 |
| cage12 | 40 | 2.38e-16 | 0.0 | 0.0 |
| cage13 | 40 | 2.59e-16 | 0.0 | 0.0 |

### Takeaways

- **Device-pointer dots stack on the hybrid**, adding **1.13–1.18×** on top of
  the SpMV overlap — a touch better than expected given the 2 surviving SpMV
  syncs.
- **cage11 flips from a loss to a win.** Baseline hybrid was **0.965×** (slower
  than pure GPU); with device-pointer dots it is **1.16×**. The most
  dot-dominated matrix (40% dots) gained the most from removing the dot syncs.
- Best single-GPU speedup across the suite went from **1.14×** (rma10, baseline
  hybrid) to **1.29×**, and all three matrices now beat pure GPU.

### Weight re-sweep for the dp-hybrid (convergent matrices)

Once the dot tax is removed the iteration is relatively more SpMV-bound, so the
optimal CPU/GPU split can move toward more CPU. Swept best-of-2, 1000 iters,
`tools/python/sweep_devptr_hybrid.py`, vs pure-GPU:

| Matrix | pure-GPU | best weight | hybrid-dp | speedup | shift vs host-ptr optimum |
|--------|---------:|-------------|----------:|:-------:|---------------------------|
| cage11 | 184.5 ms | w2720 | 160.1 ms | **1.152×** | unchanged (wants max GPU) |
| cage12 | 376.4 ms | w600  | 312.7 ms | **1.204×** | w800 → w600 (more CPU) |
| cage13 | 1058.8 ms| w500  | 906.6 ms | **1.168×** | new; optimum at w500 (CPU-heavy) |

- cage12/cage13 shifted toward more CPU as predicted — broad interior optima
  (w400–w800 within ~1% of best).
- cage11 did not shift: it is the sparsest, so the CPU can't offload enough to
  beat the full-vector D→H transfer cost; it wants max GPU (optimum likely past
  w2720, the edge of the available partitions).
- Honest headline: **1.15–1.20× over pure-GPU on genuinely-solved systems**
  (cage11/12/13). The earlier 1.29× leaned on non-converging rma10.

## Negative result: distributing the dots too (`bicgstab-hybrid-dist-dp`)

To test whether the CPU should also take a share of the dots/vecops (not just
SpMV), `src/entry/bicgstab_hybrid_dist_dp.c` splits *every* vector device/host:
the GPU owns rows `[0,ng)`, the CPU owns `[ng,n)`. Vecops run independently on
each side (no communication); each dot becomes `GPU_partial + CPU_partial`,
combined on-device (`hhp_dp_add`) with the GPU partial in dp mode and the CPU
partial pushed up via H→D. Built with overlap (concurrent partials, device-side
combine so the GPU critical path stays on-device, async copies).

It is correct (cage11 @40 iters converges to 2.64e-16, X matches pure-GPU to
printed precision) but **slower** than keeping the dots on the GPU
(best-of-2, 1000 iters, tuned OpenMP thread count + fused host loops):

| Matrix | weight | gpu-dp | hybrid-dp | dist-dp | dist / gpu-dp | dist / hybrid-dp | dist / old-GPU |
|--------|--------|-------:|----------:|--------:|:-------------:|:----------------:|:--------------:|
| cage11 | w2720 | 0.1525 | 0.1598 | 0.242 | **0.63×** | 0.66× | 0.76× |
| cage12 | w600 | 0.3313 | 0.3135 | 0.455 | **0.73×** | 0.69× | 0.83× |
| cage13 | w500 | 1.0029 | 0.9126 | 1.205 | **0.83×** | 0.76× | 0.88× |

**Distributing the dots is a net loss (~0.8× of even the old GPU baseline)**, but
the cause is the unavoidable per-dot CPU↔GPU rendezvous: the combined scalar
gates the next vector op, so each dot needs ~3 `cudaStreamSynchronize`/iter on
top of the 2 SpMV syncs and cannot be hidden. dp mode cannot help — a
*distributed* reduction is intrinsically synchronizing, the exact thing dp
removes only for a *GPU-only* dot.

**Benchmarking caveat (corrected):** the first measurement of this solver used
the default 24 OpenMP threads, which is pathological for the tiny CPU slices
(fork/join over 24 threads on ~1400 elements) and reported a much worse
0.41–0.77× vs gpu-dp. Tuning the thread count (1–4) gave most of the recovery;
fusing the host vecops into single loops (`P=(P-ωV)β+R` etc., one region/one pass
instead of four) added only a further 2–7%. So the dominant cost is the dot
rendezvous, NOT OpenMP overhead — region count barely matters once threads are
sane. Always pin OpenMP threads to the CPU slice size for these small-`nc` runs.

This is the single-node microcosm of the multi-node Allreduce wall (see
HANDOFF.md): distributing a reduction across processors inserts a synchronizing
rendezvous per dot that swamps the offloaded compute. The lesson points the
opposite way from "distribute more" — do *fewer* reductions.

### Next

- Fuse `T·S`/`T·T` (shared operand `T`) into one batched reduction — fewer dots.
- Evaluate pipelined / communication-avoiding BiCGStab to cut the reduction
  count per iteration (the real lever, single-node and multi-node alike).
- Generate finer partitions below w400 / above w2720 to bracket the true optima
  (cage11 high end, cage12/13 low end).
