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

### Next

- **Port to the hybrid solver** (`bicgstab-hybrid-async`) — same change, applied
  on top of the CPU/GPU SpMV overlap. The hybrid currently tops out at 1.14×
  (rma10 w400); the dots are still full-size GPU work there, so device-pointer
  mode should stack on top.
- Then fuse `T·S`/`T·T` and evaluate pipelined BiCGStab.
