# Running HetHypPar on TRUBA

Practical guide for taking the heterogeneous BiCGStab benchmarks from a single
consumer GPU (RTX 3070) to TRUBA's multi-node CPU/GPU clusters. Sourced from
TRUBA's user docs (`https://docs.truba.gov.tr/`, mdBook site, fetched
2026-07-25 — the docs use a self-signed/incomplete cert chain, so `curl -k` or
an explicit trust-store fix is needed to fetch them programmatically) plus a
live query against the SuiteSparse Matrix Collection stats index for §6.

Everything below is grounded in the fetched doc pages or a verified HTTP
request — nothing here is a guessed module name/version or a guessed matrix
URL. Anywhere the exact string depends on your account (module versions,
account name, ARF-ACC access), it's called out explicitly as **verify this**.

## 1. Which TRUBA resource to target

TRUBA is actually several separate clusters behind separate login nodes:

| Resource | Login node | What's there | Access |
|---|---|---|---|
| **ARF** | `arf-ui1..5.yonetim` | `orfoz` (504×112-core, no GPU), `barbun`/`hamsi` (CPU-only), `barbun-cuda` (24×2×P100), `akya-cuda` (24×4×V100), `smp` (1 big-memory node), `debug` | Standard TRUBA account |
| **ARF-ACC** | `cuda-ui.yonetim` | `palamut-cuda` (9×8×A100), `kolyoz-cuda` (24×4×H100 + 48×4×H200, NDR IB) | **Restricted** — only research-center infrastructure projects and ULAKBİM-contracted projects. Regular accounts are pointed back to `barbun-cuda`/`akya-cuda`. |
| MareNostrum 5 / EuroHPC systems | separate | LUMI, Leonardo, MareNostrum5, etc. | Separate call/allocation process, not covered here |

**Recommendation for this project:** use **ARF**, not ARF-ACC — the H100/H200/A100
queues would be nice but you likely don't have the contracted-project access,
and asking for it is a separate bureaucratic step. Within ARF:

- **`orfoz`** for CPU-only multi-node MPI scaling (`bicgstab-mpi`, `bicgstab-cpu`)
  — 504 nodes × 112 cores is by far the most CPU horsepower/parallelism TRUBA
  offers, and there's no GPU queue anywhere close to that node count.
- **`akya-cuda`** for multi-node MPI+GPU hybrid runs (`bicgstab-mpi-gpu`,
  `bicgstab-hybrid-*`) — 24 nodes × 4× V100 = up to 96 GPUs across nodes, which
  is the whole point of going to TRUBA (real inter-node `MPI_Allreduce`/halo
  cost, vs. the single-node PCIe/NVLink-free cheap sync we measured locally).
  `barbun-cuda` (P100, 2/node) is the fallback if `akya-cuda` is oversubscribed.
- **`debug`** (max 4h, spans all server families via `-C <family>`) for every
  build/smoke-test iteration. **Never build or run anything non-trivial
  directly on the `arf-ui*` login nodes** — that's an explicit rule in the
  docs, and login-node compiles previously caused the CUDA-arch JIT gotcha we
  already know about locally (see §4).

Queue hardware table (from `arf_kuyruk_bilgisi.html` / `hesaplama_kumeleri.html`):

| Queue | Nodes | CPU | Cores/node | GPU | Min cores | Min GPU | Max time |
|---|---:|---|---:|---|---:|---:|---|
| orfoz | 504 | 2× Xeon Platinum 8480+ | 112 | — | 56 | — | 3 days |
| hamsi | 144 | 2× Xeon Gold 6338 | 56 | — | 28 | — | 3 days |
| barbun | 119 | 2× Xeon Gold 6248R | 40 | — | 20 | — | 3 days |
| barbun-cuda | 24 | 2× Xeon Gold 6248R | 40 | 2× P100 16GB | 20 | 1 | 3 days |
| akya-cuda | 24 | 2× Xeon Gold 6248R | 40 | 4× V100 16GB NVLink | 10 | 1 | 3 days |
| smp | 1 | 4× Xeon Gold 6248R | 224 | — | 4 | — | 3 days |
| debug | 200 (mixed) | varies | varies | varies | 1 | — | **4 hours** |

All families share high-speed Infiniband and the same `/arf/home`, `/arf/scratch`
filesystems. Default job time if you forget `--time` is **2 minutes** — always set it.

## 2. Getting in

1. Need an active TRUBA account + project allocation (`0-truba/kullanici_basvurulari.html`,
   `0-truba/proje_basvurulari.html` if you don't have one yet).
2. **OpenVPN must be connected first** — nothing below works without it
   (`2-temel_bilgiler/openvpn_installation/openvpn_info.html`).
3. SSH to a login node:
   ```
   ssh <username>@172.16.6.11    # arf-ui1, or .12/.13/.14/.15 for ui2-5
   ```
   For file transfer specifically, the docs recommend `arf-ui4`/`arf-ui5`
   (172.16.6.14/.15) as the faster pair.
4. For ARF-ACC (if you get access): `ssh -l <username> 172.16.6.16` (`cuda-ui`).
5. Grafana monitoring: `http://172.16.6.25:3000/login`. Open OnDemand web UI
   also exists for a browser-based Linux desktop / interactive session.

## 3. Storage — read this before you `git clone`

Two filesystems, both parallel/high-performance, **neither backed up**:

| Path | Purpose | Lifetime |
|---|---|---|
| `/arf/home/$USER` | scripts, configs, small persistent files | user-controlled, no auto-delete |
| `/arf/scratch/$USER` | active job I/O, big datasets | **auto-deleted after 30 days** |

- Quota: **2 TB and 500K inodes per user**, per the docs' wording spanning both
  filesystems together — check your actual split with the quota view on
  `arf-ui1` rather than trusting that number blindly.
- `#SBATCH --output=` / `--error=` **must** point into scratch.
- **Do not `conda`/`pip`/`micromamba` install onto `/arf`** — the docs call
  this out explicitly: thousands of tiny package files blow the inode quota
  and degrade the shared filesystem for everyone. This directly conflicts with
  this repo's `micromamba run -n octave python <script>` convention — see §7
  for the workaround (module-provided Python/Octave, or an Apptainer
  container).
- Use `/tmp` (NVMe on every node, 1.4TB dedicated on `akya-cuda`) for
  high-I/O scratch within a single job; it does not survive past the job.
- `module load` and containers instead of per-user installs — both are called
  out as the recommended way to avoid inode bloat.

## 4. Building

Modules are discovered with the `avci` search tool (wraps `module avail` with
highlighting):
```
avci gcc
avci openmpi
avci cuda
avci mkl        # returns nothing directly -- MKL ships inside comp/oneapi/*
avci octave     # check whether Octave is available before assuming you need conda
```
**Confirmed on this account (2026-07-27)** — there is no bare `mkl` module;
MKL (`mkl_intel_lp64`/`mkl_intel_thread`/`mkl_core`) and `iomp5` come from
`comp/oneapi/{2022,2023,2024}`, which `src/CMakeLists.txt`'s
`find_package(MKL REQUIRED)` needs regardless of whether you're also building
CUDA targets. NVHPC (`comp/nvhpc/nvhpc-23.11`/`25.3`) looks like an
all-in-one CUDA+MPI+compiler stack but does **not** substitute for this — it
doesn't provide Intel MKL, which this project links unconditionally.

Working module set for a GPU build (`akya-cuda`, V100):
```bash
module purge
module load comp/gcc/12.3.0
module load lib/cuda/12.4
module load lib/openmpi/5.0.4-cuda-12.4    # CUDA-aware build, paired with the CUDA version above
module load comp/oneapi/2024                # MKL + iomp5 -- load last
```
For CPU-only builds/runs (`orfoz`, no GPU) drop the two CUDA lines — but
**keep `comp/oneapi/2024` loaded even there**: MKL is linked unconditionally,
so a CPU-only binary still needs `libmkl_intel_lp64.so`/`libiomp5.so`
resolvable at runtime, not just at build time. An alternate CUDA/MPI pairing
exists if 12.4 has driver issues on the older V100 nodes:
`lib/cuda/13.0` + `lib/openmpi/5.0.10-cuda-13.0`.

Because `comp/oneapi/*` and `comp/gcc/*` can both land `cc`/`gcc` on `PATH`,
don't rely on load order to pick the compiler — pass it explicitly:
```bash
cmake -S src -B build -G Ninja -DCMAKE_C_COMPILER=$(command -v gcc) -DCMAKE_CUDA_ARCHITECTURES=70
```
and sanity-check after loading modules, before trusting a build:
```bash
which gcc mpicc nvcc; echo "MKLROOT=$MKLROOT"
```

CUDA architecture flag — same gotcha as the local RTX 3070 (`CMAKE_CUDA_ARCHITECTURES`
defaulting wrong causes an in-loop JIT compile that wrecks the timing):

| GPU | Arch | `-DCMAKE_CUDA_ARCHITECTURES=` |
|---|---|---|
| P100 (barbun-cuda) | Pascal | `60` |
| V100 (akya-cuda) | Volta | `70` |
| A100 (palamut-cuda) | Ampere | `80` |
| H100 / H200 (kolyoz-cuda) | Hopper | `90` |

Do the module load + cmake configure/build from the recipe above inside a
`debug` interactive session (§5), never on the login node.

## 5. Interactive / debug testing

```bash
# generic debug node
srun -p debug -N 1 -n 1 -c 110 -A <account> -J test --time=0:30:00 --pty /usr/bin/bash -i
# pin to a specific family with -C
srun -p debug -C akya-cuda   -N 1 -n 1 -c 10 --gres=gpu:1 -A <account> -J test --time=0:30:00 --pty /usr/bin/bash -i
srun -p debug -C barbun-cuda -N 1 -n 1 -c 20 --gres=gpu:1 -A <account> -J test --time=0:30:00 --pty /usr/bin/bash -i
srun -p debug -C orfoz       -N 1 -n 1 -c 55                -A <account> -J test --time=0:30:00 --pty /usr/bin/bash -i
```
Use this for: building, single-node smoke tests of each solver variant,
generating partitions with `patpart` (remember the known stack-VLA-overflow
fix from CLAUDE.md — `ulimit -s 262144` before invoking it, needed above
~2.1M rows, which every matrix in §6 exceeds).

Batch templates for the actual experiments are in `tools/slurm/` (see §8) —
`sbatch tools/slurm/<template>.slurm`.

## 6. Matrices big enough to need multiple nodes

**The problem:** the local `/matrices/` mirror tops out at `ML_Geer`
(110.9M nnz, see `data/matrix_survey.tsv`) — even that finishes a SpMV in
~1ms on the 3070, so a full solve is over in a few seconds. That's nowhere
near enough work to make multi-node communication (real Infiniband
`MPI_Allreduce`, not a same-PCIe-root CPU↔GPU sync) show up in the timing.

**The fix:** pull directly from the full SuiteSparse Matrix Collection
(`sparse.tamu.edu`) instead of the university's local subset. I queried the
live collection stats index (`https://sparse.tamu.edu/files/ssstats.csv`,
2904 matrices) for **square, real-valued, non-binary** matrices above 200M
nnz, and verified the download URL pattern
(`https://sparse.tamu.edu/MM/<Group>/<Name>.tar.gz`) actually resolves for
each of these (HTTP 200, real gzip bytes, confirmed live):

| Group | Matrix | Rows | nnz | nnz/row | Kind |
|---|---|---:|---:|---:|---|
| Fluorem | **HV15R** | 2.0M | 283M | **140.3** | CFD — very dense, should show the CPU-split benefit strongly (recall density↔benefit, Spearman 0.91, cutoff ~8-10) |
| Janna | **Queen_4147** | 4.1M | 316M | 76.3 | 2D/3D FEM — same SuiteSparse group as our existing `cage`/`rma10`/`ML_Geer` |
| Schenk | nlpkkt160 | 8.3M | 225M | 27.0 | KKT optimization |
| Schenk | nlpkkt200 | 16.2M | 440M | 27.1 | KKT optimization |
| Schenk | nlpkkt240 | 28.0M | 761M | 27.2 | KKT optimization |
| VLSI | stokes | 11.4M | 349M | 30.5 | Semiconductor process |
| GAP | GAP-kron / GAP-urand | 134M | 4.2–4.3B | ~32 | Synthetic R-MAT graphs — huge but not a physical PDE/circuit; timing-only like several matrices we already use |
| Sybrandt | MOLIERE_2016 | 30.2M | **6.7B** | 220.5 | Weighted graph — the single largest real matrix in the whole collection |

**Caution:** `nlpkkt120` is already documented as crashing even the plain GPU
solver (`docs/big-benchmark-findings.md`, "one casualty"). Test **nlpkkt160**
first, in isolation, before committing a multi-day multi-node allocation to
`nlpkkt200`/`240` — if it's a family-wide numerical issue (likely, given KKT
systems are indefinite) rather than a size issue, the whole Schenk row above
is a dead end and you should lean on HV15R/Queen_4147/stokes instead.

**Recommended order:** start with **HV15R** (best density-for-hybrid-benefit,
smallest download at ~200MB compressed) and **Queen_4147** (familiar Janna
family, moderate size) to get the pipeline working end-to-end on a real
multi-node allocation, then scale up to GAP-kron/MOLIERE_2016 once you've
confirmed the partitioning + I/O + runtime pipeline holds together — those
are big enough downloads/memory footprints that you want the mechanics
already debugged.

Fetch + prep on `/arf/scratch` (inside a `debug` session, not the login node):
```bash
tools/scripts/fetch_suitesparse_matrix.sh Fluorem HV15R /arf/scratch/$USER/matrices
# -> /arf/scratch/$USER/matrices/HV15R/HV15R.mtx

# same input-vector pipeline as local matrices (needs Octave -- see the
# conda/module conflict note in §3/§7)
octave --no-gui --eval "addpath('tools/scripts'); process_matrixi('HV15R', '/arf/scratch/$USER/matrices')"

# partition for N ranks (needs patpart built + the stack-limit fix)
ulimit -s 262144
python tools/python/matrix_partition.py -w data/weights -o gpu-cpu   # or your own weight files for N nodes
```

## 7. The Octave/conda dependency problem

This repo's whole tooling assumes `micromamba run -n octave python <script>`
and an Octave install for matrix prep (`process_matrixi.m`,
`prepare_spmv_input.m`). TRUBA explicitly asks you not to install
conda/pip packages onto `/arf`. Two ways out, in order of effort:

1. **Check `avci octave` and `avci python`** — if TRUBA ships module-provided
   Octave and a Python with numpy/scipy/pandas, just `module load` those and
   skip conda entirely. Cheapest if it works.
2. **Apptainer/Singularity container** — TRUBA supports it
   (`2-temel_bilgiler/konteyner_kullanimi.html`), built from a Docker image or
   your own def file. Build it **inside a `debug` srun session**, not on the
   login node (explicitly required by the docs). This gets you the exact
   `octave` env (pyarrow/matplotlib/cairosvg included) without touching `/arf`'s
   inode budget beyond the one `.sif` file.

Not investigated further here — pick based on what `avci octave`/`avci python`
turns up once you're logged in.

## 8. SLURM templates

Three ready-to-edit templates in `tools/slurm/` (every `<...>` placeholder —
account name, module versions, matrix name, partition file — needs filling in
for your account; nothing in them is guessed):

- **`debug_smoke_test.slurm`** — single-node build + single-solver smoke test
  on the `debug` queue. Run this first, always.
- **`mpi_cpu_scaling.slurm`** — multi-node `bicgstab-mpi` (CPU-only) on
  `orfoz`, one rank per node, full 112 cores/rank via OpenMP. Sweep `-N` for a
  weak/strong scaling curve.
- **`mpi_gpu_hybrid.slurm`** — multi-node `bicgstab-mpi-gpu` on `akya-cuda`,
  one rank per GPU (4 ranks/node). Sweep `-N` the same way.

```bash
sbatch tools/slurm/debug_smoke_test.slurm
sbatch --nodes=4 tools/slurm/mpi_cpu_scaling.slurm
sbatch --nodes=2 tools/slurm/mpi_gpu_hybrid.slurm
```

## 9. What this is actually for

The single-node conclusions (PaToH ≈ naive for 2-way GPU/CPU splits,
pipelined BiCGStab loses, dot-product device-pointer mode wins) all rest on
the same premise: **on one node, synchronization is cheap**, so techniques
that trade extra compute for fewer synchronizations don't pay off
(`docs/ca-sgd-relation.md` makes this explicit). TRUBA's actual point is to
re-run the same comparisons where that premise is false:

- Does PaToH's cut-minimization start mattering once the "communication" is a
  real Infiniband `MPI_Allreduce`/halo exchange across nodes, not a
  same-machine CPU↔GPU sync? (The single-node data already shows PaToH's edge
  scales with how much a solver communicates — `mpi-gpu` at ~4-5% is the
  biggest single-node PaToH win we've measured; multi-node should widen that
  gap in PaToH's favor.)
- Does the pipelined variant (built to hide reduction latency behind compute)
  flip from <1× to a real win once there's real network latency to hide?
- Does the CPU-vs-GPU-bandwidth ceiling (~1.06-1.3×) change shape once you're
  aggregating many GPUs's worth of bandwidth against many CPU nodes' worth,
  rather than one of each?

Matrices from §6 large enough to run for tens of seconds to minutes per
configuration (not the sub-second local runs) are required to get a clean
signal on any of this — that's the actual reason the current SuiteSparse
subset (max 111M nnz) isn't sufficient, independent of TRUBA's raw compute
capacity.
