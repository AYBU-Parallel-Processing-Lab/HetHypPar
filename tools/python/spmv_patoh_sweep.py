#!/usr/bin/env python3
"""
SpMV-only PaToH-vs-naive comparison. Measures the *isolated* hybrid SpMV time
(spmv-hybrid-async, which times only Y=A*X with the GPU+CPU split + halo) for
each matrix under both partitions, so the partition method's effect is not
diluted by the dot-products / vector-ops of the full BiCGStab loop.

For each (matrix, weight) we run with the PaToH partition and the naive partition
and record `hybrid_per_iter`. The partition method is then compared at the SAME
weight (same nnz target) so only cut quality differs.

Run: micromamba run -n octave python tools/python/spmv_patoh_sweep.py [m1 m2 ...]
Default matrices = the 14 usable from the big benchmark.
Outputs:
  data/spmv_patoh/results.tsv
  docs/perf-profile-patoh-naive-spmv.{png,svg}
"""
import os, re, subprocess, sys
from pathlib import Path
import numpy as np

ROOT  = Path(__file__).resolve().parent.parent.parent
BUILD = ROOT / "build"
ISGPU = ROOT / "data/is_gpu/g2_2.txt"
OUT   = ROOT / "data/spmv_patoh"; OUT.mkdir(parents=True, exist_ok=True)
BIN   = BUILD / "spmv-hybrid-async"
REPS  = int(os.environ.get("REPS", "2"))
NITER = int(os.environ.get("NITER", "300"))
THREADS = [1, 2, 4]
PART = {"patoh": "gpu-cpu", "naive": "gpu-cpu-naive"}

DEFAULT = ["RM07R", "cage14", "Hamrle3", "rajat30_mc64_5", "circuit5M_dc", "circuit5M",
           "cage15", "ML_Geer", "vas_stokes_1M", "nv2", "Transport", "Freescale2", "dgreen", "ss"]

HYB_RE  = re.compile(r"hybrid_per_iter\s*:\s*([\d.eE+-]+)")
FULL_RE = re.compile(r"gpu_full_per_iter\s*:\s*([\d.eE+-]+)")

def run(mtx, x, part, nt):
    e = dict(os.environ); e["LC_ALL"] = "C"; e["OMP_NUM_THREADS"] = str(nt)
    cmd = [str(BIN), "-m", mtx, "-x", x, "-p", part, "-g", str(ISGPU), "-o", "/tmp/sp_y.txt", "-n", str(NITER)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=e)
    except subprocess.TimeoutExpired:
        return None, None
    h = HYB_RE.search(r.stdout); f = FULL_RE.search(r.stdout)
    return (float(h.group(1)) if h else None, float(f.group(1)) if f else None)

def best(mtx, x, part):
    bt = bf = None
    for nt in THREADS:
        for _ in range(REPS):
            t, f = run(mtx, x, part, nt)
            if t is not None and (bt is None or t < bt):
                bt, bf = t, f
    return bt, bf

def weights(m, src):
    d = ROOT / f"data/matrices/{m}/in/part/{PART[src]}"
    out = []
    for p in sorted(d.glob("*_2_i1.part")):
        mt = re.match(r"(w\d+)_2_i1\.part", p.name)
        if mt: out.append((mt.group(1), str(p)))
    return sorted(out, key=lambda t: int(t[0][1:]))

def sweep(matrices):
    rows = []
    for m in matrices:
        mtx = f"/matrices/{m}.mtx"
        x = str(ROOT / f"data/matrices/{m}/in/X_init.txt")
        if not Path(x).exists():
            print(f"skip {m}: no X_init", file=sys.stderr); continue
        print(f"### {m}", file=sys.stderr, flush=True)
        for src in ("patoh", "naive"):
            for w, part in weights(m, src):
                t, f = best(mtx, x, part)
                rows.append((m, src, w, t, f))
                print(f"  {src:<6} {w:<6} hybrid={t}", file=sys.stderr, flush=True)
    tsv = OUT / "results.tsv"
    with open(tsv, "w") as fo:
        fo.write("matrix\tpart_src\tweight\thybrid_per_iter\tgpu_full_per_iter\n")
        for (m, s, w, t, f) in rows:
            fo.write(f"{m}\t{s}\t{w}\t{t if t else ''}\t{f if f else ''}\n")
    print(f"wrote {tsv}", file=sys.stderr)
    return rows

def plot(rows):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from perf_profile import line_profile, log_x   # reuse the styled helpers
    # pair patoh vs naive at the same (matrix, weight)
    d = {}
    for (m, s, w, t, f) in rows:
        if t: d.setdefault((m, w), {})[s] = t
    pa, na = [], []
    for k, v in d.items():
        if "patoh" in v and "naive" in v:
            mn = min(v["patoh"], v["naive"])
            pa.append(v["patoh"] / mn); na.append(v["naive"] / mn)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    line_profile(ax, pa, "PaToH", "#2c5e9e", "o")
    line_profile(ax, na, "naive (nnz-balance)", "#e39a2e", "s")
    log_x(ax, max(max(pa), max(na)))
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("SpMV performance ratio  (method SpMV time / faster of {PaToH, naive})  —  log scale")
    ax.set_ylabel("fraction of (matrix, weight) cases")
    ax.set_title(f"PaToH vs naive — ISOLATED SpMV time  ({len(pa)} cases)")
    ax.grid(alpha=0.3); ax.legend(fontsize=10, loc="lower right", framealpha=0.95)
    fig.tight_layout()
    out = ROOT / "docs/perf-profile-patoh-naive-spmv"
    fig.savefig(f"{out}.png", dpi=140); fig.savefig(f"{out}.svg")
    print(f"wrote {out}.png/.svg  ({len(pa)} cases)", file=sys.stderr)

if __name__ == "__main__":
    mats = sys.argv[1:] or DEFAULT
    sys.path.insert(0, str(ROOT / "tools/python"))
    rows = sweep(mats)
    plot(rows)
