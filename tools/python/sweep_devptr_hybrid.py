#!/usr/bin/env python3
"""
sweep_devptr_hybrid.py -- re-sweep CPU/GPU weights for the device-pointer hybrid.

The per-matrix weight optima in docs/dot-product-profiling.md were tuned for the
dot-heavy host-pointer hybrid. With device-pointer dots the dot tax is largely
gone, so the optimal CPU/GPU split may shift. This sweeps every available
gpu-cpu weight for bicgstab-hybrid-async-dp, times it best-of-N vs the pure-GPU
baseline, and marks the best weight per matrix.

Only run on CONVERGENT matrices (so the timing reflects a real solve): default
cage11, cage12, cage13. rma10 diverges with unpreconditioned BiCGStab and is
excluded.

Build first (native arch):
    cmake -S src -B build -G Ninja -DCMAKE_CUDA_ARCHITECTURES=86
    cmake --build build --target bicgstab-gpu bicgstab-hybrid-async-dp

Usage:
    micromamba run -n octave python tools/python/sweep_devptr_hybrid.py \
        [--iters 1000] [--repeats 2] [--matrices cage11,cage12,cage13]
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MTX = Path("/matrices")
IS_GPU = ROOT / "data/is_gpu/g2_2.txt"


def loop_time(cmd):
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, cwd=ROOT)
    if p.returncode != 0:
        sys.stderr.write(f"FAILED: {' '.join(map(str,cmd))}\n{p.stderr[-300:]}\n")
        return None
    m = re.search(r"^spmv\s*:\s*([-+0-9.eE]+)", p.stdout, re.M)
    return float(m.group(1)) if m else None


def best_of(cmd, repeats):
    ts = [t for _ in range(repeats) if (t := loop_time(cmd)) is not None]
    return min(ts) if ts else None


def weights_for(m):
    d = ROOT / f"data/matrices/{m}/in/part/gpu-cpu"
    ws = sorted(p.stem.split("_")[0] for p in d.glob("*_2_i1.part"))
    # numeric sort by the wNNN value
    return sorted(ws, key=lambda w: int(w[1:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", default="1000")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--matrices", default="cage11,cage12,cage13")
    args = ap.parse_args()

    for m in args.matrices.split(","):
        mtx, x, b = MTX / f"{m}.mtx", ROOT / f"data/matrices/{m}/in/X_init.txt", ROOT / f"data/matrices/{m}/in/B.txt"
        if not mtx.exists():
            sys.stderr.write(f"skip {m}: {mtx} missing\n"); continue
        gpu = best_of(["build/bicgstab-gpu", "-m", mtx, "-x", x, "-y", b,
                       "-o", "/tmp/g.txt", "-n", args.iters], args.repeats)
        print(f"\n=== {m} ===  pure-GPU baseline = {gpu*1e3:.2f} ms")
        print(f"{'weight':<8}{'hybrid_dp(ms)':>14}{'speedup_vs_gpu':>16}")
        rows = []
        for w in weights_for(m):
            part = ROOT / f"data/matrices/{m}/in/part/gpu-cpu/{w}_2_i1.part"
            t = best_of(["build/bicgstab-hybrid-async-dp", "-m", mtx, "-x", x, "-y", b,
                         "-p", part, "-g", IS_GPU, "-o", "/tmp/hd.txt", "-n", args.iters],
                        args.repeats)
            if t is None:
                continue
            rows.append((w, t, gpu / t))
            print(f"{w:<8}{t*1e3:>14.2f}{gpu/t:>15.3f}x")
        if rows:
            bw, bt, bs = max(rows, key=lambda r: r[2])
            print(f"  -> best: {bw}  {bs:.3f}x vs pure-GPU  ({bt*1e3:.2f} ms)")


if __name__ == "__main__":
    main()
