#!/usr/bin/env python3
"""
compare_devptr_hybrid.py -- measure the device-pointer-mode dots inside the
CPU+GPU hybrid solver.

Compares, per matrix at its best CPU/GPU weight:
  * bicgstab-gpu              pure-GPU baseline (weight-independent)
  * bicgstab-hybrid-async     host-pointer hybrid (CPU/GPU SpMV overlap only)
  * bicgstab-hybrid-async-dp  hybrid + device-pointer-mode dots (this change)

Reports best-of-N loop time for each, the dp-vs-baseline-hybrid speedup, the
dp-hybrid-vs-pure-GPU speedup, and a correctness check (max abs diff of the
solution vectors at a low, convergent iteration count -- should be ~0).

Build first, for the GPU's NATIVE arch (else first-launch JIT pollutes timing):
    cmake -S src -B build -G Ninja -DCMAKE_CUDA_ARCHITECTURES=86
    cmake --build build --target bicgstab-gpu bicgstab-hybrid-async bicgstab-hybrid-async-dp

Usage:
    micromamba run -n octave python tools/python/compare_devptr_hybrid.py \
        [--iters 1000] [--repeats 3] [--check-iters 30]

Default weights are the per-matrix optima found in docs/dot-product-profiling.md.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MTX_DIR = Path("/matrices")
IS_GPU = ROOT / "data/is_gpu/g2_2.txt"

# Per-matrix best CPU/GPU weight (gpu-cpu partition dir) for the BiCGStab hybrid.
BEST_WEIGHT = {"cage11": "w2720", "cage12": "w800", "rma10": "w400"}


def paths(m):
    return (MTX_DIR / f"{m}.mtx",
            ROOT / f"data/matrices/{m}/in/X_init.txt",
            ROOT / f"data/matrices/{m}/in/B.txt")


def run(cmd):
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, cwd=ROOT)
    if p.returncode != 0:
        sys.stderr.write(f"FAILED: {' '.join(map(str, cmd))}\n{p.stderr[-400:]}\n")
        return None, None
    t = re.search(r"^spmv\s*:\s*([-+0-9.eE]+)", p.stdout, re.M)
    r = re.search(r"^relative_residual\s*:\s*([-+0-9.eEnNaA]+)", p.stdout, re.M)
    return (float(t.group(1)) if t else None), (r.group(1) if r else None)


def gpu_cmd(m, out, iters):
    mtx, x, b = paths(m)
    return ["build/bicgstab-gpu", "-m", mtx, "-x", x, "-y", b, "-o", out, "-n", iters]


def hybrid_cmd(binname, m, w, out, iters):
    mtx, x, b = paths(m)
    part = ROOT / f"data/matrices/{m}/in/part/gpu-cpu/{w}_2_i1.part"
    return [f"build/{binname}", "-m", mtx, "-x", x, "-y", b, "-p", part,
            "-g", IS_GPU, "-o", out, "-n", iters]


def best_of(cmd_fn, repeats):
    best = None
    for _ in range(repeats):
        t, _ = run(cmd_fn())
        if t is not None and (best is None or t < best):
            best = t
    return best


def max_abs_diff(f1, f2):
    a = Path(f1).read_text().split()
    b = Path(f2).read_text().split()
    return max(abs(float(x) - float(y)) for x, y in zip(a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", default="1000")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--check-iters", default="30",
                    help="iteration count for the correctness check (must converge, stay finite)")
    ap.add_argument("--matrices", default="cage11,cage12,rma10")
    args = ap.parse_args()

    for b in ("bicgstab-gpu", "bicgstab-hybrid-async", "bicgstab-hybrid-async-dp"):
        if not (ROOT / "build" / b).exists():
            sys.exit(f"Missing build/{b}. Build first (see header).")

    print(f"{'matrix':<8} {'weight':<6} {'gpu':>9} {'hybrid':>9} {'hybrid_dp':>10} "
          f"{'dp/hyb':>8} {'dp/gpu':>8}  {'max|dX|':>9}")
    for m in args.matrices.split(","):
        w = BEST_WEIGHT.get(m)
        if w is None:
            sys.stderr.write(f"skip {m}: no best weight known\n"); continue
        if not paths(m)[0].exists():
            sys.stderr.write(f"skip {m}: matrix missing\n"); continue

        # correctness at a convergent (finite) iteration count
        _, rb = run(hybrid_cmd("bicgstab-hybrid-async", m, w, "/tmp/hyb_chk.txt", args.check_iters))
        _, rd = run(hybrid_cmd("bicgstab-hybrid-async-dp", m, w, "/tmp/hdp_chk.txt", args.check_iters))
        dx = max_abs_diff("/tmp/hyb_chk.txt", "/tmp/hdp_chk.txt")

        # timing, best-of-N
        tg = best_of(lambda: gpu_cmd(m, "/tmp/g.txt", args.iters), args.repeats)
        th = best_of(lambda: hybrid_cmd("bicgstab-hybrid-async", m, w, "/tmp/h.txt", args.iters), args.repeats)
        td = best_of(lambda: hybrid_cmd("bicgstab-hybrid-async-dp", m, w, "/tmp/hd.txt", args.iters), args.repeats)

        print(f"{m:<8} {w:<6} {tg:>9.4f} {th:>9.4f} {td:>10.4f} "
              f"{th/td:>7.3f}x {tg/td:>7.3f}x  {dx:>9.2e}")
        if rb != rd:
            sys.stderr.write(f"  WARN {m}: residual mismatch baseline={rb} dp={rd}\n")


if __name__ == "__main__":
    main()
