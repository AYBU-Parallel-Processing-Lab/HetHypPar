#!/usr/bin/env python3
"""
compare_devptr.py -- measure the device-pointer-mode dot optimization.

Compares the pure-GPU BiCGStab baseline (bicgstab-gpu, cuBLAS host-pointer mode,
5 host<->device stalls/iter) against the prototype (bicgstab-gpu-dp, cuBLAS
device-pointer mode, scalar updates in device kernels, 0 stalls/iter).

For each matrix it reports the best-of-N loop time for each binary, their
speedup, and the max abs difference of the solution vectors (correctness:
should be ~0, the math is identical).

IMPORTANT: build the prototype kernels for the GPU's NATIVE arch, else the
sm_75 default makes the first kernel launch JIT-compile *inside* the timed loop
and pollutes the measurement:
    cmake -S src -B build -G Ninja -DCMAKE_CUDA_ARCHITECTURES=86
    cmake --build build --target bicgstab-gpu bicgstab-gpu-dp

Usage:
    micromamba run -n octave python tools/python/compare_devptr.py \
        [--iters 1000] [--repeats 4] [--matrices cage11,cage12,rma10]
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MTX_DIR = Path("/matrices")


def run_loop_time(binary: Path, mtx: Path, x: Path, b: Path, out: Path, iters: int):
    p = subprocess.run(
        [str(binary), "-m", str(mtx), "-x", str(x), "-y", str(b),
         "-o", str(out), "-n", str(iters)],
        capture_output=True, text=True, cwd=ROOT)
    if p.returncode != 0:
        sys.stderr.write(f"FAILED: {binary.name} on {mtx.name}\n{p.stderr[-500:]}\n")
        return None
    m = re.search(r"^spmv\s*:\s*([-+0-9.eE]+)", p.stdout, re.M)
    return float(m.group(1)) if m else None


def best_of(binary, mtx, x, b, out, iters, repeats):
    times = [t for _ in range(repeats)
             if (t := run_loop_time(binary, mtx, x, b, out, iters)) is not None]
    return min(times) if times else None


def max_abs_diff(f1: Path, f2: Path) -> float:
    a = f1.read_text().split()
    b = f2.read_text().split()
    return max(abs(float(x) - float(y)) for x, y in zip(a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--matrices", default="cage11,cage12,rma10")
    args = ap.parse_args()

    base = ROOT / "build/bicgstab-gpu"
    dp = ROOT / "build/bicgstab-gpu-dp"
    for binp in (base, dp):
        if not binp.exists():
            sys.exit(f"Missing {binp}. Build first (see header).")

    print(f"{'matrix':<10} {'baseline(s)':>12} {'devptr(s)':>12} "
          f"{'speedup':>9} {'max|dX|':>10}")
    for m in args.matrices.split(","):
        mtx = MTX_DIR / f"{m}.mtx"
        x = ROOT / f"data/matrices/{m}/in/X_init.txt"
        b = ROOT / f"data/matrices/{m}/in/B.txt"
        if not mtx.exists():
            sys.stderr.write(f"skip {m}: {mtx} missing\n")
            continue
        xb = Path(f"/tmp/X_{m}_base.txt")
        xd = Path(f"/tmp/X_{m}_dp.txt")
        tb = best_of(base, mtx, x, b, xb, args.iters, args.repeats)
        td = best_of(dp, mtx, x, b, xd, args.iters, args.repeats)
        if tb is None or td is None:
            continue
        dx = max_abs_diff(xb, xd)
        print(f"{m:<10} {tb:>12.4f} {td:>12.4f} {tb/td:>8.3f}x {dx:>10.2e}")


if __name__ == "__main__":
    main()
