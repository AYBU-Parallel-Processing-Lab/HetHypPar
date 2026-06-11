#!/usr/bin/env python3
"""
profile_dot_breakdown.py -- measure where BiCGStab / SpMV spend their time,
with a focus on the dot-product contribution (the professor's question).

For each (matrix, weight) it runs:
  * bicgstab-hybrid-async  : CPU+GPU hybrid solver (the 1.06x winner)
  * bicgstab-gpu           : pure-GPU solver (weight-independent, run once/matrix)
  * spmv-hybrid-async      : pure SpMV breakdown (gpu_full/slice/cpu_slice/hybrid)

Each solver is run TWICE per repeat:
  - clean    (HHP_PROFILE unset)  -> true loop_total / speedup  (no sync perturbation)
  - profiled (HHP_PROFILE=1)      -> dot / spmv / vec breakdown  (sync barriers added)
The clean loop_total is the honest number; the profiled run's *ratios* tell us
how the loop time splits across categories. We report both.

Reproducibility / setup
------------------------
  - Build first:   cmake --build build --target bicgstab-hybrid-async bicgstab-gpu spmv-hybrid-async
  - Raw matrices:  /matrices/<name>.mtx  (shared server)
  - Inputs:        data/matrices/<name>/in/{X_init.txt,B.txt}
  - Partitions:    data/matrices/<name>/in/part/gpu-cpu/<weight>_2_i1.part
  - is_gpu file:   data/is_gpu/g2_2.txt   (rank0=GPU, rank1=CPU)

Usage
-----
  micromamba run -n octave python tools/python/profile_dot_breakdown.py \
      [--iters 1000] [--repeats 3] [--matrices cage11,cage12,rma10] \
      [--weights w800,w1200,w2000,w2720] [--outdir data/profile_dot]

Outputs
-------
  <outdir>/breakdown.tsv   one row per (matrix, solver, weight)
  prints a human-readable summary table to stdout
"""
import argparse
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

# Resolve project root from this file: tools/python/<this> -> root
ROOT = Path(__file__).resolve().parents[2]

MTX_DIR = Path("/matrices")
IS_GPU = ROOT / "data/is_gpu/g2_2.txt"


def parse_kv(stdout: str) -> dict:
    """Extract all 'key : value' float lines into a dict."""
    out = {}
    for line in stdout.splitlines():
        m = re.match(r"^\s*([A-Za-z0-9_]+)\s*:\s*([-+0-9.eE]+)\s*$", line)
        if m:
            try:
                out[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
    return out


def run(cmd: list, profile: bool) -> dict:
    env = dict(os.environ)
    if profile:
        env["HHP_PROFILE"] = "1"
    else:
        env.pop("HHP_PROFILE", None)
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT)
    if p.returncode != 0:
        sys.stderr.write(f"FAILED ({p.returncode}): {' '.join(cmd)}\n{p.stderr[-800:]}\n")
        return {}
    return parse_kv(p.stdout)


def median_runs(cmd: list, profile: bool, repeats: int, keys: list) -> dict:
    """Run cmd `repeats` times, return per-key median of the requested keys."""
    samples = {k: [] for k in keys}
    for _ in range(repeats):
        d = run(cmd, profile)
        for k in keys:
            if k in d:
                samples[k].append(d[k])
    return {k: (statistics.median(v) if v else float("nan")) for k, v in samples.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--matrices", default="cage11,cage12,rma10")
    ap.add_argument("--weights", default="w800,w1200,w2000,w2720")
    ap.add_argument("--outdir", default="data/profile_dot")
    args = ap.parse_args()

    matrices = args.matrices.split(",")
    weights = args.weights.split(",")
    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    bin_hybrid = ROOT / "build/bicgstab-hybrid-async"
    bin_gpu = ROOT / "build/bicgstab-gpu"
    bin_spmv = ROOT / "build/spmv-hybrid-async"
    for b in (bin_hybrid, bin_gpu, bin_spmv):
        if not b.exists():
            sys.exit(f"Missing binary {b}. Build first:\n  cmake --build build")

    prof_keys = ["PROFILE_loop_total", "PROFILE_dot", "PROFILE_spmv", "PROFILE_vec",
                 "PROFILE_dot_pct", "PROFILE_spmv_pct", "PROFILE_vec_pct"]
    rows = []

    for m in matrices:
        mtx = MTX_DIR / f"{m}.mtx"
        xin = ROOT / f"data/matrices/{m}/in/X_init.txt"
        bin_ = ROOT / f"data/matrices/{m}/in/B.txt"
        if not mtx.exists():
            sys.stderr.write(f"skip {m}: matrix {mtx} missing\n")
            continue

        # --- pure GPU: weight-independent, run once per matrix ---
        gpu_base = [str(bin_gpu), "-m", str(mtx), "-x", str(xin), "-y", str(bin_),
                    "-o", "/tmp/Xg.txt", "-n", str(args.iters)]
        gpu_clean = median_runs(gpu_base, False, args.repeats, ["spmv"])
        gpu_prof = median_runs(gpu_base, True, args.repeats, prof_keys)
        gpu_total = gpu_clean.get("spmv", float("nan"))
        rows.append(dict(
            matrix=m, solver="gpu", weight="-",
            loop_total=gpu_total, speedup_vs_gpu=1.0,
            per_iter_us=1e6 * gpu_total / args.iters,
            dot=gpu_prof.get("PROFILE_dot"), spmv=gpu_prof.get("PROFILE_spmv"),
            vec=gpu_prof.get("PROFILE_vec"),
            dot_pct=gpu_prof.get("PROFILE_dot_pct"),
            spmv_pct=gpu_prof.get("PROFILE_spmv_pct"),
            vec_pct=gpu_prof.get("PROFILE_vec_pct"),
        ))
        print(f"[{m}] gpu        total={gpu_total*1e3:8.3f}ms  "
              f"dot={gpu_prof.get('PROFILE_dot_pct'):5.1f}%  "
              f"spmv={gpu_prof.get('PROFILE_spmv_pct'):5.1f}%  "
              f"vec={gpu_prof.get('PROFILE_vec_pct'):5.1f}%")

        for w in weights:
            part = ROOT / f"data/matrices/{m}/in/part/gpu-cpu/{w}_2_i1.part"
            if not part.exists():
                sys.stderr.write(f"skip {m}/{w}: partition {part} missing\n")
                continue

            # --- hybrid BiCGStab ---
            hy_base = [str(bin_hybrid), "-m", str(mtx), "-x", str(xin), "-y", str(bin_),
                       "-p", str(part), "-g", str(IS_GPU), "-o", "/tmp/Xh.txt",
                       "-n", str(args.iters)]
            hy_clean = median_runs(hy_base, False, args.repeats, ["spmv"])
            hy_prof = median_runs(hy_base, True, args.repeats, prof_keys)
            hy_total = hy_clean.get("spmv", float("nan"))
            speedup = gpu_total / hy_total if hy_total and hy_total == hy_total else float("nan")
            rows.append(dict(
                matrix=m, solver="hybrid", weight=w,
                loop_total=hy_total, speedup_vs_gpu=speedup,
                per_iter_us=1e6 * hy_total / args.iters,
                dot=hy_prof.get("PROFILE_dot"), spmv=hy_prof.get("PROFILE_spmv"),
                vec=hy_prof.get("PROFILE_vec"),
                dot_pct=hy_prof.get("PROFILE_dot_pct"),
                spmv_pct=hy_prof.get("PROFILE_spmv_pct"),
                vec_pct=hy_prof.get("PROFILE_vec_pct"),
            ))
            print(f"[{m}] hybrid {w:>6} total={hy_total*1e3:8.3f}ms  "
                  f"speedup={speedup:5.3f}x  "
                  f"dot={hy_prof.get('PROFILE_dot_pct'):5.1f}%  "
                  f"spmv={hy_prof.get('PROFILE_spmv_pct'):5.1f}%  "
                  f"vec={hy_prof.get('PROFILE_vec_pct'):5.1f}%")

            # --- pure SpMV breakdown (self-reporting) ---
            sp_base = [str(bin_spmv), "-m", str(mtx), "-x", str(xin),
                       "-p", str(part), "-g", str(IS_GPU), "-o", "/tmp/Ys.txt",
                       "-n", str(args.iters)]
            sp = median_runs(sp_base, False, args.repeats,
                             ["gpu_full_per_iter", "gpu_slice_per_iter",
                              "cpu_slice_per_iter", "hybrid_per_iter",
                              "speedup_vs_gpu_full"])
            rows.append(dict(
                matrix=m, solver="spmv", weight=w,
                loop_total=sp.get("hybrid_per_iter"),
                speedup_vs_gpu=sp.get("speedup_vs_gpu_full"),
                per_iter_us=1e6 * sp.get("hybrid_per_iter", float("nan")),
                dot=None, spmv=sp.get("hybrid_per_iter"), vec=None,
                dot_pct=None, spmv_pct=None, vec_pct=None,
                gpu_full_us=1e6 * sp.get("gpu_full_per_iter", float("nan")),
                gpu_slice_us=1e6 * sp.get("gpu_slice_per_iter", float("nan")),
                cpu_slice_us=1e6 * sp.get("cpu_slice_per_iter", float("nan")),
            ))
            print(f"[{m}] spmv   {w:>6} hybrid/iter={sp.get('hybrid_per_iter',0)*1e6:8.2f}us  "
                  f"speedup={sp.get('speedup_vs_gpu_full',0):5.3f}x")

    # --- write TSV ---
    cols = ["matrix", "solver", "weight", "loop_total", "per_iter_us",
            "speedup_vs_gpu", "dot", "spmv", "vec",
            "dot_pct", "spmv_pct", "vec_pct",
            "gpu_full_us", "gpu_slice_us", "cpu_slice_us"]
    tsv = outdir / "breakdown.tsv"
    with open(tsv, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(
                "" if r.get(c) is None else
                (f"{r[c]:.6g}" if isinstance(r.get(c), float) else str(r.get(c)))
                for c in cols) + "\n")
    print(f"\nWrote {tsv}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
