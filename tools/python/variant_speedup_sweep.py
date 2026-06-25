#!/usr/bin/env python3
"""
Reproducible speedup sweep across all BiCGStab solver variants.

Runs every solver variant on cage12/cage13, sweeping the gpu-cpu partition
weights for the hybrid/MPI variants, and writes a long-format TSV plus a printed
speedup table (vs single-GPU `bicgstab-gpu`).

Why this exists: the variant comparison done interactively this session was never
saved to disk. This script regenerates it reproducibly.

Run (from project root):
    micromamba run -n octave python tools/python/variant_speedup_sweep.py
Env overrides: ITERS (default 1000), REPS (default 2), MATRICES (comma list).

Output:
    data/variant_sweep/results.tsv   -- one row per (matrix, solver, weight, threads)
    data/variant_sweep/speedup.md    -- pivot table, speedup vs gpu
Timing metric: the solver's `spmv` line (loop wall time), best-of-REPS, best OMP
thread count. Decimals forced to dots via LC_ALL=C (solvers print locale-comma).
"""
import os, re, subprocess, sys
from pathlib import Path

ROOT  = Path(__file__).resolve().parent.parent.parent
BUILD = ROOT / "build"
ISGPU = ROOT / "data/is_gpu/g2_2.txt"
OUT   = ROOT / "data/variant_sweep"; OUT.mkdir(exist_ok=True)

MATRICES = os.environ.get("MATRICES", "cage12,cage13").split(",")
ITERS    = int(os.environ.get("ITERS", "1000"))
REPS     = int(os.environ.get("REPS", "2"))

# kind: single (no -p/-g) | hybrid (-p -g) | mpi (mpirun, -p) | mpigpu (mpirun, -p -g)
SOLVERS = [
    dict(name="cpu",                   bin="bicgstab-cpu",                kind="single", weighted=False, threads=[4,8,16], env={}),
    dict(name="gpu",                   bin="bicgstab-gpu",                kind="single", weighted=False, threads=[1],     env={}),
    dict(name="gpu-dp",                bin="bicgstab-gpu-dp",             kind="single", weighted=False, threads=[1],     env={}),
    dict(name="gpu-pipelined",         bin="bicgstab-gpu-pipelined",      kind="single", weighted=False, threads=[1],     env={"HHP_REPLACE":"20"}),
    dict(name="hybrid-async",          bin="bicgstab-hybrid-async",       kind="hybrid", weighted=True,  threads=[1,2,4], env={}),
    dict(name="hybrid-async-dp",       bin="bicgstab-hybrid-async-dp",    kind="hybrid", weighted=True,  threads=[1,2,4], env={}),
    dict(name="hybrid-dist-dp",        bin="bicgstab-hybrid-dist-dp",     kind="hybrid", weighted=True,  threads=[1,2,4], env={}),
    dict(name="hybrid-dist-pipelined", bin="bicgstab-hybrid-dist-pipelined", kind="hybrid", weighted=True, threads=[1,2,4], env={"HHP_REPLACE":"20"}),
    dict(name="mpi",                   bin="bicgstab-mpi",                kind="mpi",    weighted=True,  threads=[1],     env={}),
    dict(name="mpi-gpu",               bin="bicgstab-mpi-gpu",            kind="mpigpu", weighted=True,  threads=[1],     env={}),
]

SPMV_RE = re.compile(r"^spmv\s*:\s*([\d.eE+-]+)", re.M)
RES_RE  = re.compile(r"relative_residual\s*:\s*([\d.eE+-]+)", re.M)

def mpaths(m):
    base = ROOT / f"data/matrices/{m}/in"
    return dict(mtx=f"/matrices/{m}.mtx", x=str(base/"X_init.txt"), y=str(base/"B.txt"))

def weights_for(m):
    d = ROOT / f"data/matrices/{m}/in/part/gpu-cpu"
    ws = []
    for f in sorted(d.glob("*_2_i1.part")):
        mt = re.match(r"(w\d+)_2_i1\.part", f.name)
        if mt: ws.append((mt.group(1), str(f)))
    # sort by numeric weight
    return sorted(ws, key=lambda t: int(t[0][1:]))

def build_cmd(s, m, part):
    p = mpaths(m)
    common = ["-m", p["mtx"], "-x", p["x"], "-y", p["y"], "-o", "/tmp/vs_out.txt", "-n", str(ITERS)]
    b = str(BUILD / s["bin"])
    if s["kind"] == "single":
        return [b] + common
    if s["kind"] == "hybrid":
        return [b] + common + ["-p", part, "-g", str(ISGPU)]
    mpirun = ["mpirun", "--oversubscribe", "-bind-to", "none", "-n", "2", b]
    if s["kind"] == "mpi":
        return mpirun + common + ["-p", part]
    if s["kind"] == "mpigpu":
        return mpirun + common + ["-p", part, "-g", str(ISGPU)]
    raise ValueError(s["kind"])

def run_once(cmd, env):
    e = dict(os.environ); e.update(env); e["LC_ALL"] = "C"
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=e)
    except subprocess.TimeoutExpired:
        return None, None, "timeout"
    ms = SPMV_RE.search(r.stdout)
    if not ms:
        return None, None, f"parsefail(rc={r.returncode})"
    mr = RES_RE.search(r.stdout)
    res = float("nan")
    if mr:
        try: res = float(mr.group(1))   # solvers can print '-' / 'nan' for residual
        except ValueError: res = float("nan")
    return float(ms.group(1)), res, "ok"

def best(s, m, part):
    bt = bres = None; status = "skip"
    for nt in s["threads"]:
        env = dict(s["env"]); env["OMP_NUM_THREADS"] = str(nt)
        for _ in range(REPS):
            t, res, st = run_once(build_cmd(s, m, part), env)
            if t is not None and (bt is None or t < bt):
                bt, bres, bnt, status = t, res, nt, "ok"
            elif status != "ok":
                status = st
    return (bt, bres, (bnt if bt is not None else 0), status)

def main():
    rows = []  # (matrix, solver, weight, threads, time, residual, status)
    for m in MATRICES:
        ws = weights_for(m)
        print(f"\n### {m}  (weights: {[w for w,_ in ws]})", file=sys.stderr)
        for s in SOLVERS:
            targets = ws if s["weighted"] else [("-", None)]
            for w, part in targets:
                t, res, nt, st = best(s, m, part)
                rows.append((m, s["name"], w, nt, t, res, st))
                ts = f"{t:.4f}s" if t else "   -   "
                print(f"  {s['name']:<22} {w:<7} {ts}  res={res:.1e}" if t else
                      f"  {s['name']:<22} {w:<7}   FAIL ({st})", file=sys.stderr)

    # write long TSV
    tsv = OUT / "results.tsv"
    with open(tsv, "w") as f:
        f.write("matrix\tsolver\tweight\tomp_threads\tspmv_time\trelative_residual\tspeedup_vs_gpu\tstatus\n")
        gpu = {m: next((t for (mm, sn, w, nt, t, r, st) in rows if mm == m and sn == "gpu" and t), None) for m in MATRICES}
        for (m, sn, w, nt, t, r, st) in rows:
            sp = (gpu[m] / t) if (t and gpu[m]) else ""
            spS = f"{sp:.3f}" if sp != "" else ""
            f.write(f"{m}\t{sn}\t{w}\t{nt}\t{t if t else ''}\t{r if t else ''}\t{spS}\t{st}\n")

    # pivot: best speedup per (solver, matrix)  +  markdown
    md = OUT / "speedup.md"
    solvers = [s["name"] for s in SOLVERS]
    with open(md, "w") as f:
        f.write(f"# Variant speedup vs single-GPU `bicgstab-gpu`  (iters={ITERS}, best-of-{REPS})\n\n")
        f.write("Best speedup across weights (hybrids/MPI) — see results.tsv for the full per-weight sweep.\n\n")
        f.write("| solver | " + " | ".join(MATRICES) + " |\n")
        f.write("|" + "---|" * (len(MATRICES) + 1) + "\n")
        for sn in solvers:
            cells = []
            for m in MATRICES:
                cand = [(gpu[m] / t) for (mm, s2, w, nt, t, r, st) in rows if mm == m and s2 == sn and t]
                cells.append(f"{max(cand):.2f}×" if cand else "—")
            f.write(f"| `{sn}` | " + " | ".join(cells) + " |\n")
    print(f"\nWrote {tsv}\nWrote {md}", file=sys.stderr)
    print(open(md).read())

if __name__ == "__main__":
    main()
