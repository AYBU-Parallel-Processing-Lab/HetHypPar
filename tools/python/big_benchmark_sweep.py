#!/usr/bin/env python3
"""
Big cross-product benchmark: every solver variant x every gpu-cpu weight x
{PaToH, naive} partitions, across a matrix list. Extends variant_speedup_sweep
with a partition-source dimension so we can ask: does hypergraph (cut-minimizing)
partitioning beat a cut-blind naive nnz-balance split?

Run (from project root):
    micromamba run -n octave python tools/python/big_benchmark_sweep.py m1 m2 ...
Env: ITERS (default 1000), REPS (2), WEIGHTS (comma w-list; default all w*_2),
     SOLVERS (comma names; default all), PARTSRCS (default patoh,naive),
     OUT (default data/big_benchmark).

Partition-using solvers run for each (part_src, weight); single-process solvers
run once per matrix. Timing = solver `spmv` line (loop wall time), best-of-REPS,
best OMP threads. Decimals forced to dots (LC_ALL=C).

Outputs under OUT/:
    results.tsv    -- long: matrix, solver, part_src, weight, threads, time, residual, speedup_vs_gpu, status
    summary.md     -- best-speedup pivot per solver, and a PaToH-vs-naive delta table
"""
import os, re, subprocess, sys
from pathlib import Path

ROOT  = Path(__file__).resolve().parent.parent.parent
BUILD = ROOT / "build"
ISGPU = ROOT / "data/is_gpu/g2_2.txt"
OUT   = Path(os.environ.get("OUT", ROOT / "data/big_benchmark")); OUT.mkdir(parents=True, exist_ok=True)

# iterations scale by matrix size: big matrices use fewer iters (loop time scales
# ~linearly, so per-iter speedup is unaffected) to tame the pure-CPU/MPI long tail.
ITERS_BIG   = int(os.environ.get("ITERS_BIG", "200"))
ITERS_SMALL = int(os.environ.get("ITERS_SMALL", "1000"))
NNZ_THRESH  = int(os.environ.get("NNZ_THRESH", "15000000"))
REPS        = int(os.environ.get("REPS", "2"))
_NNZ = {}

def iters_for(m):
    if m not in _NNZ:
        with open(f"/matrices/{m}.mtx") as f:
            line = f.readline()
            while line.startswith("%"): line = f.readline()
            _NNZ[m] = int(line.split()[2])
    return ITERS_BIG if _NNZ[m] > NNZ_THRESH else ITERS_SMALL
PARTSRCS = os.environ.get("PARTSRCS", "patoh,naive").split(",")
PART_DIR = {"patoh": "gpu-cpu", "naive": "gpu-cpu-naive"}

# kind: single (no -p/-g) | hybrid (-p -g) | mpi (mpirun, -p) | mpigpu (mpirun, -p -g)
ALL_SOLVERS = [
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
_want = os.environ.get("SOLVERS")
SOLVERS = [s for s in ALL_SOLVERS if (not _want or s["name"] in _want.split(","))]

SPMV_RE = re.compile(r"^spmv\s*:\s*([\d.eE+-]+)", re.M)
RES_RE  = re.compile(r"relative_residual\s*:\s*([\d.eE+-]+)", re.M)

def mpaths(m):
    base = ROOT / f"data/matrices/{m}/in"
    return dict(mtx=f"/matrices/{m}.mtx", x=str(base/"X_init.txt"), y=str(base/"B.txt"))

def weights_for(m, src):
    d = ROOT / f"data/matrices/{m}/in/part/{PART_DIR[src]}"
    want = set(os.environ.get("WEIGHTS", "").split(",")) - {""}
    ws = []
    for f in sorted(d.glob("*_2_i1.part")):
        mt = re.match(r"(w\d+)_2_i1\.part", f.name)
        if not mt: continue
        if want and mt.group(1) not in want: continue
        ws.append((mt.group(1), str(f)))
    return sorted(ws, key=lambda t: int(t[0][1:]))

def build_cmd(s, m, part):
    p = mpaths(m)
    common = ["-m", p["mtx"], "-x", p["x"], "-y", p["y"], "-o", "/tmp/bb_out.txt", "-n", str(iters_for(m))]
    b = str(BUILD / s["bin"])
    if s["kind"] == "single": return [b] + common
    if s["kind"] == "hybrid": return [b] + common + ["-p", part, "-g", str(ISGPU)]
    mpirun = ["mpirun", "--oversubscribe", "-bind-to", "none", "-n", "2", b]
    if s["kind"] == "mpi":    return mpirun + common + ["-p", part]
    if s["kind"] == "mpigpu": return mpirun + common + ["-p", part, "-g", str(ISGPU)]
    raise ValueError(s["kind"])

def run_once(cmd, env):
    e = dict(os.environ); e.update(env); e["LC_ALL"] = "C"
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=e)
    except subprocess.TimeoutExpired:
        return None, None, "timeout"
    ms = SPMV_RE.search(r.stdout)
    if not ms: return None, None, f"parsefail(rc={r.returncode})"
    mr = RES_RE.search(r.stdout); res = float("nan")
    if mr:
        try: res = float(mr.group(1))
        except ValueError: res = float("nan")
    return float(ms.group(1)), res, "ok"

def best(s, m, part):
    bt = bres = None; bnt = 0; status = "skip"
    for nt in s["threads"]:
        env = dict(s["env"]); env["OMP_NUM_THREADS"] = str(nt)
        for _ in range(REPS):
            t, res, st = run_once(build_cmd(s, m, part), env)
            if t is not None and (bt is None or t < bt):
                bt, bres, bnt, status = t, res, nt, "ok"
            elif status != "ok":
                status = st
    return bt, bres, bnt, status

def main():
    matrices = sys.argv[1:]
    if not matrices:
        print(__doc__); sys.exit(1)
    rows = []  # matrix, solver, part_src, weight, threads, time, residual, status
    for m in matrices:
        print(f"\n######## {m} ########", file=sys.stderr, flush=True)
        for s in SOLVERS:
            if not s["weighted"]:
                t, res, nt, st = best(s, m, None)
                rows.append([m, s["name"], "-", "-", nt, t, res, st])
                print(f"  {s['name']:<22} {'-':<6} {'-':<6} " +
                      (f"{t:.4f}s" if t else f"FAIL({st})"), file=sys.stderr, flush=True)
            else:
                for src in PARTSRCS:
                    for w, part in weights_for(m, src):
                        t, res, nt, st = best(s, m, part)
                        rows.append([m, s["name"], src, w, nt, t, res, st])
                        print(f"  {s['name']:<22} {src:<6} {w:<6} " +
                              (f"{t:.4f}s" if t else f"FAIL({st})"), file=sys.stderr, flush=True)
        _write(rows, matrices)  # checkpoint after each matrix
    print(f"\nWrote {OUT/'results.tsv'} and {OUT/'summary.md'}", file=sys.stderr)
    print((OUT/"summary.md").read_text())

def _write(rows, matrices):
    gpu = {}
    for r in rows:
        if r[1] == "gpu" and r[5]:
            gpu[r[0]] = r[5]
    with open(OUT/"results.tsv", "w") as f:
        f.write("matrix\tsolver\tpart_src\tweight\tthreads\tspmv_time\trelative_residual\tspeedup_vs_gpu\tstatus\n")
        for (m, sn, src, w, nt, t, res, st) in rows:
            sp = f"{gpu[m]/t:.3f}" if (t and gpu.get(m)) else ""
            f.write(f"{m}\t{sn}\t{src}\t{w}\t{nt}\t{t if t else ''}\t{res if t else ''}\t{sp}\t{st}\n")
    # summary: best speedup per (solver, matrix) and patoh-vs-naive best per hybrid
    seen_m = [m for m in matrices if any(r[0] == m for r in rows)]
    with open(OUT/"summary.md", "w") as f:
        f.write(f"# Big benchmark — speedup vs single-GPU `bicgstab-gpu` "
                f"(iters: {ITERS_SMALL} small / {ITERS_BIG} for nnz>{NNZ_THRESH//10**6}M, best-of-{REPS})\n\n")
        f.write("## Best speedup per solver (across weights & partition source)\n\n")
        f.write("| solver | " + " | ".join(seen_m) + " |\n|" + "---|"*(len(seen_m)+1) + "\n")
        for s in SOLVERS:
            cells = []
            for m in seen_m:
                cand = [gpu[m]/r[5] for r in rows if r[0]==m and r[1]==s["name"] and r[5] and gpu.get(m)]
                cells.append(f"{max(cand):.2f}×" if cand else "—")
            f.write(f"| `{s['name']}` | " + " | ".join(cells) + " |\n")
        # PaToH vs naive: best speedup each, per hybrid solver, per matrix
        f.write("\n## PaToH vs naive (best speedup each) — does cut-minimization help?\n\n")
        hyb = [s["name"] for s in SOLVERS if s["weighted"]]
        f.write("| matrix | solver | PaToH | naive | PaToH/naive |\n|---|---|---|---|---|\n")
        for m in seen_m:
            for sn in hyb:
                pa = [gpu[m]/r[5] for r in rows if r[0]==m and r[1]==sn and r[2]=="patoh" and r[5] and gpu.get(m)]
                na = [gpu[m]/r[5] for r in rows if r[0]==m and r[1]==sn and r[2]=="naive" and r[5] and gpu.get(m)]
                if pa and na:
                    f.write(f"| {m} | `{sn}` | {max(pa):.2f}× | {max(na):.2f}× | {max(pa)/max(na):.2f}× |\n")

if __name__ == "__main__":
    main()
