#!/usr/bin/env python3
"""
Prepare matrices for the hybrid benchmark. For each matrix, idempotently makes:
  1. input vectors  (process_matrixi.m: X_target=[1..n], X_init=1, B=A*X_target)
  2. PaToH 2-rank gpu-cpu partitions  (patpart, one per data/weights/gpu-cpu/w*_2.txt)
  3. naive contiguous-nnz partitions  (gen_naive_partition.py)

Run (octave env so `octave` + `python` resolve):
    micromamba run -n octave python tools/python/prep_big_matrix.py <name> [<name> ...]

Skips any step whose outputs already exist. Big matrices: vector gen reads the
full .mtx in Octave (slow/RAM-heavy for multi-GB files) and PaToH can take a
while on tens-of-millions of nonzeros.
"""
import subprocess, sys, re, os, resource
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent.parent
WDIR    = ROOT / "data/weights/gpu-cpu"
PATPART = Path.home() / ".local/bin/patpart"

def _raise_stack():
    # patpart's CalcPartVec uses a stack VLA `int nweights[n]`; the 8 MB default
    # stack overflows (SIGSEGV) above ~2.1M rows. Raise the child's stack limit.
    soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
    want = 1 << 30  # 1 GiB
    new = want if hard == resource.RLIM_INFINITY else min(want, hard)
    resource.setrlimit(resource.RLIMIT_STACK, (new, hard))

def have_vectors(name):
    d = ROOT / f"data/matrices/{name}/in"
    return (d / "X_init.txt").exists() and (d / "B.txt").exists()

def gen_vectors(name):
    print(f"[{name}] generating vectors (octave process_matrixi)...", flush=True)
    subprocess.run(["octave", "--no-gui", "--eval",
                    f"addpath('tools/scripts'); process_matrixi('{name}')"],
                   cwd=ROOT, check=True)

def gen_patoh(name):
    mtx  = f"/matrices/{name}.mtx"
    outd = ROOT / f"data/matrices/{name}/in/part/gpu-cpu"
    outd.mkdir(parents=True, exist_ok=True)
    wfiles = sorted(WDIR.glob("w*_2.txt"), key=lambda p: int(re.match(r"w(\d+)", p.name).group(1)))
    for wf in wfiles:
        w = re.match(r"w(\d+)", wf.name).group(1)
        out = outd / f"w{w}_2_i1.part"
        log = outd / f"w{w}_2_i1.log"
        if out.exists():
            continue
        print(f"[{name}] patpart w{w} (2-rank, imbal 1%)...", flush=True)
        try:
            subprocess.run([str(PATPART), mtx, "2", str(wf), "1", "1", str(out), str(log)],
                           check=True, timeout=7200, preexec_fn=_raise_stack)
        except Exception as e:
            print(f"[{name}] patpart w{w} FAILED: {e}", flush=True)

def gen_naive(name):
    print(f"[{name}] naive nnz-balance partitions...", flush=True)
    subprocess.run([sys.executable, str(ROOT / "tools/python/gen_naive_partition.py"), name],
                   cwd=ROOT, check=True)

def main():
    args = sys.argv[1:]
    patoh_only = "--patoh-only" in args          # only (re)generate PaToH partitions
    names = [a for a in args if not a.startswith("-")]
    if not names:
        print(__doc__); sys.exit(1)
    for name in names:
        try:
            if not Path(f"/matrices/{name}.mtx").exists():
                print(f"SKIP {name}: /matrices/{name}.mtx missing", flush=True); continue
            if patoh_only:
                gen_patoh(name)
                print(f"[{name}] PATOH DONE\n", flush=True); continue
            if have_vectors(name):
                print(f"[{name}] vectors present, skipping octave", flush=True)
            else:
                gen_vectors(name)
            gen_patoh(name)
            gen_naive(name)
            print(f"[{name}] PREP DONE\n", flush=True)
        except Exception as e:
            print(f"[{name}] PREP FAILED: {e} — continuing\n", flush=True)

if __name__ == "__main__":
    main()
