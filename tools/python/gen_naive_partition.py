#!/usr/bin/env python3
"""
Generate NAIVE 2-rank (GPU+CPU) partitions that balance nonzeros by the weight
ratio, WITHOUT any cut/halo minimization — the cut-blind baseline to compare
against PaToH hypergraph partitioning.

Scheme: contiguous nnz-balanced block. Rows are walked in natural order; the
first contiguous chunk holding the GPU's nnz quota goes to partition 0 (GPU,
matching is_gpu g2_2 where rank 0 = GPU), the remainder to partition 1 (CPU).
A single boundary, no attempt to minimize shared columns.

Weight ratio: weight file `w<W>_2.txt` contains `<W> 100` => GPU:CPU = W:100, so
the GPU gets W/(W+100) of the total stored nonzeros. We read the same `w*_2.txt`
files PaToH uses so the two partition families target identical nnz fractions.

Usage (from project root):
    python tools/python/gen_naive_partition.py <name>            # all w*_2 weights
    python tools/python/gen_naive_partition.py <name> w600 w1200 # subset
Reads  /matrices/<name>.mtx  and  data/weights/gpu-cpu/w*_2.txt
Writes data/matrices/<name>/in/part/gpu-cpu-naive/w<W>_2_i1.part
       (one partition id per line, n lines — same format as patpart output)
"""
import sys, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
WDIR = ROOT / "data/weights/gpu-cpu"

def row_nnz(mtx_path, ):
    """Single pass: return (n_rows, list of nnz per row) from a MatrixMarket file."""
    with open(mtx_path) as f:
        line = f.readline()
        while line.startswith("%"):
            line = f.readline()
        n, _, nnz = (int(x) for x in line.split())
        counts = [0] * n
        for line in f:
            if not line or line[0] == "%":
                continue
            sp = line.find(" ")
            if sp <= 0:
                continue
            r = int(line[:sp])
            counts[r - 1] += 1
    return n, counts, nnz

def weights(names):
    out = []
    files = sorted(WDIR.glob("w*_2.txt"), key=lambda p: int(re.match(r"w(\d+)_2", p.name).group(1)))
    for f in files:
        w = int(re.match(r"w(\d+)_2", f.name).group(1))
        if names and f"w{w}" not in names:
            continue
        out.append(w)
    return out

def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    name = sys.argv[1]
    want = set(sys.argv[2:])
    mtx = Path(f"/matrices/{name}.mtx")
    if not mtx.exists():
        sys.exit(f"missing matrix {mtx}")
    outdir = ROOT / f"data/matrices/{name}/in/part/gpu-cpu-naive"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[{name}] reading nnz-per-row from {mtx} ...", flush=True)
    n, counts, total = row_nnz(mtx)
    print(f"[{name}] n={n} stored_nnz={total}", flush=True)

    for w in weights(want):
        gpu_frac = w / (w + 100.0)
        target = gpu_frac * total
        part = bytearray(b"0\n" * 0)  # build as text
        lines = []
        cum = 0
        boundary = n  # first row that goes to CPU
        for i in range(n):
            if cum >= target:
                boundary = i
                break
            cum += counts[i]
        # rows [0,boundary) -> GPU (0), [boundary,n) -> CPU (1); guard non-empty
        if boundary <= 0: boundary = 1
        if boundary >= n: boundary = n - 1
        gpu_nnz = sum(counts[:boundary])
        out = outdir / f"w{w}_2_i1.part"
        with open(out, "w") as fo:
            fo.write("\n".join("0" if i < boundary else "1" for i in range(n)))
            fo.write("\n")
        cpu_pct = 100.0 * (total - gpu_nnz) / total
        print(f"  w{w:<4} -> gpu_rows={boundary} cpu_rows={n-boundary} "
              f"cpu_nnz={100*(total-gpu_nnz)/total:.1f}% (target {100*(1-gpu_frac):.1f}%)  {out.name}", flush=True)

if __name__ == "__main__":
    main()
