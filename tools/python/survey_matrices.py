#!/usr/bin/env python3
"""
Survey all matrices in /matrices: read each MatrixMarket header (cheap — first
few lines only) and record dimensions, field (real/pattern/...), symmetry, and
the SuiteSparse 'kind:' tag (domain). Used to pick a principled matrix set for
the benchmark expansion.

Run: micromamba run -n octave python tools/python/survey_matrices.py
Writes data/matrix_survey.tsv and prints a summary (counts by size bucket & kind).
"""
import re, sys
from pathlib import Path

MDIR = Path("/matrices")
OUT  = Path(__file__).resolve().parent.parent.parent / "data/matrix_survey.tsv"
VRAM_NNZ = 130_000_000   # rough fit for pure-GPU on the 8 GB card

def header(path):
    field = symm = kind = ""
    rows = cols = nnz = None
    with open(path, errors="ignore") as f:
        first = f.readline()
        m = re.match(r"%%MatrixMarket\s+matrix\s+coordinate\s+(\S+)\s+(\S+)", first, re.I)
        if m:
            field, symm = m.group(1).lower(), m.group(2).lower()
        line = first
        while line.startswith("%"):
            km = re.search(r"%\s*kind:\s*(.+)", line, re.I)
            if km:
                kind = km.group(1).strip()
            line = f.readline()
            if not line:
                return None
        try:
            rows, cols, nnz = (int(x) for x in line.split()[:3])
        except Exception:
            return None
    return rows, cols, nnz, field, symm, kind

def main():
    rows_out = []
    for p in sorted(MDIR.glob("*.mtx")):
        h = header(p)
        if not h:
            continue
        r, c, nz, field, symm, kind = h
        rows_out.append((p.stem, r, c, nz, field, symm, kind))
    with open(OUT, "w") as f:
        f.write("matrix\trows\tcols\tnnz\tfield\tsymmetry\tkind\n")
        for t in rows_out:
            f.write("\t".join(str(x) for x in t) + "\n")
    print(f"wrote {OUT}  ({len(rows_out)} matrices)\n")

    sq = [t for t in rows_out if t[1] == t[2]]            # square only (solver needs square)
    fit = [t for t in sq if t[3] <= VRAM_NNZ]
    print(f"total matrices: {len(rows_out)}   square: {len(sq)}   square & fit-VRAM(nnz<={VRAM_NNZ//10**6}M): {len(fit)}")

    print("\nsize buckets (square, fit-VRAM) by nnz:")
    buckets = [(0,1e6,"<1M"),(1e6,5e6,"1-5M"),(5e6,2e7,"5-20M"),(2e7,6e7,"20-60M"),(6e7,VRAM_NNZ,"60-130M")]
    for lo,hi,lbl in buckets:
        n=sum(1 for t in fit if lo<=t[3]<hi)
        print(f"  {lbl:<8} {n}")

    print("\ntop 'kind' (domain) tags among square fit-VRAM matrices:")
    from collections import Counter
    cc = Counter((t[6] or "(none)") for t in fit)
    for k,n in cc.most_common(20):
        print(f"  {n:>4}  {k}")

    print("\nsymmetry among square fit-VRAM (symmetric ones may need expansion / can crash solver):")
    cs = Counter(t[5] for t in fit)
    for k,n in cs.most_common():
        print(f"  {n:>4}  {k}")

if __name__ == "__main__":
    main()
