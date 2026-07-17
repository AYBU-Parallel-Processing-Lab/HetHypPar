#!/usr/bin/env python3
"""
Pick ~50 matrices for the benchmark expansion, stratified by domain (SuiteSparse
'kind') and size, from data/matrix_survey.tsv. Always keeps the 14 already
benchmarked (already prepped). Deterministic (no RNG).

Run: micromamba run -n octave python tools/python/select_matrices.py [TARGET]
Writes data/expansion_matrices.txt (one name/line) and prints the breakdown.
"""
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent.parent
SURVEY = ROOT / "data/matrix_survey.tsv"
OUT = ROOT / "data/expansion_matrices.txt"

EXISTING = ["RM07R", "cage14", "Hamrle3", "rajat30_mc64_5", "circuit5M_dc", "circuit5M",
            "cage15", "ML_Geer", "vas_stokes_1M", "nv2", "Transport", "Freescale2", "dgreen", "ss"]

def main():
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    rows = []
    with open(SURVEY) as f:
        next(f)
        for line in f:
            name, r, c, nnz, field, symm, kind = (line.rstrip("\n").split("\t") + [""] * 7)[:7]
            r, c, nnz = int(r), int(c), int(nnz)
            rows.append(dict(name=name, rows=r, nnz=nnz, field=field, symm=symm, kind=kind or "(none)"))

    # filter: square, general, real-ish, sane size (avoid trivial & >VRAM)
    def ok(m):
        return (m["rows"] == m["rows"] and m["symm"] == "general"
                and m["field"] in ("real", "integer")
                and m["rows"] >= 10_000 and 50_000 <= m["nnz"] <= 130_000_000)
    pool = {m["name"]: m for m in rows if ok(m)}

    chosen = [m for m in EXISTING if m in pool] + [m for m in EXISTING if m not in pool]
    chosen = list(dict.fromkeys(EXISTING))               # keep existing 14 (even if filtered)
    need = max(0, target - len(chosen))

    # candidates = pool minus already-chosen; group by kind
    by_kind = defaultdict(list)
    for name, m in pool.items():
        if name in chosen:
            continue
        by_kind[m["kind"]].append(m)
    # within each kind, sort by nnz and keep size-spread picks first (largest, median, smallest...)
    def spread(lst):
        lst = sorted(lst, key=lambda m: m["nnz"])
        order, lo, hi = [], 0, len(lst) - 1
        take_hi = True
        while lo <= hi:
            order.append(lst[hi] if take_hi else lst[lo])
            if take_hi: hi -= 1
            else: lo += 1
            take_hi = not take_hi
        return order
    queues = {k: spread(v) for k, v in by_kind.items()}
    # round-robin across kinds (largest domains first) until we have `need`
    kinds_by_size = sorted(queues, key=lambda k: -len(queues[k]))
    picked = []
    while len(picked) < need and any(queues.values()):
        for k in kinds_by_size:
            if queues[k] and len(picked) < need:
                picked.append(queues[k].pop(0)["name"])
    chosen += picked

    OUT.write_text("\n".join(chosen) + "\n")
    info = {m["name"]: m for m in rows}
    print(f"selected {len(chosen)} matrices -> {OUT}\n")
    print(f"{'existing 14 kept':<22}: {sum(1 for c in chosen if c in EXISTING)}")
    print(f"{'newly added':<22}: {len(chosen) - 14}\n")
    bk = defaultdict(int); bs = defaultdict(int)
    for c in chosen:
        m = info.get(c, {})
        bk[m.get("kind", "?")] += 1
        nz = m.get("nnz", 0)
        b = "<1M" if nz < 1e6 else "1-5M" if nz < 5e6 else "5-20M" if nz < 2e7 else "20-60M" if nz < 6e7 else ">60M"
        bs[b] += 1
    print("by domain:")
    for k, n in sorted(bk.items(), key=lambda x: -x[1]):
        print(f"  {n:>3}  {k}")
    print("\nby size (nnz):")
    for b in ["<1M", "1-5M", "5-20M", "20-60M", ">60M"]:
        print(f"  {bs.get(b,0):>3}  {b}")

if __name__ == "__main__":
    main()
