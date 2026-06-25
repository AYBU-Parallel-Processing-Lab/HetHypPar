#!/usr/bin/env python3
"""
Merge the PaToH-fill sweep results into the canonical big_benchmark results.

The original run had no PaToH rows for the 4 large matrices (patpart segfaulted
on >2.1M rows). After regenerating those partitions with a raised stack and
sweeping their PaToH half, this appends the new patoh rows, recomputes
speedup_vs_gpu from each matrix's existing gpu baseline, and rewrites
results.tsv (backing up the pre-merge version).

Run: micromamba run -n octave python tools/python/merge_patoh_fill.py
"""
import shutil
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
ORIG = ROOT / "data/big_benchmark/results.tsv"
FILL = ROOT / "data/big_benchmark_patoh_fill/results.tsv"
KEY  = ["matrix", "solver", "part_src", "weight", "threads"]

def main():
    do = pd.read_csv(ORIG, sep="\t")
    df = pd.read_csv(FILL, sep="\t")
    new = df[df["part_src"] == "patoh"].copy()          # only the newly-recovered rows
    print(f"original rows={len(do)}  fill patoh rows={len(new)}")

    both = pd.concat([do, new], ignore_index=True)
    both = both.drop_duplicates(subset=KEY, keep="last")  # fill wins on any collision

    # recompute speedup_vs_gpu from each matrix's min gpu spmv_time
    g = both[(both["solver"] == "gpu") & both["spmv_time"].notna()]
    gpu = g.groupby("matrix")["spmv_time"].min().to_dict()
    def sp(r):
        if pd.notna(r["spmv_time"]) and r["matrix"] in gpu:
            return round(gpu[r["matrix"]] / r["spmv_time"], 3)
        return ""
    both["speedup_vs_gpu"] = both.apply(sp, axis=1)

    both = both.sort_values(["matrix", "solver", "part_src", "weight"]).reset_index(drop=True)

    bak = ORIG.with_name("results_pre_patoh_fill.tsv")
    if not bak.exists():
        shutil.copy(ORIG, bak)
        print(f"backed up original -> {bak.name}")
    both.to_csv(ORIG, sep="\t", index=False)
    print(f"merged rows={len(both)} -> {ORIG}")

    # show the now-filled PaToH-vs-naive for the recovered matrices
    print("\n=== PaToH vs naive on the recovered matrices ===")
    hyb = ["hybrid-async", "hybrid-async-dp", "hybrid-dist-dp", "hybrid-dist-pipelined", "mpi-gpu"]
    for m in new["matrix"].unique():
        print(f"\n{m}:")
        for s in hyb:
            sub = both[(both.matrix == m) & (both.solver == s) & both.speedup_vs_gpu.astype(str).str.len().gt(0)]
            sub = sub.copy(); sub["speedup_vs_gpu"] = pd.to_numeric(sub["speedup_vs_gpu"], errors="coerce")
            pa = sub[sub.part_src == "patoh"]["speedup_vs_gpu"].max()
            na = sub[sub.part_src == "naive"]["speedup_vs_gpu"].max()
            if pd.notna(pa) and pd.notna(na):
                print(f"  {s:<22} patoh {pa:.2f}x  naive {na:.2f}x  ratio {pa/na:.2f}")

if __name__ == "__main__":
    main()
