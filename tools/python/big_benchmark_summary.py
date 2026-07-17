#!/usr/bin/env python3
"""
Regenerate data/big_benchmark/summary.md from results.parquet, carrying BOTH
baselines (vs old bicgstab-gpu and vs bicgstab-gpu-dp) plus the PaToH-vs-naive
comparison.

Run: micromamba run -n octave python tools/python/big_benchmark_summary.py
"""
from pathlib import Path
import pandas as pd

import os
ROOT = Path(__file__).resolve().parent.parent.parent
PARQ = Path(os.environ.get("PARQ", ROOT / "data/big_benchmark/results.parquet"))
OUT  = Path(os.environ.get("SUMMARY_OUT", ROOT / "data/big_benchmark/summary.md"))

SOLVERS = ["gpu", "gpu-dp", "hybrid-async-dp", "hybrid-dist-dp", "hybrid-async",
           "hybrid-dist-pipelined", "gpu-pipelined", "mpi-gpu", "mpi", "cpu"]
HYB = ["hybrid-async", "hybrid-async-dp", "hybrid-dist-dp", "hybrid-dist-pipelined", "mpi-gpu"]

def best_pivot(df, col):
    b = df.groupby(["matrix", "solver"])[col].max().reset_index()
    return b.pivot(index="solver", columns="matrix", values=col)

def main():
    df = pd.read_parquet(PARQ)
    for c in ("matrix", "solver", "part_src"):
        df[c] = df[c].astype(str)
    ok = df[df["ok"]].copy()
    mats = [m for m in sorted(ok["matrix"].unique())]
    usable = [m for m in mats if ok[ok.matrix == m]["speedup_vs_gpu"].notna().any()]

    lines = []
    lines.append("# Big benchmark — summary\n")
    lines.append("Best `spmv` loop time (best-of-2, best OMP threads) across weights & partition "
                 "source. Two baselines: **old `bicgstab-gpu`** and **`bicgstab-gpu-dp`** "
                 "(the device-pointer pure-GPU solver — the fairer 'is the CPU split worth it?' baseline).\n")

    for col, title in [("speedup_vs_gpu", "Speedup vs old `bicgstab-gpu`"),
                       ("speedup_vs_gpudp", "Speedup vs `bicgstab-gpu-dp`")]:
        piv = best_pivot(ok, col)
        lines.append(f"\n## {title}\n")
        lines.append("| solver | " + " | ".join(usable) + " |")
        lines.append("|" + "---|" * (len(usable) + 1))
        for s in SOLVERS:
            if s not in piv.index:
                continue
            cells = []
            for m in usable:
                v = piv.loc[s, m] if m in piv.columns else None
                cells.append(f"{v:.2f}×" if pd.notna(v) else "—")
            lines.append(f"| `{s}` | " + " | ".join(cells) + " |")

    # PaToH vs naive
    lines.append("\n## PaToH vs naive — best speedup each (vs old-gpu), ratio = patoh/naive\n")
    lines.append("| matrix | solver | PaToH | naive | ratio |")
    lines.append("|---|---|---|---|---|")
    ratios = {s: [] for s in HYB}
    for m in usable:
        for s in HYB:
            sub = ok[(ok.matrix == m) & (ok.solver == s)]
            pa = sub[sub.part_src == "patoh"]["speedup_vs_gpu"].max()
            na = sub[sub.part_src == "naive"]["speedup_vs_gpu"].max()
            if pd.notna(pa) and pd.notna(na):
                lines.append(f"| {m} | `{s}` | {pa:.2f}× | {na:.2f}× | {pa/na:.2f} |")
                ratios[s].append(pa / na)
    lines.append("\n### Mean PaToH/naive ratio per solver\n")
    lines.append("| solver | mean ratio | n matrices |")
    lines.append("|---|---|---|")
    for s in HYB:
        r = ratios[s]
        if r:
            lines.append(f"| `{s}` | {sum(r)/len(r):.3f} | {len(r)} |")

    # failures
    failed_full = [m for m in mats if not ok[ok.matrix == m]["speedup_vs_gpu"].notna().any()]
    all_mats = sorted(df["matrix"].astype(str).unique())
    fully_dead = [m for m in all_mats if m not in usable]
    lines.append(f"\n## Notes\n")
    lines.append(f"- Usable matrices: {len(usable)}. Fully-failed (excluded): "
                 f"{fully_dead if fully_dead else 'none'} (solver-level crash).")
    lines.append("- `speedup_vs_gpudp` is the honest 'does the heterogeneous CPU split beat just "
                 "running pure GPU device-pointer?' baseline.")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}  ({len(usable)} usable matrices)")

if __name__ == "__main__":
    main()
