#!/usr/bin/env python3
"""
Dolan-More performance profiles from data/big_benchmark/results.parquet.

Construction (per the standard performance profile):
  1. best execution time per (matrix, tool)  -> table, rows=matrices, cols=tools
  2. divide each row by that row's minimum   -> performance ratio r>=1
     (r=1 means this tool was the fastest on that matrix)
  3. sort each tool's ratios ascending; plot the cumulative fraction of matrices
     (y) against the ratio (x) as a step curve starting at the bottom-left.
  A tool fastest on 90% of matrices rises vertically at x=1 to y=0.9, then drifts
  right; the higher/more-left a curve, the better the tool.

Plot 1: per-solver speed profile (competitive GPU/hybrid solvers).
Plot 2: PaToH vs naive partitioning, over (matrix, hybrid-solver) cases.

Run: micromamba run -n octave python tools/python/perf_profile.py
Outputs: docs/perf-profile-solvers.{png,svg}, docs/perf-profile-patoh-naive.{png,svg}
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

import os
ROOT = Path(__file__).resolve().parent.parent.parent
PARQ = Path(os.environ.get("PARQ", ROOT / "data/big_benchmark/results.parquet"))
DOCS = ROOT / "docs"

# competitive solvers (cpu/mpi are ~10x slower and would blow out the x-axis)
SOLVERS = ["gpu", "gpu-dp", "hybrid-async-dp", "hybrid-dist-dp",
           "hybrid-async", "hybrid-dist-pipelined", "gpu-pipelined"]
COLORS = {"gpu": "#888888", "gpu-dp": "#1a1a1a", "hybrid-async-dp": "#2f7d32",
          "hybrid-dist-dp": "#2c5e9e", "hybrid-async": "#7fb07a",
          "hybrid-dist-pipelined": "#b03030", "gpu-pipelined": "#9a6a1e"}
MARKERS = {"gpu": "o", "gpu-dp": "s", "hybrid-async-dp": "^", "hybrid-dist-dp": "D",
           "hybrid-async": "v", "hybrid-dist-pipelined": "P", "gpu-pipelined": "X"}
HYB = ["hybrid-async", "hybrid-async-dp", "hybrid-dist-dp", "hybrid-dist-pipelined", "mpi-gpu"]

def line_profile(ax, ratios, label, color=None, marker="o"):
    r = np.sort(np.asarray(ratios, dtype=float))
    n = len(r)
    y = np.arange(1, n + 1) / n
    x = np.concatenate([[r[0]], r])          # anchor bottom-left (vertical riser at r[0])
    yy = np.concatenate([[0.0], y])
    nfast = int((r <= 1.0 + 1e-9).sum())
    me = max(1, len(x) // 12)                # thin markers on dense lines (~12 per line)
    ax.plot(x, yy, "-", marker=marker, ms=6, markevery=me, lw=2.0, color=color,
            markeredgecolor="white", markeredgewidth=0.5,
            label=f"{label}  (fastest on {nfast}/{n})")

def log_x(ax, xmax):
    ax.set_xscale("log")
    ax.set_xlim(1.0 / 1.008, xmax * 1.05)
    # coarse ticks for a wide range (avoids label crowding near 1), fine when narrow
    if xmax > 1.3:
        cand = [1.0, 1.05, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0]
    else:
        cand = [1.0, 1.01, 1.02, 1.03, 1.05, 1.07, 1.1, 1.15, 1.2, 1.25]
    ticks = [t for t in cand if t <= xmax * 1.05]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))

def load():
    df = pd.read_parquet(PARQ)
    for c in ("matrix", "solver", "part_src"):
        df[c] = df[c].astype(str)
    return df[df.ok & df.spmv_time.notna()].copy()

def solver_plot(ok):
    best = (ok[ok.solver.isin(SOLVERS)]
            .groupby(["matrix", "solver"])["spmv_time"].min().unstack("solver"))
    best = best.dropna(how="any")                    # matrices where every solver ran
    ratio = best.div(best.min(axis=1), axis=0)       # each row / row-min
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for s in SOLVERS:
        if s in ratio:
            line_profile(ax, ratio[s].values, s, COLORS.get(s), MARKERS.get(s, "o"))
    log_x(ax, float(ratio.max().max()))
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("performance ratio  (solver time / fastest time on that matrix)  —  log scale")
    ax.set_ylabel("fraction of matrices")
    ax.set_title(f"Solver performance profile  ({len(ratio)} matrices)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5, loc="lower right", framealpha=0.95)
    fig.tight_layout()
    out = DOCS / "perf-profile-solvers"
    fig.savefig(f"{out}.png", dpi=140); fig.savefig(f"{out}.svg")
    print(f"wrote {out}.png/.svg  ({len(ratio)} matrices)")

def patoh_naive_plot(ok):
    h = ok[ok.solver.isin(HYB) & ok.part_src.isin(["patoh", "naive"])]
    best = (h.groupby(["matrix", "solver", "part_src"])["spmv_time"].min()
            .unstack("part_src"))
    best = best.dropna(how="any")                    # (matrix,solver) where both ran
    ratio = best.div(best.min(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    line_profile(ax, ratio["patoh"].values, "PaToH", "#2c5e9e", "o")
    line_profile(ax, ratio["naive"].values, "naive (nnz-balance)", "#e39a2e", "s")
    log_x(ax, float(ratio.max().max()))
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("performance ratio  (method time / faster of {PaToH, naive})  —  log scale")
    ax.set_ylabel("fraction of (matrix, solver) cases")
    ax.set_title(f"PaToH vs naive partitioning  ({len(ratio)} cases)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc="lower right", framealpha=0.95)
    fig.tight_layout()
    out = DOCS / "perf-profile-patoh-naive"
    fig.savefig(f"{out}.png", dpi=140); fig.savefig(f"{out}.svg")
    print(f"wrote {out}.png/.svg  ({len(ratio)} cases)")

if __name__ == "__main__":
    ok = load()
    solver_plot(ok)
    patoh_naive_plot(ok)
