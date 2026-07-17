#!/usr/bin/env python3
"""
Plots for the professor report, from the unified HHP_PROF profiling data.

(1) Section breakdown: per-matrix panels (ordered by density nnz/row), stacked
    bars of the per-iteration section times (spmv/dot/vecops/sync) for each solver.
    Shows where time goes and how the mix shifts with matrix structure.
(2) Per-iteration timeline: one solver+matrix, each section vs iteration, showing
    the iteration-0 startup outlier and the steady state (HHP_PROF=2 detail).

Inputs:
  data/prof/summary.tsv                (from tools/scripts/run_profiling_report.sh)
  data/prof/<solver>__<matrix>.tsv     per-iteration dumps

Run: micromamba run -n octave python tools/python/plot_profiling_report.py
Writes docs/profiling-breakdown.{png,svg}, docs/profiling-timeline.{png,svg}
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent.parent
SUMMARY = ROOT / "data/prof/summary.tsv"
FEAT = ROOT / "data/expansion_full/matrix_features.tsv"
ITERS = 200

# solver display order + short labels
SOLVERS = ["gpu", "gpu-dp", "gpu-pipelined", "hybrid-async-dp",
           "hybrid-dist-dp", "hybrid-dist-pipelined"]
SHORT = {"gpu": "gpu", "gpu-dp": "gpu-dp", "gpu-pipelined": "gpu-pipe",
         "hybrid-async-dp": "async-dp", "hybrid-dist-dp": "dist-dp",
         "hybrid-dist-pipelined": "dist-pipe"}
SECTIONS = ["spmv", "dot", "vecops", "sync"]
COLORS = {"spmv": "#2c6fbb", "dot": "#e08214", "vecops": "#6aa84f", "sync": "#b03060"}


def density_map():
    if not FEAT.exists():
        return {}
    f = pd.read_csv(FEAT, sep="\t")
    return dict(zip(f["matrix"].astype(str), f["avg_deg"].astype(float)))


def breakdown(df, dens):
    mats = [m for m in df["matrix"].unique()]
    mats.sort(key=lambda m: dens.get(m, 0))
    n = len(mats)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 4.6), squeeze=False)
    axes = axes[0]
    for ax, m in zip(axes, mats):
        sub = df[df.matrix == m].set_index("solver")
        solv = [s for s in SOLVERS if s in sub.index]
        x = np.arange(len(solv))
        bottom = np.zeros(len(solv))
        for sec in SECTIONS:
            vals = np.array([sub.loc[s, sec] for s in solv]) / ITERS * 1e6  # us/iter
            ax.bar(x, vals, bottom=bottom, color=COLORS[sec], width=0.72,
                   edgecolor="white", linewidth=0.4)
            bottom += vals
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[s] for s in solv], rotation=45, ha="right", fontsize=8)
        d = dens.get(m, float("nan"))
        ax.set_title(f"{m}\nnnz/row = {d:.1f}", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        # mark the fastest (lowest loop_total) solver
        tot = np.array([sub.loc[s, "loop_total"] for s in solv])
        best = int(np.argmin(tot))
        ax.text(best, bottom[best], "★", ha="center", va="bottom",
                fontsize=13, color="#d4a017")
    axes[0].set_ylabel("time per iteration  (µs)")
    handles = [Patch(facecolor=COLORS[s], label=s) for s in SECTIONS]
    handles.append(Patch(facecolor="none", edgecolor="none", label="★ fastest"))
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 1.005))
    fig.suptitle("Per-section time breakdown (HHP_PROF), by matrix density — "
                 "left = sparse, right = dense", y=1.06, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = ROOT / "docs/profiling-breakdown"
    fig.savefig(f"{out}.png", dpi=140, bbox_inches="tight")
    fig.savefig(f"{out}.svg", bbox_inches="tight")
    print(f"wrote {out}.png/.svg  ({n} matrices)")


def timeline(solver, matrix):
    tsv = ROOT / f"data/prof/{solver}__{matrix}.tsv"
    if not tsv.exists():
        print(f"skip timeline: {tsv} missing")
        return
    d = pd.read_csv(tsv, sep="\t")
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.4),
                                   gridspec_kw={"width_ratios": [1, 3]})
    for sec in SECTIONS:
        if d[sec].sum() == 0:
            continue
        y = d[sec] * 1e6  # us
        ax0.plot(d["iter"], y, "-", color=COLORS[sec], lw=1.3, label=sec)
        ax1.plot(d["iter"], y, "-", color=COLORS[sec], lw=1.1, label=sec)
    ax0.set_title("iterations 0–9 (startup)", fontsize=10)
    ax0.set_xlim(-0.3, 9)
    ax0.set_xlabel("iteration"); ax0.set_ylabel("time (µs)")
    ax0.grid(alpha=0.3); ax0.axvspan(-0.3, 0.5, color="#ffd8d8", alpha=0.6)
    ax0.text(0.1, ax0.get_ylim()[1] * 0.92, "iter 0\noutlier", fontsize=8, color="#b03030")

    steady = d[d["iter"] >= 1]
    ax1.set_title("steady state (iter ≥ 1)", fontsize=10)
    ax1.set_xlim(1, d["iter"].max())
    lo = min(steady[s].min() for s in SECTIONS if steady[s].sum() > 0) * 1e6
    hi = max(steady[s].max() for s in SECTIONS if steady[s].sum() > 0) * 1e6
    ax1.set_ylim(max(0, lo * 0.7), hi * 1.15)
    ax1.set_xlabel("iteration"); ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9, ncol=4, loc="upper right")
    fig.suptitle(f"Per-iteration section timeline — {solver} on {matrix} "
                 f"(HHP_PROF=2)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = ROOT / "docs/profiling-timeline"
    fig.savefig(f"{out}.png", dpi=140); fig.savefig(f"{out}.svg")
    print(f"wrote {out}.png/.svg  ({solver} / {matrix})")


def main():
    df = pd.read_csv(SUMMARY, sep="\t")
    for c in SECTIONS + ["loop_total", "preprocess"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    dens = density_map()
    breakdown(df, dens)
    timeline("hybrid-dist-dp", "marine1")   # has a visible sync bucket + iter-0 spike


if __name__ == "__main__":
    main()
