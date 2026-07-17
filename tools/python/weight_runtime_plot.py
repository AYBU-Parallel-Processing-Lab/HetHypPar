#!/usr/bin/env python3
"""
Weight vs SpMV runtime, per matrix, to check the optimal split is bracketed by
the swept weight range (interior minimum = U-shaped, not at an endpoint).

x = CPU share (100/(w+100)); y = hybrid SpMV time normalized to each matrix's own
minimum. Each thin line is one matrix; the bold line is the median. An interior
dip means the weight range captured the optimum.

Run: micromamba run -n octave python tools/python/weight_runtime_plot.py [patoh|naive]
Writes docs/weight-runtime-profile.{png,svg}
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
SPMV = ROOT / "data/spmv_patoh/results.tsv"
SRC  = sys.argv[1] if len(sys.argv) > 1 else "naive"

df = pd.read_csv(SPMV, sep="\t").dropna(subset=["hybrid_per_iter"])
df = df[df.part_src == SRC].copy()
df["wnum"] = df.weight.str.extract(r"w(\d+)").astype(int)
df["cpu_pct"] = 100.0 * 100.0 / (df.wnum + 100.0)        # CPU share of the weight

fig, ax = plt.subplots(figsize=(8.6, 5.4))
norm_curves = []
endpoint = []
ws = sorted(df.wnum.unique()); lo, hi = ws[0], ws[-1]
for m, sub in df.groupby("matrix"):
    sub = sub.sort_values("cpu_pct")
    y = sub.hybrid_per_iter.values
    yn = y / y.min()
    ax.plot(sub.cpu_pct.values, yn, "-", color="#9bb8d8", lw=0.8, alpha=0.6, zorder=1)
    norm_curves.append(pd.Series(yn, index=sub.cpu_pct.values))
    amin = sub.loc[sub.hybrid_per_iter.idxmin(), "wnum"]
    if amin in (lo, hi):
        endpoint.append(m)

med = pd.concat(norm_curves, axis=1).median(axis=1).sort_index()
ax.plot(med.index, med.values, "-o", color="#b03030", lw=2.5, ms=5, zorder=3, label="median over matrices")
ax.axhline(1.0, color="#444", lw=0.8, ls="--", alpha=0.6)

ax.set_xscale("log")
ax.set_xlabel("CPU share of the split  (%)  —  log scale   (left = less CPU / more GPU, right = more CPU)")
ax.set_ylabel("SpMV time  /  matrix's own minimum")
ax.set_title(f"Weight vs SpMV runtime ({SRC}) — {df.matrix.nunique()} matrices "
             f"({df.matrix.nunique()-len(endpoint)} interior optima)")
ax.set_ylim(0.98, None)
ax.grid(alpha=0.3, which="both")
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
ax.xaxis.set_major_locator(FixedLocator([3, 5, 7, 10, 14, 20, 33, 50]))
ax.xaxis.set_minor_locator(NullLocator())
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.legend(loc="upper center", fontsize=10)
note = "all optima interior ✓" if not endpoint else f"endpoint optima: {', '.join(endpoint)}"
ax.text(0.98, 0.04, note, transform=ax.transAxes, ha="right", fontsize=9,
        color="#2f7d32" if not endpoint else "#b03030")
fig.tight_layout()
out = ROOT / "docs/weight-runtime-profile"
fig.savefig(f"{out}.png", dpi=140); fig.savefig(f"{out}.svg")
print(f"wrote {out}.png/.svg  ({df.matrix.nunique()} matrices, {len(endpoint)} endpoint optima: {endpoint})")
