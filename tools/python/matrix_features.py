#!/usr/bin/env python3
"""
Build the ML-ready feature table: per matrix, structural features joined with the
benchmark outcomes (best solver, async-dp vs gpu-dp, SpMV PaToH-vs-naive). This is
the dataset for predicting best-method-from-structure.

Reads each matrix once for degree/bandwidth stats; joins with
  data/expansion_full/results.parquet  (solver outcomes)
  data/spmv_patoh/results.tsv           (SpMV PaToH-vs-naive)
  data/matrix_survey.tsv                (domain 'kind')

Run: micromamba run -n octave python tools/python/matrix_features.py
Writes data/expansion_full/matrix_features.tsv
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
PARQ = ROOT / "data/expansion_full/results.parquet"
SPMV = ROOT / "data/spmv_patoh/results.tsv"
SURV = ROOT / "data/matrix_survey.tsv"
OUT  = ROOT / "data/expansion_full/matrix_features.tsv"
COMP = ["gpu-dp", "hybrid-async-dp", "hybrid-dist-dp", "hybrid-async"]

def features(name):
    p = Path(f"/matrices/{name}.mtx")
    with open(p) as f:
        line = f.readline()
        while line.startswith("%"):
            line = f.readline()
        n, c, nnz = (int(x) for x in line.split()[:3])
        rdeg = np.zeros(n + 1, dtype=np.int64)
        cdeg = np.zeros(c + 1, dtype=np.int64)
        band_max = 0; band_sum = 0; diag = 0
        for line in f:
            if not line or line[0] == "%":
                continue
            i = line.find(" "); j = line.find(" ", i + 1)
            r = int(line[:i]); cc = int(line[i + 1:j])
            rdeg[r] += 1; cdeg[cc] += 1
            b = r - cc; b = -b if b < 0 else b
            if b > band_max: band_max = b
            band_sum += b
            if r == cc: diag += 1
    rd = rdeg[1:n + 1].astype(float); m = rd.mean()
    return dict(rows=n, nnz=nnz, avg_deg=nnz / n,
                row_deg_cv=float(rd.std() / m) if m else 0.0,       # hub/irregularity
                row_deg_max=int(rd.max()),
                row_deg_maxratio=float(rd.max() / m) if m else 0.0, # max / mean degree
                col_deg_max=int(cdeg[1:c + 1].max()),
                band_frac=band_max / n,                              # normalized bandwidth (locality)
                band_mean_frac=(band_sum / nnz) / n,
                diag_frac=diag / n)

def main():
    spmv = pd.read_csv(SPMV, sep="\t").dropna(subset=["hybrid_per_iter"])
    mats = sorted(spmv["matrix"].unique())
    surv = pd.read_csv(SURV, sep="\t").set_index("matrix")

    # solver outcomes
    df = pd.read_parquet(PARQ)
    for c in ("matrix", "solver"): df[c] = df[c].astype(str)
    ok = df[df.ok & df.spmv_time.notna()]
    best = ok.groupby(["matrix", "solver"]).speedup_vs_gpu.max().reset_index()
    piv = best.pivot_table(index="matrix", columns="solver", values="speedup_vs_gpu")

    # SpMV patoh/naive mean ratio per matrix (paired by weight)
    p = spmv[spmv.part_src == "patoh"].set_index(["matrix", "weight"])["hybrid_per_iter"]
    n = spmv[spmv.part_src == "naive"].set_index(["matrix", "weight"])["hybrid_per_iter"]
    sp = (p / n).groupby(level=0).mean()

    rows = []
    for i, mm in enumerate(mats, 1):
        print(f"  [{i}/{len(mats)}] {mm}", flush=True)
        feat = features(mm)
        comp = piv.loc[mm] if mm in piv.index else {}
        gd = comp.get("gpu-dp", np.nan)
        best_solver = (piv.loc[mm, COMP].astype(float).idxmax()
                       if mm in piv.index and piv.loc[mm, COMP].notna().any() else "")
        rows.append({"matrix": mm, "kind": surv.loc[mm, "kind"] if mm in surv.index else "",
                     **feat,
                     "best_solver": best_solver,
                     "asyncdp_vs_gpudp": (comp.get("hybrid-async-dp", np.nan) / gd) if gd else np.nan,
                     "distdp_vs_gpudp": (comp.get("hybrid-dist-dp", np.nan) / gd) if gd else np.nan,
                     "spmv_patoh_naive": float(sp.get(mm, np.nan))})
    out = pd.DataFrame(rows)
    out.to_csv(OUT, sep="\t", index=False)
    print(f"\nwrote {OUT}  ({len(out)} matrices)")

    # quick signal check: does irregularity predict PaToH hurting the SpMV?
    v = out.dropna(subset=["spmv_patoh_naive"])
    print("\ncorrelations with spmv_patoh_naive (>0 => more of feature -> PaToH worse):")
    for col in ["row_deg_cv", "row_deg_maxratio", "col_deg_max", "band_frac", "avg_deg", "nnz"]:
        print(f"  {col:<18} r = {v[col].corr(v['spmv_patoh_naive']):+.2f}")

if __name__ == "__main__":
    main()
