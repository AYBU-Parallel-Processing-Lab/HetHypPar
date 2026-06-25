#!/usr/bin/env python3
"""
Parse a benchmark results TSV into a typed, zstd-compressed Parquet file.

Reads the long-format results written by big_benchmark_sweep.py /
variant_speedup_sweep.py, applies proper dtypes (categoricals + numerics),
derives a few convenience columns, optionally enriches each row with its
matrix's dimensions (rows/cols/nnz from the MatrixMarket header), and writes
Parquet with ZSTD compression.

Usage (octave env has pandas+pyarrow):
    micromamba run -n octave python tools/python/results_to_parquet.py \
        [INPUT.tsv] [-o OUTPUT.parquet] [--level N] [--no-meta]

Defaults: INPUT = data/big_benchmark/results.tsv
          OUTPUT = <input stem>.parquet   (zstd level 19)

Schema written:
  matrix, solver, part_src, weight, status   -> category
  weight_ratio (W from 'wW', NaN for baselines), threads, rows, cols, nnz -> ints
  spmv_time, relative_residual, speedup_vs_gpu                            -> float64
  ok (status=='ok'), is_baseline (part_src=='-')                         -> bool
"""
import argparse, re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent

def mtx_dims(name):
    """(rows, cols, nnz) from /matrices/<name>.mtx header, or (None,)*3."""
    p = Path(f"/matrices/{name}.mtx")
    if not p.exists():
        return (None, None, None)
    try:
        with open(p) as f:
            line = f.readline()
            while line.startswith("%"):
                line = f.readline()
            r, c, nz = (int(x) for x in line.split()[:3])
            return (r, c, nz)
    except Exception:
        return (None, None, None)

def main():
    ap = argparse.ArgumentParser(description="benchmark results TSV -> zstd Parquet")
    ap.add_argument("input", nargs="?", default=str(ROOT / "data/big_benchmark/results.tsv"),
                    help="input results TSV")
    ap.add_argument("-o", "--output", default=None, help="output .parquet (default: input stem)")
    ap.add_argument("--level", type=int, default=19, help="zstd compression level (default 19)")
    ap.add_argument("--no-meta", action="store_true", help="skip matrix dimension enrichment")
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output) if args.output else inp.with_suffix(".parquet")

    # empty cells (failed runs) -> NaN
    df = pd.read_csv(inp, sep="\t", dtype=str, keep_default_na=True, na_values=[""])

    # numeric columns
    for col in ("spmv_time", "relative_residual", "speedup_vs_gpu"):
        if col in df: df[col] = pd.to_numeric(df[col], errors="coerce")
    df["threads"] = pd.to_numeric(df["threads"], errors="coerce").astype("Int32")

    # derived
    df["weight_ratio"] = df["weight"].str.extract(r"w(\d+)").astype("Int32")  # NaN for '-'
    df["ok"] = (df["status"] == "ok")
    df["is_baseline"] = (df["part_src"] == "-")

    # speedup relative to the gpu-dp baseline (per-matrix min gpu-dp spmv_time),
    # alongside the existing speedup_vs_gpu (old bicgstab-gpu baseline)
    gpudp = (df[df["solver"] == "gpu-dp"].dropna(subset=["spmv_time"])
             .groupby("matrix")["spmv_time"].min())
    df["speedup_vs_gpudp"] = df.apply(
        lambda r: round(gpudp[r["matrix"]] / r["spmv_time"], 3)
        if (pd.notna(r["spmv_time"]) and r["matrix"] in gpudp.index) else None, axis=1)
    df["speedup_vs_gpudp"] = pd.to_numeric(df["speedup_vs_gpudp"], errors="coerce")

    # matrix dimensions
    if not args.no_meta:
        dims = {m: mtx_dims(m) for m in df["matrix"].unique()}
        df["rows"] = df["matrix"].map(lambda m: dims[m][0]).astype("Int64")
        df["cols"] = df["matrix"].map(lambda m: dims[m][1]).astype("Int64")
        df["nnz"]  = df["matrix"].map(lambda m: dims[m][2]).astype("Int64")

    # categoricals (compress well + nice for analysis)
    for col in ("matrix", "solver", "part_src", "weight", "status"):
        if col in df: df[col] = df[col].astype("category")

    # stable column order
    order = ["matrix", "rows", "cols", "nnz", "solver", "part_src", "weight", "weight_ratio",
             "threads", "spmv_time", "relative_residual", "speedup_vs_gpu", "speedup_vs_gpudp",
             "ok", "is_baseline", "status"]
    df = df[[c for c in order if c in df.columns]]

    df.to_parquet(out, engine="pyarrow", compression="zstd", compression_level=args.level, index=False)

    tsv_kb = inp.stat().st_size / 1024
    pq_kb  = out.stat().st_size / 1024
    print(f"wrote {out}")
    print(f"  rows={len(df)}  cols={len(df.columns)}  ok={int(df['ok'].sum())}  failed={int((~df['ok']).sum())}")
    print(f"  size: {tsv_kb:.1f} KB TSV -> {pq_kb:.1f} KB parquet(zstd-{args.level})  ({tsv_kb/pq_kb:.1f}x)")
    print("  dtypes:")
    for c, d in df.dtypes.items():
        print(f"    {c:<18} {d}")

if __name__ == "__main__":
    main()
