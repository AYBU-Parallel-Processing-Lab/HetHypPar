#!/usr/bin/env python3
"""
Generate two box plots comparing mpi_gpu per-iteration timing components to
the single-GPU baseline, one for cutsize-0 partitions and one for the rest.

For each mpi_gpu run:
  - Parse profile_iterations JSON from benchmark_summary.tsv
  - Exclude iteration 0 (startup outlier)
  - Average each component across iterations 1..N-1
  - Divide by (gpu_spmv / niters) for the same matrix to get a ratio:
    "how many single-iteration GPU baselines does this component cost?"

Inputs:
  data/results/benchmark_summary.tsv  - output of parse_benchmark_results.py
  ~/Templates/results_medium/master_summary_clean.csv  - cutsize metadata

Outputs:
  data/results/profile_boxplot_cutsize0.png
  data/results/profile_boxplot_normal.png

Run:
  micromamba run -n octave python tools/python/plot_profile_boxplots.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import pandas as pd

SUMMARY_TSV = Path("data/results/benchmark_summary.tsv")
CUTSIZE_CSV = Path.home() / "Templates/results_medium/master_summary_clean.csv"
OUT_DIR = Path("data/results")

# Components to plot (label, list of profile fields summed to form the component)
COMPONENTS = [
    ("r0 local_spmv", 0, ["local_spmv"]),
    ("r0 shared_spmv", 0, ["shared_spmv"]),
    ("r0 comm", 0, ["send_fill", "comm_wait", "send_wait"]),
    ("r0 vecops", 0, ["vecops"]),
    ("r1 local_spmv", 1, ["local_spmv"]),
    ("r1 comm", 1, ["send_fill", "comm_wait", "send_wait"]),
    ("r1 vecops", 1, ["vecops"]),
    ("loop total (max)", None, None),  # special: max of r0/r1 total across iters
]


def per_iter_avg_excluding_iter0(iters_json, niters):
    """Return dict of per-rank per-component average over iters 1..N-1.

    Returns (r0_totals, r1_totals, loop_totals) where each is a dict of field
    -> average per iteration (excluding iter 0). loop_totals is a single float
    (max(r0, r1) per iter, averaged).
    """
    iters = json.loads(iters_json)
    r0 = [i for i in iters if i["rank"] == 0 and i["iter"] > 0]
    r1 = [i for i in iters if i["rank"] == 1 and i["iter"] > 0]
    if not r0 or not r1 or len(r0) != len(r1):
        return None

    fields = ["spmv", "vecops", "send_fill", "local_spmv",
              "comm_wait", "shared_spmv", "send_wait"]
    r0_avg = {f: sum(x[f] for x in r0) / len(r0) for f in fields}
    r1_avg = {f: sum(x[f] for x in r1) / len(r1) for f in fields}

    # Per-iter loop total = max of (spmv+vecops) across ranks, averaged
    loop_totals = []
    for a, b in zip(r0, r1):
        t0 = a["spmv"] + a["vecops"]
        t1 = b["spmv"] + b["vecops"]
        loop_totals.append(max(t0, t1))
    loop_avg = sum(loop_totals) / len(loop_totals)

    return r0_avg, r1_avg, loop_avg


def component_value(r0_avg, r1_avg, loop_avg, rank, fields):
    """Compute a component value from the averaged per-iter profile."""
    if rank is None:
        return loop_avg
    source = r0_avg if rank == 0 else r1_avg
    return sum(source[f] for f in fields)


def main():
    print(f"Loading {SUMMARY_TSV} ...")
    df = pd.read_csv(SUMMARY_TSV, sep="\t", low_memory=False)
    df = df[df["status"] == "success"].copy()

    # GPU baselines (one per matrix)
    gpu = df[df["solver_type"] == "gpu"][["matrix", "spmv"]].rename(
        columns={"spmv": "gpu_spmv"}
    )
    mg = df[df["solver_type"] == "mpi_gpu"].merge(gpu, on="matrix")
    mg["gpu_spmv"] = pd.to_numeric(mg["gpu_spmv"], errors="coerce")
    mg["n_iters"] = pd.to_numeric(mg["n_iters"], errors="coerce")
    mg = mg.dropna(subset=["gpu_spmv", "n_iters", "profile_iterations"])
    print(f"mpi_gpu runs with matching GPU baseline and profile: {len(mg)}")

    # Load cutsize metadata and build a set for fast lookup
    cs = pd.read_csv(CUTSIZE_CSV)
    cs0_keys = set(
        zip(
            cs[cs["Cut"] == 0]["Matrix"],
            cs[cs["Cut"] == 0]["Weight"],
            cs[cs["Cut"] == 0]["Imbalance"].astype(int),
            cs[cs["Cut"] == 0]["Seed"].astype(int),
        )
    )

    def is_cutsize0(row):
        try:
            key = (row["matrix"], row["weight"], int(row["imbalance"]), int(row["seed"]))
        except (ValueError, TypeError):
            return False
        return key in cs0_keys

    mg["cutsize0"] = mg.apply(is_cutsize0, axis=1)
    print(f"  cutsize-0 runs:     {mg['cutsize0'].sum()}")
    print(f"  non-cutsize-0 runs: {(~mg['cutsize0']).sum()}")

    # Build the plotting frame: one row per run, one column per component,
    # value = component_avg_per_iter / (gpu_spmv / niters)
    records = []
    for _, row in mg.iterrows():
        avgs = per_iter_avg_excluding_iter0(row["profile_iterations"], row["n_iters"])
        if avgs is None:
            continue
        r0_avg, r1_avg, loop_avg = avgs

        baseline_per_iter = row["gpu_spmv"] / row["n_iters"]
        if baseline_per_iter <= 0:
            continue

        rec = {"cutsize0": row["cutsize0"]}
        for label, rank, fields in COMPONENTS:
            val = component_value(r0_avg, r1_avg, loop_avg, rank, fields)
            rec[label] = val / baseline_per_iter
        records.append(rec)

    pdf = pd.DataFrame(records)
    print(f"Plotable records: {len(pdf)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for cutsize0, tag, title in (
        (True, "cutsize0", "cutsize-0 partitions"),
        (False, "normal", "non-cutsize-0 partitions"),
    ):
        subset = pdf[pdf["cutsize0"] == cutsize0]
        if subset.empty:
            print(f"no data for {tag}; skipping")
            continue

        labels = [c[0] for c in COMPONENTS]
        data = [subset[label].values for label in labels]

        fig, ax = plt.subplots(figsize=(11, 6))
        bp = ax.boxplot(
            data,
            tick_labels=labels,
            showfliers=True,
            whis=(5, 95),
            medianprops={"color": "red", "linewidth": 1.5},
        )
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1,
                   label="single-GPU per-iteration time")
        ax.set_ylabel("component time ÷ (gpu_spmv / niters)")
        ax.set_title(
            f"mpi_gpu per-iteration components vs single-GPU baseline\n"
            f"{title} — {len(subset)} runs, iter 0 excluded"
        )
        ax.set_yscale("log")
        ax.grid(True, axis="y", which="both", alpha=0.3)
        ax.legend(loc="upper right")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()

        out_path = OUT_DIR / f"profile_boxplot_{tag}.png"
        plt.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"wrote {out_path}  ({len(subset)} runs)")


if __name__ == "__main__":
    main()
