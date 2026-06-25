#!/usr/bin/env bash
# PaToH-fill: recover the 4 large matrices whose patpart segfaulted (stack-VLA
# bug), then run ONLY the missing work — their PaToH partition-using solvers —
# and merge into the canonical results. nlpkkt120 is excluded (its solver also
# crashes, independent of partitioning).
#
#   tmux new-session -d -s patohfill 'bash tools/scripts/run_patoh_fill.sh'
#   tail -f data/big_benchmark/patoh_fill.log
set -u
cd "$(dirname "$0")/../.." || exit 1
export LC_ALL=C
LOG=data/big_benchmark/patoh_fill.log
MATS="circuit5M_dc circuit5M cage15 Freescale2"

{
  echo "================ PATOH FILL START $(date) ================"
  echo "matrices: $MATS"
  echo "--- prep: PaToH partitions only (raised stack) ---"
  micromamba run -n octave python tools/python/prep_big_matrix.py --patoh-only $MATS

  echo "--- targeted sweep: PaToH only, partition-using solvers ---"
  OUT=data/big_benchmark_patoh_fill PARTSRCS=patoh \
  SOLVERS=hybrid-async,hybrid-async-dp,hybrid-dist-dp,hybrid-dist-pipelined,mpi,mpi-gpu \
    micromamba run -n octave python tools/python/big_benchmark_sweep.py $MATS

  echo "--- merge into canonical results.tsv ---"
  micromamba run -n octave python tools/python/merge_patoh_fill.py

  echo "--- regenerate zstd parquet ---"
  micromamba run -n octave python tools/python/results_to_parquet.py

  echo "================ PATOH FILL DONE $(date) ================"
} 2>&1 | tee -a "$LOG"
