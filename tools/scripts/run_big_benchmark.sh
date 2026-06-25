#!/usr/bin/env bash
# Full big-matrix hybrid benchmark: prep (vectors + PaToH + naive partitions),
# then the cross-product sweep (10 solvers x 12 weights x {patoh,naive}).
#
# Resumable & idempotent (prep skips existing outputs; sweep re-runs are cheap).
# Designed to run under tmux so it survives an SSH/VPN drop:
#
#   tmux new-session -d -s bigbench 'bash tools/scripts/run_big_benchmark.sh'
#   tmux attach -t bigbench          # watch live   (Ctrl-b d to detach)
#   tail -f data/big_benchmark/run.log
#
# Progress + results persist to data/big_benchmark/ regardless of the connection.
set -u
cd "$(dirname "$0")/../.." || exit 1          # project root
export LC_ALL=C
mkdir -p data/big_benchmark
LOG=data/big_benchmark/run.log

MATRICES="RM07R cage14 Hamrle3 rajat30_mc64_5 circuit5M_dc circuit5M cage15 \
ML_Geer nlpkkt120 vas_stokes_1M nv2 Transport Freescale2 dgreen ss"

{
  echo "================ BIG BENCHMARK START $(date) ================"
  echo "matrices: $MATRICES"
  echo
  echo "---------------- PREP (vectors + partitions) $(date) ----------------"
  micromamba run -n octave python tools/python/prep_big_matrix.py $MATRICES
  echo
  echo "---------------- SWEEP $(date) ----------------"
  micromamba run -n octave python tools/python/big_benchmark_sweep.py $MATRICES
  echo
  echo "================ BIG BENCHMARK DONE $(date) ================"
} 2>&1 | tee -a "$LOG"
