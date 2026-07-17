#!/usr/bin/env bash
# Expansion sweeps, run SEQUENTIALLY (both use the GPU -> no concurrency):
#   1. full-solver sweep on the 35 NEW matrices  -> data/expansion_benchmark/
#      (all 10 solvers x 6 weights x {patoh,naive}; merge with the existing 14 later)
#   2. SpMV-only PaToH-vs-naive across ALL 49     -> data/spmv_patoh/ (clobbers; one clean set)
#
#   tmux new-session -d -s expsweep 'bash tools/scripts/run_expansion_sweeps.sh'
#   tail -f data/expansion/sweeps.log
set -u
cd "$(dirname "$0")/../.." || exit 1
export LC_ALL=C
mkdir -p data/expansion
LOG=data/expansion/sweeps.log

EXISTING="RM07R cage14 Hamrle3 rajat30_mc64_5 circuit5M_dc circuit5M cage15 ML_Geer vas_stokes_1M nv2 Transport Freescale2 dgreen ss"
# drop atmosmodm (octave mmread fails on its integer field)
ALL=$(grep -vxF atmosmodm data/expansion_matrices.txt | tr '\n' ' ')
NEW=$(grep -vxF -f <(printf '%s\n' $EXISTING) data/expansion_matrices.txt | grep -vxF atmosmodm | tr '\n' ' ')

{
  echo "================ EXPANSION SWEEPS START $(date) ================"
  echo "NEW ($(echo $NEW | wc -w)) for full-solver; ALL ($(echo $ALL | wc -w)) for SpMV"
  echo
  echo "---------------- 1. FULL-SOLVER sweep (NEW matrices) $(date) ----------------"
  OUT=data/expansion_benchmark WEIGHTS=w400,w600,w800,w1000,w1200,w2000 \
    micromamba run -n octave python tools/python/big_benchmark_sweep.py $NEW
  echo
  echo "---------------- 2. SpMV-only sweep (ALL 49) $(date) ----------------"
  REPS=2 NITER=300 \
    micromamba run -n octave python tools/python/spmv_patoh_sweep.py $ALL
  echo
  echo "================ EXPANSION SWEEPS DONE $(date) ================"
} 2>&1 | tee -a "$LOG"
