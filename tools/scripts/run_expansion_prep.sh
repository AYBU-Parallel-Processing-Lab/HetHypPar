#!/usr/bin/env bash
# Prep the NEW expansion matrices (vectors + PaToH + naive partitions). The 14
# already-benchmarked matrices are skipped. Idempotent; raised-stack patpart.
#
#   tmux new-session -d -s exprep 'bash tools/scripts/run_expansion_prep.sh'
#   tail -f data/expansion/prep.log
set -u
cd "$(dirname "$0")/../.." || exit 1
export LC_ALL=C
mkdir -p data/expansion
LOG=data/expansion/prep.log

EXISTING="RM07R cage14 Hamrle3 rajat30_mc64_5 circuit5M_dc circuit5M cage15 ML_Geer vas_stokes_1M nv2 Transport Freescale2 dgreen ss"
NEW=$(grep -vxF -f <(printf '%s\n' $EXISTING) data/expansion_matrices.txt | tr '\n' ' ')

{
  echo "================ EXPANSION PREP START $(date) ================"
  echo "new matrices ($(echo $NEW | wc -w)): $NEW"
  echo
  micromamba run -n octave python tools/python/prep_big_matrix.py $NEW
  echo
  echo "================ EXPANSION PREP DONE $(date) ================"
} 2>&1 | tee -a "$LOG"
