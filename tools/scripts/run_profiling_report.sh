#!/usr/bin/env bash
# Collect unified HHP_PROF profiling data for the professor report.
#
# Runs the key BiCGStab solvers at HHP_PROF=2 (stderr summary + per-iteration TSV)
# on a set of matrices spanning the density range, so the section breakdown can be
# plotted per solver and per matrix, plus a per-iteration timeline.
#
# Outputs:
#   data/prof/<solver>__<matrix>.tsv     per-iteration dumps (written by the solver)
#   data/prof/summary.tsv                parsed per-section accumulated totals
#
# Reproducible: pins LC_ALL=C (dot decimals), OMP_NUM_THREADS=4 (hybrid sweet spot),
# a fixed weight per matrix, g2_2 is_gpu, gpu-cpu PaToH partitions.
#
# Usage: tools/scripts/run_profiling_report.sh
set -u
cd "$(dirname "$0")/../.."

ITERS=200
WEIGHT=w1000_2          # mid GPU:CPU ratio; interior of every matrix's sweep
ISGPU=data/is_gpu/g2_2.txt
OUT=data/prof/summary.tsv
export LC_ALL=C OMP_NUM_THREADS=4

# matrix list (low -> high nnz/row density)
MATRICES=(poli4 trans4 marine1 Goodwin_127 TSOPF_RS_b2383_c1)

# solvers: single-process GPU baselines + the two winning hybrids.
# gpu/gpu-dp/gpu-pipelined take no -p/-g; the hybrids do.
HYBRID="hybrid-async-dp hybrid-dist-dp hybrid-dist-pipelined"
PUREGPU="gpu gpu-dp gpu-pipelined"

mkdir -p data/prof
echo -e "matrix\tsolver\tspmv\tdot\tvecops\tcomm\tallreduce\tsync\tloop_total\tpreprocess" > "$OUT"

# parse the [HHP_PROF] stderr block for one run into a TSV row.
parse_prof() {  # args: matrix solver logfile
  awk -v m="$1" -v s="$2" '
    /^\[HHP_PROF\] solver=/ { for(i=1;i<=NF;i++){ if($i ~ /^preprocess=/){ sub(/preprocess=/,"",$i); sub(/s$/,"",$i); pre=$i } } }
    /^  (spmv|dot|vecops|comm|allreduce|sync)  *[0-9]/ { v[$1]=$2 }
    /^  loop_total  *[0-9]/ { tot=$2 }
    END {
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", m, s,
        (v["spmv"]?v["spmv"]:0),(v["dot"]?v["dot"]:0),(v["vecops"]?v["vecops"]:0),
        (v["comm"]?v["comm"]:0),(v["allreduce"]?v["allreduce"]:0),(v["sync"]?v["sync"]:0),
        (tot?tot:0),(pre?pre:0)
    }' "$3"
}

for m in "${MATRICES[@]}"; do
  X=data/matrices/$m/in/X_init.txt
  B=data/matrices/$m/in/B.txt
  P=data/matrices/$m/in/part/gpu-cpu/${WEIGHT}_i1.part
  MTX=/matrices/$m.mtx
  [ -f "$P" ] || { echo "SKIP $m (no $WEIGHT partition)"; continue; }

  for s in $PUREGPU; do
    echo ">> $m  $s"
    log=$(mktemp)
    HHP_PROF=2 ./build/bicgstab-$s -m "$MTX" -x "$X" -y "$B" -o /tmp/Y.txt -n $ITERS 2>"$log" >/dev/null
    parse_prof "$m" "$s" "$log" >> "$OUT"; rm -f "$log"
  done

  for s in $HYBRID; do
    echo ">> $m  $s"
    log=$(mktemp)
    HHP_PROF=2 ./build/bicgstab-$s -m "$MTX" -x "$X" -y "$B" -p "$P" -g "$ISGPU" -o /tmp/Y.txt -n $ITERS 2>"$log" >/dev/null
    parse_prof "$m" "$s" "$log" >> "$OUT"; rm -f "$log"
  done
done

echo "wrote $OUT"
column -t -s $'\t' "$OUT"
