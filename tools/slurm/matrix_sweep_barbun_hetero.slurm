#!/bin/bash
# Run bicgstab-hybrid-async-dp (single-process CPU+GPU hybrid, SpMV split
# across CPU/GPU, ALL dot products stay GPU-resident via cuBLAS device-pointer
# mode) back-to-back over a fixed list of matrices, in one barbun-cuda job,
# with per-run logs + a summary table.
# Edit the EDIT-marked lines, then: sbatch tools/slurm/matrix_sweep_barbun_hetero.slurm

#SBATCH -p barbun-cuda
#SBATCH -A ysuhail
#SBATCH -J hhp-hetero-sweep
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 20
#SBATCH --sockets-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH -o /arf/scratch/%u/hhp-hetero-sweep-%j.out
#SBATCH -e /arf/scratch/%u/hhp-hetero-sweep-%j.err

# NOTE: deliberately no -e -- one failing/missing matrix must not kill the
# rest of the sweep. Every command's exit status is checked explicitly below.
set -uo pipefail

REPO_DIR=/arf/scratch/$USER/HetHypPar
MATRICES_ROOT=/arf/scratch/$USER/matrices/data/matrices
MATRICES=(torso1_r0-2_r1-6)

ITERS=100

# is_gpu file: 2 lines, exactly one "1". Checked into the repo under
# data/is_gpu/ (not matrix-specific, so it stays alongside the repo, not on
# the scratch matrices mount). g2_2.txt = rank0 GPU, rank1 CPU -- must agree
# with which rank prep_big_matrix.py/patpart treated as the GPU weight
# (first line of the w*_2.txt weight file) when it built the partition above.
ISGPU="$REPO_DIR/data/is_gpu/g2_2.txt"

cd "$REPO_DIR" || { echo "ERROR: cannot cd to $REPO_DIR" >&2; exit 1; }

LOGDIR="$REPO_DIR/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
mkdir -p "$LOGDIR"
TIMELINE="$LOGDIR/timeline.log"
SUMMARY="$LOGDIR/summary.tsv"
printf 'matrix\tsolver\tstatus\tseconds\tlogfile\n' > "$SUMMARY"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$TIMELINE"; }

log "job $SLURM_JOB_ID starting on $(hostname), logs under $LOGDIR"

if [ ! -f "$ISGPU" ]; then
    log "ERROR: is_gpu file '$ISGPU' not found -- aborting"
    exit 1
fi

module purge
# Verified module set (2026-07-27) -- see docs/truba-hpc-guide.md sec.4. MKL
# (find_package(MKL) in src/CMakeLists.txt) and MPI (link_libraries(MPI::MPI_C)
# applies to every target, including these single-process ones) are both
# unconditional link dependencies -- both modules must stay loaded at RUNTIME,
# not just when you built.
module load comp/gcc/12.3.0
module load lib/cuda/12.4
module load lib/openmpi/5.0.4-cuda-12.4
module load comp/oneapi/2024

# barbun-cuda = P100 = sm_60. Rebuilding here on purpose: if your existing
# build/ was configured for a different queue (e.g. sm_70 for akya-cuda), the
# device-pointer kernels JIT-recompile inside the timed loop on first launch
# and badly skew the first matrix's numbers (see CLAUDE.md's CUDA-arch gotcha).
# log "configuring + building for sm_60 (P100)"
# rm -rf build/
# cmake -S src -B build -DCMAKE_C_COMPILER="$(command -v gcc)" -DCMAKE_CUDA_ARCHITECTURES=60 \
#     >> "$LOGDIR/build.log" 2>&1
# cmake --build build -j >> "$LOGDIR/build.log" 2>&1
# BUILD_RC=$?
# if [ $BUILD_RC -ne 0 ]; then
#     log "BUILD FAILED (rc=$BUILD_RC) -- see $LOGDIR/build.log, aborting sweep"
#     exit 1
# fi
# if [ ! -x "./build/bicgstab-hybrid-async-dp" ]; then
#     log "ERROR: ./build/bicgstab-hybrid-async-dp missing/not executable after build -- aborting"
#     exit 1
# fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OMP_PLACES=cores
export OMP_PROC_BIND=close

run_case() {
    local matrix=$1 label=$2; shift 2
    local outlog="$LOGDIR/${matrix}__${label}.out"
    local errlog="$LOGDIR/${matrix}__${label}.err"
    local start=$SECONDS
    log "START  $matrix / $label"
    "$@" > "$outlog" 2> "$errlog"
    local rc=$?
    local elapsed=$((SECONDS - start))
    if [ $rc -eq 0 ]; then
        log "OK     $matrix / $label (${elapsed}s)"
        printf '%s\t%s\tOK\t%d\t%s\n' "$matrix" "$label" "$elapsed" "$outlog" >> "$SUMMARY"
    else
        log "FAILED $matrix / $label (rc=$rc, ${elapsed}s) -- see $errlog"
        printf '%s\t%s\tFAILED(rc=%d)\t%d\t%s\n' "$matrix" "$label" "$rc" "$elapsed" "$errlog" >> "$SUMMARY"
    fi
}

for m in "${MATRICES[@]}"; do
    MDIR="$MATRICES_ROOT/$m"
    A="$MDIR/A.mtx"; B="$MDIR/B.txt"; X="$MDIR/X_init.txt"
    PART="$MDIR/in/part/workload.part"
    if [ ! -f "$A" ] || [ ! -f "$B" ] || [ ! -f "$X" ]; then
        log "SKIP   $m -- missing input file(s) under $MDIR (need A.mtx, B.txt, X_init.txt)"
        printf '%s\t(all)\tSKIPPED(missing input)\t0\t-\n' "$m" >> "$SUMMARY"
        continue
    fi
    if [ ! -f "$PART" ]; then
        log "SKIP   $m -- missing partition file '$PART' (run tools/python/prep_big_matrix.py for weight $WEIGHT first)"
        printf '%s\t(all)\tSKIPPED(missing partition)\t0\t-\n' "$m" >> "$SUMMARY"
        continue
    fi

    # bicgstab-hybrid-async-dp: full solve. x.txt = initial guess (-x), b.txt
    # = RHS (-y), -p 2-rank partition (GPU rows / CPU rows), -g is_gpu (which
    # of the 2 ranks is the GPU).
    run_case "$m" "bicgstab-hybrid-async-dp" \
        ./build/bicgstab-hybrid-async-dp -m "$A" -x "$X" -y "$B" -p "$PART" -g "$ISGPU" \
        -o "$LOGDIR/${m}__bicgstab-hybrid-async-dp_X.txt" -n "$ITERS"
done

log "sweep complete"
echo ""
echo "=== SUMMARY ($SUMMARY) ==="
column -t -s $'\t' "$SUMMARY"

# Non-zero exit if anything failed or was skipped, so job-completion emails /
# `sacct` state clearly flag it instead of showing a clean COMPLETED.
if grep -qE "FAILED|SKIPPED" "$SUMMARY"; then
    exit 1
fi
exit 0
