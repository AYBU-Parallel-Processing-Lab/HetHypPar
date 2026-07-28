#!/bin/bash
# Run bicgstab-gpu-dp back-to-back over a fixed list of
# matrices, in one barbun-cuda job, with per-run logs + a summary table.
# Edit the EDIT-marked lines, then: sbatch tools/slurm/matrix_sweep_barbun_cuda.slurm

#SBATCH -p barbun-cuda
#SBATCH -A ysuhail
#SBATCH -J hhp-6mat-sweep
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 96
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH -o /arf/scratch/%u/hhp-6mat-sweep-%j.out
#SBATCH -e /arf/scratch/%u/hhp-6mat-sweep-%j.err

# NOTE: deliberately no -e -- one failing/missing matrix must not kill the
# rest of the sweep. Every command's exit status is checked explicitly below.
set -uo pipefail

REPO_DIR=/arf/scratch/$USER/HetHypPar
MATRICES_ROOT=/arf/scratch/$USER/matrices/data/matrices
MATRICES=(cage15 circuit5M HV15R stokes vas_stokes_2M vas_stokes_4M)

ITERS=100

cd "$REPO_DIR" || { echo "ERROR: cannot cd to $REPO_DIR" >&2; exit 1; }

LOGDIR="$REPO_DIR/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
mkdir -p "$LOGDIR"
TIMELINE="$LOGDIR/timeline.log"
SUMMARY="$LOGDIR/summary.tsv"
printf 'matrix\tsolver\tstatus\tseconds\tlogfile\n' > "$SUMMARY"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$TIMELINE"; }

log "job $SLURM_JOB_ID starting on $(hostname), logs under $LOGDIR"

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
# for exe in spmv-cpu-mt bicgstab-gpu-dp; do
#     if [ ! -x "./build/$exe" ]; then
#         log "ERROR: ./build/$exe missing/not executable after build -- aborting"
#         exit 1
#     fi
# done

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
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
    if [ ! -f "$A" ] || [ ! -f "$B" ] || [ ! -f "$X" ]; then
        log "SKIP   $m -- missing input file(s) under $MDIR (need A.mtx, b.txt, x.txt)"
        printf '%s\t(all)\tSKIPPED(missing input)\t0\t-\n' "$m" >> "$SUMMARY"
        continue
    fi

    # bicgstab-gpu-dp: full solve. x.txt = initial guess (-x), b.txt = RHS (-y)
    run_case "$m" "bicgstab-gpu-dp" \
        ./build/bicgstab-gpu-dp -m "$A" -x "$X" -y "$B" -o "$LOGDIR/${m}__bicgstab-gpu-dp_X.txt" -n "$ITERS"
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
