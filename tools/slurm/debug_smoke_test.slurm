#!/bin/bash
# Single-node build + smoke test on the debug queue (max 4h).
# Run this FIRST, before any multi-node allocation, every time you touch the
# build or move to a new matrix. See docs/truba-hpc-guide.md.
#
# Edit the placeholders below, then: sbatch tools/slurm/debug_smoke_test.slurm

#SBATCH -p debug
#SBATCH -C akya-cuda            # EDIT: akya-cuda | barbun-cuda | orfoz | hamsi | barbun
#SBATCH -A CHANGE_ME_ACCOUNT     # EDIT: your TRUBA account
#SBATCH -J hhp-smoke
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10                    # EDIT: 10*ngpu (akya-cuda) / 20*ngpu (barbun-cuda) / min-core count for CPU-only families
#SBATCH --gres=gpu:1             # EDIT: delete this line entirely for orfoz/hamsi/barbun (no GPU there)
#SBATCH --time=00:30:00
#SBATCH -o /arf/scratch/%u/hhp-smoke-%j.out
#SBATCH -e /arf/scratch/%u/hhp-smoke-%j.err

set -euo pipefail

cd /arf/scratch/"$USER"/HetHypPar   # EDIT: path to your checked-out repo on scratch

module purge
# EDIT: fill in with versions `avci gcc` / `avci openmpi` / `avci cuda` / `avci mkl` report
# for YOUR account -- do not copy version numbers from the docs verbatim.
module load <gcc-version>
module load <openmpi-version>
module load <cuda-version>
module load <oneapi-or-mkl-version>

# EDIT: match CMAKE_CUDA_ARCHITECTURES to the queue above (see table in the guide):
#   P100(barbun-cuda)=60  V100(akya-cuda)=70  A100(palamut-cuda)=80  H100/H200(kolyoz-cuda)=90
cmake -S src -B build -G Ninja -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build build -j

echo "SLURM_NODELIST $SLURM_NODELIST"
echo "NUMBER OF TASKS $SLURM_NTASKS"

# EDIT: point at a small matrix you've already staged under /arf/scratch (see
# docs/truba-hpc-guide.md section 6 for how to pull one from SuiteSparse).
MATRIX_DIR=/arf/scratch/$USER/matrices/HV15R

./build/bicgstab-gpu \
  -m "$MATRIX_DIR/HV15R.mtx" \
  -x "$MATRIX_DIR/in/X_init.txt" \
  -y "$MATRIX_DIR/in/B.txt" \
  -o /arf/scratch/$USER/hhp-smoke-out.txt \
  -n 50
