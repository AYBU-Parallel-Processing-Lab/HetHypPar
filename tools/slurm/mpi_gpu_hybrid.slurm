#!/bin/bash
# Multi-node MPI+GPU hybrid run (bicgstab-mpi-gpu) on akya-cuda.
# One MPI rank per GPU (4 GPUs/node) -- sweep -N for a multi-node scaling
# curve across up to 24 nodes (96 GPUs), e.g.:
#   sbatch --nodes=1  tools/slurm/mpi_gpu_hybrid.slurm   # 4 GPUs, single node
#   sbatch --nodes=2  tools/slurm/mpi_gpu_hybrid.slurm   # 8 GPUs, real inter-node Allreduce
#   sbatch --nodes=4  tools/slurm/mpi_gpu_hybrid.slurm
# See docs/truba-hpc-guide.md.

#SBATCH -p akya-cuda
#SBATCH -A CHANGE_ME_ACCOUNT       # EDIT: your TRUBA account
#SBATCH -J hhp-mpi-gpu
#SBATCH -N 2                        # EDIT (or override with --nodes= on the sbatch command line)
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH -c 10                       # TRUBA's minimum-core rule for akya-cuda: 10 cores/GPU
#SBATCH --time=03:00:00
#SBATCH -o /arf/scratch/%u/hhp-mpi-gpu-%j.out
#SBATCH -e /arf/scratch/%u/hhp-mpi-gpu-%j.err

set -euo pipefail

cd /arf/scratch/"$USER"/HetHypPar   # EDIT: path to your checked-out repo on scratch

module purge
module load <openmpi-version>       # EDIT: from `avci openmpi` for your account
module load <cuda-version>          # EDIT: from `avci cuda` for your account

export OMP_NUM_THREADS=2            # CLAUDE.md: keep this 1-4 for hybrid solvers -- the
                                     # default core count is pathological for the small
                                     # per-rank CPU row-slices in these solvers

echo "SLURM_NODELIST $SLURM_NODELIST"
echo "NUMBER OF TASKS $SLURM_NTASKS"

# EDIT: matrix + partition + is_gpu file. Partition file needs exactly
# $SLURM_NTASKS partition IDs (nodes * 4 here); is_gpu file needs one 0/1 per
# rank (all 1s below -- pure multi-GPU, no CPU ranks -- since the whole point
# of this run is to stress the multi-node GPU-to-GPU reduction path; add CPU
# ranks back in if you also want the CPU-split question at this scale).
MATRIX=HV15R
MATRIX_DIR=/arf/scratch/$USER/matrices/$MATRIX
PART=$MATRIX_DIR/in/part/w100_${SLURM_NTASKS}_i1.part
ISGPU=/arf/scratch/$USER/is_gpu/allgpu_${SLURM_NTASKS}.txt   # EDIT: generate as $SLURM_NTASKS lines of "1"

mkdir -p /arf/scratch/$USER/out

mpirun ./build/bicgstab-mpi-gpu \
  -m "$MATRIX_DIR/$MATRIX.mtx" \
  -x "$MATRIX_DIR/in/X_init.txt" \
  -y "$MATRIX_DIR/in/B.txt" \
  -p "$PART" \
  -g "$ISGPU" \
  -o /arf/scratch/$USER/out/${MATRIX}_mpigpu_${SLURM_NNODES}n.txt \
  -n 200
