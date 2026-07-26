#!/bin/bash
# Multi-node CPU-only MPI scaling run (bicgstab-mpi) on orfoz.
# One MPI rank per node, full 112 cores/rank via OpenMP -- sweep -N for a
# weak/strong scaling curve, e.g.:
#   sbatch --nodes=2  tools/slurm/mpi_cpu_scaling.slurm
#   sbatch --nodes=4  tools/slurm/mpi_cpu_scaling.slurm
#   sbatch --nodes=8  tools/slurm/mpi_cpu_scaling.slurm
# See docs/truba-hpc-guide.md.

#SBATCH -p orfoz
#SBATCH -A CHANGE_ME_ACCOUNT      # EDIT: your TRUBA account
#SBATCH -J hhp-mpi-cpu
#SBATCH -N 2                       # EDIT (or override with --nodes= on the sbatch command line)
#SBATCH --ntasks-per-node=1
#SBATCH -c 112
#SBATCH --time=03:00:00
#SBATCH -o /arf/scratch/%u/hhp-mpi-cpu-%j.out
#SBATCH -e /arf/scratch/%u/hhp-mpi-cpu-%j.err

set -euo pipefail

cd /arf/scratch/"$USER"/HetHypPar   # EDIT: path to your checked-out repo on scratch

module purge
module load <openmpi-version>       # EDIT: from `avci openmpi` for your account

export OMP_NUM_THREADS=112          # 1 rank/node here, so the full core count is correct
                                     # (CLAUDE.md's "tune down to 1-4" warning is for the
                                     # hybrid solvers' small per-rank CPU slices, not this)

echo "SLURM_NODELIST $SLURM_NODELIST"
echo "NUMBER OF TASKS $SLURM_NTASKS"

# EDIT: matrix + partition. Partition file must have exactly $SLURM_NTASKS
# (== node count here) distinct partition IDs -- regenerate with
# tools/python/matrix_partition.py for this specific rank count if you don't
# already have one.
MATRIX=HV15R
MATRIX_DIR=/arf/scratch/$USER/matrices/$MATRIX
PART=$MATRIX_DIR/in/part/w100_${SLURM_NTASKS}_i1.part

mkdir -p /arf/scratch/$USER/out

mpirun ./build/bicgstab-mpi \
  -m "$MATRIX_DIR/$MATRIX.mtx" \
  -x "$MATRIX_DIR/in/X_init.txt" \
  -y "$MATRIX_DIR/in/B.txt" \
  -p "$PART" \
  -o /arf/scratch/$USER/out/${MATRIX}_mpi_${SLURM_NNODES}n.txt \
  -n 200
