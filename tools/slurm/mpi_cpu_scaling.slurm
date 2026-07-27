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
# No CUDA here (orfoz has no GPU), but the binary is still dynamically linked
# against MKL + iomp5 (find_package(MKL) is unconditional in src/CMakeLists.txt,
# even for the CPU-only targets) -- oneapi must be loaded at RUNTIME too, not
# just when you built it, or the loader can't find libmkl_intel_lp64.so etc.
# Verified against `avci openmpi`/`avci` output on this account (2026-07-27).
module load comp/gcc/12.3.0
module load lib/openmpi/5.0.4
module load comp/oneapi/2024

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
