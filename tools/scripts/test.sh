#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <matrix_name> [iterations] [mpi_processes]"
    echo "Example: $0 cage13"
    echo "Example: $0 cage13 50"
    echo "Example: $0 cage13 20 8"
    exit 1
fi

MATNAME=$1
ITERATIONS=${2:-20}    # Default to 20 if not provided
MPI_PROCS=${3:-16}     # Default to 16 if not provided

# Run sequential version
echo "Running sequential version:"
echo "./build/bicgstab-cpu -m /matrices/${MATNAME}.mtx -n ${ITERATIONS} -o data/${MATNAME}/out/X_seq.txt -x data/${MATNAME}/in/X_init.txt -y data/${MATNAME}/in/B.txt"
./build/bicgstab-cpu -m /matrices/${MATNAME}.mtx -n ${ITERATIONS} -o data/${MATNAME}/out/X_seq.txt -x data/${MATNAME}/in/X_init.txt -y data/${MATNAME}/in/B.txt

# Run sequential version
echo "Running sequential version:"
echo "./build/bicgstab-gpu -m /matrices/${MATNAME}.mtx -n ${ITERATIONS} -o data/${MATNAME}/out/X_gpu.txt -x data/${MATNAME}/in/X_init.txt -y data/${MATNAME}/in/B.txt"
./build/bicgstab-gpu -m /matrices/${MATNAME}.mtx -n ${ITERATIONS} -o data/${MATNAME}/out/X_gpu.txt -x data/${MATNAME}/in/X_init.txt -y data/${MATNAME}/in/B.txt

echo
echo "Running MPI version:"
echo "mpirun -n ${MPI_PROCS} ./build/bicgstab-mpi -m /matrices/${MATNAME}.mtx -n ${ITERATIONS} -p data/${MATNAME}/in/row_part_${MPI_PROCS} -o data/${MATNAME}/out/X_mpi.txt -x data/${MATNAME}/in/X_init.txt -y data/${MATNAME}/in/B.txt"
mpirun --report-bindings -bind-to core -n ${MPI_PROCS} ./build/bicgstab-mpi -m /matrices/${MATNAME}.mtx -n ${ITERATIONS} -p data/${MATNAME}/in/row_part_${MPI_PROCS} -o data/${MATNAME}/out/X_mpi.txt -x data/${MATNAME}/in/X_init.txt -y data/${MATNAME}/in/B.txt

echo
echo "Running MPI-GPU version:"
echo "mpirun --report-bindings -bind-to core -n ${MPI_PROCS} ./build/bicgstab-mpi-gpu -m /matrices/${MATNAME}.mtx -n ${ITERATIONS} -p data/${MATNAME}/in/row_part_${MPI_PROCS} -o data/${MATNAME}/out/X_mpi_gpu.txt -x data/${MATNAME}/in/X_init.txt -y data/${MATNAME}/in/B.txt -g data/${MATNAME}/is_gpu.txt"
mpirun --report-bindings -bind-to core -n ${MPI_PROCS} ./build/bicgstab-mpi-gpu -m /matrices/${MATNAME}.mtx -n ${ITERATIONS} -p data/${MATNAME}/in/row_part_${MPI_PROCS} -o data/${MATNAME}/out/X_mpi_gpu.txt -x data/${MATNAME}/in/X_init.txt -y data/${MATNAME}/in/B.txt -g data/${MATNAME}/is_gpu.txt