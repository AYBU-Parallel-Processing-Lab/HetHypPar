#!/bin/bash

set -e

# Function to print and run a command
run_cmd() {
    echo "Running: $@"
    "$@"
    return $?
}

# Function to display usage
usage() {
    cat << EOF
Usage: $0 -m <matrix_name> -g <is_gpu_file> -p <partition_file> [-i <iterations>]

Required arguments:
  -m <matrix_name>      Name of the matrix (e.g., cage13)
  -g <is_gpu_file>      Path to is_gpu.txt file (relative to current directory)
  -p <partition_file>   Partition file name (relative to data/matrices/<matrix>/in/part/)

Optional arguments:
  -i <iterations>       Number of iterations (default: 20)
  -h                    Display this help message

Example:
  $0 -m cage13 -g is_gpu.txt -p w100.part
  $0 -m cage13 -g is_gpu.txt -p w100.part -i 50
EOF
    exit 1
}

# Initialize variables
MATNAME=""
IS_GPU_FILE=""
PART_FILE=""
ITERATIONS=20

# Parse command line arguments
while getopts "m:g:p:i:h" opt; do
    case ${opt} in
        m)
            MATNAME=$OPTARG
            ;;
        g)
            IS_GPU_FILE=$OPTARG
            ;;
        p)
            PART_FILE=$OPTARG
            ;;
        i)
            ITERATIONS=$OPTARG
            ;;
        h)
            usage
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            usage
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            usage
            ;;
    esac
done

# Check required arguments
if [ -z "$MATNAME" ]; then
    echo "Error: Matrix name (-m) is required" >&2
    usage
fi

if [ -z "$IS_GPU_FILE" ]; then
    echo "Error: is_gpu file (-g) is required" >&2
    usage
fi

if [ -z "$PART_FILE" ]; then
    echo "Error: Partition file (-p) is required" >&2
    usage
fi

# Verify is_gpu file exists
if [ ! -f "$IS_GPU_FILE" ]; then
    echo "Error: is_gpu file '$IS_GPU_FILE' not found" >&2
    exit 1
fi

# Set up paths
MAT_BASEDIR="data/matrices/${MATNAME}"
PART_FILE_PATH="${MAT_BASEDIR}/in/part/${PART_FILE}"

# Verify partition file exists
if [ ! -f "$PART_FILE_PATH" ]; then
    echo "Error: Partition file '$PART_FILE_PATH' not found" >&2
    exit 1
fi

# Determine MPI processes from partition file
# Assuming the partition file format allows us to count unique partition IDs
MPI_PROCS=$(sort -u "$PART_FILE_PATH" | wc -l)
echo "Detected $MPI_PROCS MPI processes from partition file"

mkdir -p ${MAT_BASEDIR}/in ${MAT_BASEDIR}/out

echo
echo "Running MPI version:"
run_cmd mpirun --report-bindings -bind-to core -n ${MPI_PROCS} ./build/bicgstab-mpi \
  -m /matrices/${MATNAME}.mtx \
  -n ${ITERATIONS} \
  -p ${PART_FILE_PATH} \
  -o ${MAT_BASEDIR}/out/X_mpi.txt \
  -x ${MAT_BASEDIR}/in/X_init.txt \
  -y ${MAT_BASEDIR}/in/B.txt
