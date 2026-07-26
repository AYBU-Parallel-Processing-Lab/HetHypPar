#!/bin/bash
# Download and extract a matrix directly from the SuiteSparse Matrix
# Collection (sparse.tamu.edu) -- for matrices bigger than what's in the
# local /matrices/ mirror. See docs/truba-hpc-guide.md section 6 for how
# candidate (Group, Name) pairs were chosen.
#
# Usage: fetch_suitesparse_matrix.sh <Group> <Name> [dest_dir]
# Example: fetch_suitesparse_matrix.sh Fluorem HV15R /arf/scratch/$USER/matrices

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <Group> <Name> [dest_dir]" >&2
    echo "Example: $0 Fluorem HV15R /arf/scratch/\$USER/matrices" >&2
    exit 1
fi

GROUP=$1
NAME=$2
DEST=${3:-.}

mkdir -p "$DEST"
cd "$DEST"

echo "Fetching https://sparse.tamu.edu/MM/${GROUP}/${NAME}.tar.gz ..."
curl -L "https://sparse.tamu.edu/MM/${GROUP}/${NAME}.tar.gz" -o "${NAME}.tar.gz"
tar xzf "${NAME}.tar.gz"
rm "${NAME}.tar.gz"

echo "Done: $DEST/${NAME}/${NAME}.mtx"
