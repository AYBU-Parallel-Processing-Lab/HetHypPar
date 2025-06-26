# MTX to Hypergraph Partitioning Tool

This script converts a sparse matrix in Matrix Market (.mtx) format to a hypergraph representation (via a temporary hMetis file) and performs partitioning using the mtkahypar library.

## Requirements

- Python 3.6+
- mtkahypar
- scipy
- numpy

## Installation

```bash
# Install dependencies
pip install scipy numpy
# Install mtkahypar following the library's instructions
```

## Basic Usage

Partition a matrix into 2 equal parts:

```bash
python mtx_partitioning.py path/to/matrix.mtx
```

## Advanced Usage

### Specify number of partitions

```bash
python mtx_partitioning.py path/to/matrix.mtx --blocks 4
```

### Custom partition weights for heterogeneous systems

For heterogeneous systems, you can specify the target weight for each partition. This is useful when you have processors with different computational capabilities.

#### Option 1: Specify weights directly

For example, to partition a matrix for 3 processors with computational capabilities in the ratio 2:1:1:

```bash
python mtx_partitioning.py path/to/matrix.mtx --blocks 3 --block-weights 0.5,0.25,0.25
```

#### Option 2: Read weights from a file

Create a text file with one weight per line:

```
# weights.txt - Comments are supported (lines starting with #)
0.4
0.3
0.3
```

Then run:

```bash
python mtx_partitioning.py path/to/matrix.mtx --blocks 3 --block-weights-file weights.txt
```

If the weights in the file don't sum to 1.0, they will be automatically normalized.

The number of weights must match the number of blocks. If both `--block-weights` and `--block-weights-file` are provided, the file takes precedence.

### Adjust partitioning quality

```bash
python mtx_partitioning.py path/to/matrix.mtx --preset quality
```

Options: `default`, `quality`, `highest_quality`

### Change allowed imbalance

```bash
python mtx_partitioning.py path/to/matrix.mtx --imbalance 0.05
```

A value of 0.05 means a 5% imbalance is allowed.

### Save partition assignment

Save the partition assignment (vertex to block mapping) to a file:

```bash
python mtx_partitioning.py path/to/matrix.mtx --output partition_result.txt
```

The output file will contain one line per vertex, with the format: `vertex_id block_id`

### Full examples

Using command line weights:
```bash
python mtx_partitioning.py path/to/matrix.mtx --blocks 4 --block-weights 0.4,0.3,0.2,0.1 --preset highest_quality --imbalance 0.02 --seed 123 --output partition.txt
```

Using a weights file:
```bash
python mtx_partitioning.py path/to/matrix.mtx --blocks 4 --block-weights-file system_capabilities.txt --preset highest_quality --imbalance 0.02 --seed 123 --output partition.txt
```

## How It Works

The script performs the following steps:

1. Loads the Matrix Market file using SciPy
2. Creates a temporary hMetis format file for row-wise partitioning:
   - Each row becomes a vertex
   - Each column becomes a hyperedge
   - Row weights are based on the number of non-zeros in each row
3. Loads the hypergraph from the temporary file using mtkahypar
4. Partitions the hypergraph using mtkahypar
5. Reports partition statistics and writes results to a file if specified
6. Cleans up the temporary file

## Customization

### Vertex Weights

By default, the script uses the number of non-zeros in each row as the vertex weight. This can be customized in the `mtx_to_hypergraph` function.

### Hyperedge Weights

By default, all hyperedges (columns) have a weight of 1. You can modify this in the `mtx_to_hypergraph` function based on column properties if needed.

## Troubleshooting

If you encounter issues with large matrices, consider:

1. Ensuring you have enough memory
2. Simplifying the matrix if possible (e.g., removing zero or near-zero entries)
3. Reducing the quality preset to "default" instead of "quality" or "highest_quality"