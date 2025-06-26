import multiprocessing
import mtkahypar
import scipy.io as sio
import numpy as np
from pathlib import Path
import argparse

def mtx_to_hypergraph(mtx_file, context):
    """
    Convert a Matrix Market file to hypergraph representation for mtkahypar.
    Row-wise partitioning: each row becomes a vertex, each column becomes a hyperedge.
    """
    # Load the MTX file using scipy
    mtx_path = Path(mtx_file)
    matrix = sio.mmread(mtx_path)
    
    # Convert to CSR format for efficient row access
    matrix_csr = matrix.tocsr()
    
    num_rows, num_cols = matrix.shape
    
    # Create hypergraph in memory (row-wise partitioning)
    # In this representation:
    # - Vertices are rows
    # - Hyperedges are columns
    # - Each vertex (row) is connected to hyperedges (columns) where it has non-zero entries
    
    # First, prepare the data structures needed by mtkahypar
    hyperedge_indices = []  # Start indices for hyperedges
    hyperedge_vertices = [] # Vertices in each hyperedge
    hyperedge_weights = []  # Weight of each hyperedge
    vertex_weights = []     # Weight of each vertex (row)
    
    # Process column by column (each column is a hyperedge)
    for col in range(num_cols):
        hyperedge_indices.append(len(hyperedge_vertices))
        
        # Get non-zero entries in this column (using CSC view would be more efficient)
        col_data = matrix.getcol(col).nonzero()[0]
        
        # Add vertices (rows) to this hyperedge
        for row in col_data:
            hyperedge_vertices.append(row)
        
        # Add weight for this hyperedge (can be customized based on column properties)
        hyperedge_weights.append(1)  # Default weight of 1
    
    # Add final index to hyperedge_indices
    hyperedge_indices.append(len(hyperedge_vertices))
    
    # Set vertex weights (can be customized based on row properties)
    # For example, weights could be proportional to number of non-zeros in each row
    for row in range(num_rows):
        # Using nnz per row as weight - customize as needed
        vertex_weights.append(matrix_csr.getrow(row).nnz)
    
    # Create the hypergraph using mtkahypar's constructor
    hypergraph = mtkahypar.Hypergraph(
        num_vertices=num_rows,
        num_hyperedges=num_cols,
        hyperedge_indices=hyperedge_indices,
        hyperedge_vertices=hyperedge_vertices,
        hyperedge_weights=hyperedge_weights,
        vertex_weights=vertex_weights
    )
    
    return hypergraph

def create_parser():
    """Create a parser with fixed help strings that escape % characters properly."""
    parser = argparse.ArgumentParser(description='Partition a matrix in MTX format using mtkahypar')
    
    # Required arguments
    parser.add_argument('mtx_file', type=str, help='Path to the MTX file')
    
    # Optional arguments - carefully check each help string for % characters
    parser.add_argument('--blocks', type=int, default=2, 
                        help='Number of partitions')
    parser.add_argument('--imbalance', type=float, default=0.03, 
                        help='Allowed imbalance (e.g., 0.03 for 3%%)')  # Note the double %% to escape
    parser.add_argument('--preset', type=str, default='default', 
                        choices=['default', 'quality', 'highest_quality'], 
                        help='Preset type for partitioning')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed')
    parser.add_argument('--output', type=str, default=None, 
                        help='Output file for partition results')
    parser.add_argument('--block-weights', type=str, default=None, 
                        help='Comma-separated list of target block weights as fractions (must sum to 1)')
    parser.add_argument('--block-weights-file', type=str, default=None,
                        help='Path to a file containing block weights, one per line')
    
    return parser

def main():
    # Parse command line arguments
    parser = create_parser()
    args = parser.parse_args()
    # Map preset string to enum
    preset_map = {
        'default': mtkahypar.PresetType.DEFAULT,
        'quality': mtkahypar.PresetType.QUALITY,
        'highest_quality': mtkahypar.PresetType.HIGHEST_QUALITY
    }
    preset = preset_map[args.preset.lower()]
    
    # Initialize
    mtk = mtkahypar.initialize(multiprocessing.cpu_count())
    
    # Setup partitioning context
    context = mtk.context_from_preset(preset)
    
    # Get the number of blocks
    num_blocks = args.blocks
    
    # Set custom block weights if provided
    block_weights = None
    
    # Check if weights are provided via command line
    if args.block_weights:
        block_weights = [float(w) for w in args.block_weights.split(',')]
    
    # Check if weights are provided via file (file takes precedence)
    if args.block_weights_file:
        weights_path = Path(args.block_weights_file)
        if not weights_path.exists():
            raise FileNotFoundError(f"Block weights file not found: {args.block_weights_file}")
        
        # Read weights from file, one per line
        with open(weights_path, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
            # Filter out empty lines and comments
            lines = [line for line in lines if line and not line.startswith('#')]
            block_weights = [float(line) for line in lines]
    
    # Apply block weights if provided
    if block_weights:
        if len(block_weights) != num_blocks:
            raise ValueError(f"Number of block weights ({len(block_weights)}) must match number of blocks ({num_blocks})")
        
        # Normalize weights to sum to 1.0
        weight_sum = sum(block_weights)
        if abs(weight_sum - 1.0) > 1e-6:
            print(f"Warning: Block weights sum to {weight_sum}, normalizing to 1.0")
            block_weights = [w / weight_sum for w in block_weights]
        
        # Apply custom block weights
        context.set_target_block_weights(block_weights)
        print(f"Using custom block weights: {block_weights}")
    
    # Set partitioning parameters
    context.set_partitioning_parameters(
        num_blocks,                  # number of blocks
        args.imbalance,              # imbalance parameter
        mtkahypar.Objective.KM1)     # objective function
    
    # Set seed and logging
    mtkahypar.set_seed(args.seed)
    context.logging = True
    
    # Convert MTX to hypergraph
    print(f"Loading and converting {args.mtx_file} to hypergraph...")
    hypergraph = mtx_to_hypergraph(args.mtx_file, context)
    
    # Partition hypergraph
    print(f"Partitioning into {num_blocks} blocks...")
    partitioned_hg = hypergraph.partition(context)
    
    # Output metrics
    print("\nPartition Stats:")
    print(f"Imbalance = {partitioned_hg.imbalance(context)}")
    print(f"km1       = {partitioned_hg.km1()}")
    print("\nBlock Weights:")
    for i in partitioned_hg.blocks():
        print(f"Weight of Block {i} = {partitioned_hg.block_weight(i)}")
    
    # Output partition assignment if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            for vertex in range(hypergraph.num_vertices()):
                f.write(f"{vertex} {partitioned_hg.block(vertex)}\n")
        print(f"\nPartition assignment written to {args.output}")
    
    # Print summary of partition assignment (how many vertices in each block)
    block_counts = {}
    for vertex in range(hypergraph.num_vertices()):
        block = partitioned_hg.block(vertex)
        block_counts[block] = block_counts.get(block, 0) + 1
    
    print("\nVertex Distribution:")
    for block, count in sorted(block_counts.items()):
        print(f"Block {block}: {count} vertices ({count/hypergraph.num_vertices()*100:.2f}%)")

if __name__ == "__main__":
    main()