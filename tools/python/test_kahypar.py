#!/usr/bin/env python3

import scipy.io
import scipy.sparse
import sys
import os
from pathlib import Path
import mtkahypar
import numpy as np
from scipy.io import mminfo
import multiprocessing
import traceback

def mtx_to_hmetis_rownet(mtx_filepath, hgr_filepath):
    """
    Converts a sparse matrix from Matrix Market (.mtx) format to
    hMetis (.hgr) format using the row-net hypergraph model.
    """
    try:
        matrix = scipy.io.mmread(mtx_filepath)
        
        if not isinstance(matrix, scipy.sparse.coo_matrix):
            matrix = matrix.tocoo()
        
        num_rows, num_cols = matrix.shape
        
        if num_rows == 0 or num_cols == 0:
            with open(hgr_filepath, 'w') as f_out:
                f_out.write("0 0\n")
            return
        
        hyperedges = [set() for _ in range(num_rows)]
        
        for r, c in zip(matrix.row, matrix.col):
            hyperedges[r].add(c + 1)
        
        with open(hgr_filepath, 'w') as f_out:
            f_out.write(f"{num_rows} {num_cols}\n")
            
            for i in range(num_rows):
                vertex_set = hyperedges[i]
                
                if not vertex_set:
                    f_out.write("\n")
                else:
                    vertex_list = sorted(list(vertex_set))
                    f_out.write(" ".join(map(str, vertex_list)) + "\n")
        
        return True
    except Exception as e:
        raise e

def process_matrix(matname, matrices_dir, output_dir, temp_dir, failures_log):
    """
    Process a single matrix and save results.
    """
    path_mtx = matrices_dir / f"{matname}.mtx"
    
    try:
        # Get matrix info
        numrows, numcols, nnz, format_type, field, symmetry = mminfo(path_mtx)
        
        # Check if matrix is square and non-symmetric
        if numrows != numcols:
            print(f"Skipping {matname}: not square ({numrows}x{numcols})")
            return False
        
        if symmetry == 'symmetric':
            print(f"Skipping {matname}: symmetric matrix")
            return False
        
        print(f"\nProcessing {matname} ({numrows}x{numcols}, {nnz} nnz)")
        
        # Parameters
        param_imbal = 0.1
        param_preset = mtkahypar.PresetType.HIGHEST_QUALITY
        param_seed = 42
        param_blocks = 16
        
        # Create temporary hMetis file path
        path_hypergraph = temp_dir / f"{matname}.hMetis"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate target weights
        target_weight = numrows + numrows * param_imbal
        target_weight = int(target_weight)
        
        param_weights = np.array([2.26]*8 + [1.03]*8)
        param_weights *= target_weight / np.sum(param_weights)
        param_weights = param_weights.astype(int)
        
        # Adjust for rounding errors
        for i in range(target_weight - np.sum(param_weights)):
            param_weights[-i-1] += 1
        
        # Convert matrix to hMetis format
        mtx_to_hmetis_rownet(path_mtx, path_hypergraph)
        
        try:
            # Initialize MT-KaHyPar
            mtk = mtkahypar.initialize(multiprocessing.cpu_count())
            
            # Setup context
            context = mtk.context_from_preset(param_preset)
            context.set_partitioning_parameters(
                param_blocks,
                param_imbal,
                mtkahypar.Objective.KM1
            )
            mtkahypar.set_seed(param_seed)
            context.logging = False  # Disable verbose logging
            context.set_individual_target_block_weights(param_weights.tolist())
            
            # Load and partition hypergraph
            hypergraph = mtk.hypergraph_from_file(str(path_hypergraph), context)
            context.compute_max_block_weights(numrows)
            partitioned_hg = hypergraph.partition(context)
            
            # Prepare results
            results = []
            results.append(f"Matrix: {matname}")
            results.append("Partition Stats:")
            results.append(f"Imbalance = {partitioned_hg.imbalance(context)}")
            results.append(f"km1       = {partitioned_hg.km1()}")
            results.append("Block Weights:")
            for i in partitioned_hg.blocks():
                results.append(f"Weight of Block {i} = {partitioned_hg.block_weight(i)}")
            
            # Save results
            result_file = output_dir / f"{matname}_results.txt"
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(result_file, 'w') as f:
                f.write('\n'.join(results))
            
            print(f"Results saved to {result_file}")
            
        finally:
            # Clean up hMetis file
            if path_hypergraph.exists():
                path_hypergraph.unlink()
                print(f"Deleted temporary file: {path_hypergraph}")
        
        return True
        
    except Exception as e:
        error_msg = f"Failed to process {matname}: {str(e)}\n{traceback.format_exc()}\n"
        print(f"ERROR: {error_msg}")
        
        # Log failure
        with open(failures_log, 'a') as f:
            f.write(f"{'='*60}\n")
            f.write(error_msg)
            f.write('\n')
        
        # Clean up any temporary files
        temp_file = temp_dir / f"{matname}.hMetis"
        if temp_file.exists():
            temp_file.unlink()
        
        return False

def main():
    # Directories
    matrices_dir = Path("/matrices")
    output_dir = Path("./data/temporary")
    temp_dir = Path("/tmp/hmetis_temp")
    failures_log = output_dir / "failures.txt"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear previous failures log
    if failures_log.exists():
        failures_log.unlink()
    
    # Get all .mtx files
    mtx_files = sorted(matrices_dir.glob("*.mtx"))
    
    if not mtx_files:
        print("No .mtx files found in /matrices")
        return
    
    print(f"Found {len(mtx_files)} .mtx files")
    
    # Process statistics
    processed = 0
    skipped = 0
    failed = 0
    
    for mtx_file in mtx_files:
        matname = mtx_file.stem
        result = process_matrix(matname, matrices_dir, output_dir, temp_dir, failures_log)
        
        if result is True:
            processed += 1
        elif result is False:
            # Check if it was skipped or failed
            if failures_log.exists():
                with open(failures_log, 'r') as f:
                    if matname in f.read():
                        failed += 1
                    else:
                        skipped += 1
            else:
                skipped += 1
    
    # Clean up temp directory if empty
    if temp_dir.exists() and not any(temp_dir.iterdir()):
        temp_dir.rmdir()
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total .mtx files found: {len(mtx_files)}")
    print(f"Successfully processed: {processed}")
    print(f"Skipped (not square or symmetric): {skipped}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print(f"\nCheck {failures_log} for error details")

if __name__ == "__main__":
    main()