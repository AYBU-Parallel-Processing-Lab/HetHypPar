import argparse
import os
import sys

def process(matrix_name, mtx_file, cpu_blocks, gpu_blocks):
    total_blocks = cpu_blocks + gpu_blocks
    if total_blocks == 0:
        print("Error: Total blocks must be greater than 0.")
        sys.exit(1)

    out_name = f"{matrix_name}_{cpu_blocks}c_{gpu_blocks}g"
    out_dir = f"data/matrices/{out_name}"
    in_dir = os.path.join(out_dir, "in")
    part_dir = os.path.join(in_dir, "part")
    
    os.makedirs(part_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "out"), exist_ok=True)
    
    out_mtx = f"{out_dir}/{out_name}.mtx"
    
    print(f"Reading {mtx_file}...")
    with open(mtx_file, 'r') as f_in:
        header = ""
        while True:
            line = f_in.readline()
            if not line:
                break
            if line.startswith('%'):
                header += line
            else:
                break
        rows, cols, nnz = map(int, line.strip().split())

        # Parse each nonzero once: (r, c, val_str, val_float)
        # val_str preserves original precision for the matrix file;
        # val_float is used for the B = A * X_target computation.
        entries = []
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            r = int(parts[0])
            c = int(parts[1])
            val_str = parts[2] if len(parts) > 2 else "1"
            val_float = float(val_str) if len(parts) > 2 else 1.0
            entries.append((r, c, val_str, val_float))

    print(f"Writing {total_blocks}-block diagonal matrix to {out_mtx}...")
    with open(out_mtx, 'w') as f_out:
        f_out.write(header)
        f_out.write(f"{rows * total_blocks} {cols * total_blocks} {nnz * total_blocks}\n")
        for i in range(total_blocks):
            row_off = i * rows
            col_off = i * cols
            f_out.writelines(
                f"{r + row_off} {c + col_off} {val_str}\n"
                for (r, c, val_str, _) in entries
            )

    print("Generating X_init.txt, X_target.txt, and B.txt...")
    total_rows = rows * total_blocks

    # X_target: 1..N
    with open(f"{in_dir}/X_target.txt", "w") as f:
        f.write(f"{total_rows} 1\n")
        f.writelines(f"{i}\n" for i in range(1, total_rows + 1))

    # X_init: all ones
    with open(f"{in_dir}/X_init.txt", "w") as f:
        f.write(f"{total_rows} 1\n")
        f.writelines("1\n" for _ in range(total_rows))

    print("Computing B.txt = A * X_target...")
    b_vec = [0.0] * total_rows
    for i in range(total_blocks):
        row_off = i * rows
        col_off = i * cols
        for (r, c, _val_str, val_float) in entries:
            global_r = (r - 1) + row_off
            global_c = (c - 1) + col_off
            b_vec[global_r] += val_float * (global_c + 1)

    with open(f"{in_dir}/B.txt", "w") as f:
        f.write(f"{total_rows} 1\n")
        f.writelines(f"{val:.6f}\n" for val in b_vec)

    # Rank assignment: GPU on rank 0 (matches data/is_gpu/g2_2.txt convention).
    # Rank 0 gets gpu_blocks blocks, rank 1 gets cpu_blocks blocks.
    part_file = f"{part_dir}/workload.part"
    print(f"Generating partition file: {part_file}")
    with open(part_file, "w") as f:
        f.writelines("0\n" for _ in range(gpu_blocks * rows))
        f.writelines("1\n" for _ in range(cpu_blocks * rows))

    gpu_file = "data/is_gpu/g2_2_indep.txt"
    os.makedirs(os.path.dirname(gpu_file), exist_ok=True)
    with open(gpu_file, "w") as f:
        f.write("1\n0\n")
        
    print(f"\nDone! Original matrix size: {rows}x{cols}")
    print(f"New matrix size: {total_rows}x{total_rows}")
    print(f"Workload split: CPU={cpu_blocks} block(s), GPU={gpu_blocks} block(s)")
    print("\nYou can run the solver with:")
    print(f"mpirun --oversubscribe -bind-to none -n 2 ./build/bicgstab-mpi-gpu \\")
    print(f"  -m {out_mtx} \\")
    print(f"  -x {in_dir}/X_init.txt \\")
    print(f"  -y {in_dir}/B.txt \\")
    print(f"  -p {part_file} \\")
    print(f"  -g {gpu_file} \\")
    print(f"  -o {out_dir}/out/X_mpi_gpu.txt -n 100")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct an N-block diagonal matrix and partition it for CPU/GPU")
    parser.add_argument("matrix_name", help="Name of the matrix (e.g., circuit5M_dc)")
    parser.add_argument("mtx_file", help="Path to the .mtx file")
    parser.add_argument("-c", "--cpu-blocks", type=int, default=1, help="Number of blocks to assign to CPU (Rank 1)")
    parser.add_argument("-g", "--gpu-blocks", type=int, default=1, help="Number of blocks to assign to GPU (Rank 0)")
    
    args = parser.parse_args()
    process(args.matrix_name, args.mtx_file, args.cpu_blocks, args.gpu_blocks)
