import os
from pathlib import Path

def generate_hybrid_weights():
    # Define paths using pathlib
    cpu_dir = Path("cpu-p-e")
    gpu_dir = Path("gpu-cpu")
    output_dir = Path("hybrid-cpu-gpu")

    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get file lists and sort them for consistency
    # rglob handles searching for .txt files within the directories
    cpu_files = sorted(list(cpu_dir.glob("*.txt")))
    gpu_files = sorted(list(gpu_dir.glob("*.txt")))

    if not cpu_files or not gpu_files:
        print(f"Error: Ensure '{cpu_dir}' and '{gpu_dir}' exist and contain .txt files.")
        return

    print(f"Found {len(cpu_files)} CPU files and {len(gpu_files)} GPU files.")
    print(f"Generating {len(cpu_files) * len(gpu_files)} hybrid files in '{output_dir}'...")

    for c_path in cpu_files:
        # Extract base name (e.g., 'w219' from 'w219_16.txt')
        # .stem returns the filename without the extension
        c_base = c_path.stem.split('_')[0] 
        
        # Read CPU weights and calculate sum
        # read_text().splitlines() is a clean way to get a list of lines
        cpu_lines = [line.strip() for line in c_path.read_text().splitlines() if line.strip()]
        sum_cpu_weights = sum(float(w) for w in cpu_lines)

        for g_path in gpu_files:
            # Extract base name (e.g., 'w400' from 'w400_2.txt')
            g_base = g_path.stem.split('_')[0]
            
            # Read GPU reference value (first line)
            gpu_lines = g_path.read_text().splitlines()
            if not gpu_lines:
                continue
            gpu_reference_val = float(gpu_lines[0].strip())

            # Calculate the first rank value using the formula
            gpu_rank_value = (gpu_reference_val * sum_cpu_weights) / 100.0

            # Construct the new filename (e.g., w219-400_17.txt)
            # We use g_base[1:] to remove the leading 'w' from the second weight
            output_filename = f"{c_base}-{g_base[1:]}_17.txt"
            output_path = output_dir / output_filename

            # Prepare content for the new file
            content = cpu_lines + [f"{gpu_rank_value:.1f}"] 
            
            # Write the 17 ranks (1 GPU + 16 CPU)
            output_path.write_text("\n".join(content) + "\n")

    print("Generation complete.")

if __name__ == "__main__":
    generate_hybrid_weights()
