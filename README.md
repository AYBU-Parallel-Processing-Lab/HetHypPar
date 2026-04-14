# COMPILATION INSTRUCTIONS

## 1. Configure CMake

```bash
cmake -S src -B build -G "Ninja"
```
## 2. Build the Project

```bash
cmake --build build
```

# TOOLS

## Initialize Matrices

`tools/python/init_matrices.py` batch-initializes matrix entries in `data/matrices/`. It reads a text file containing matrix names (one per line), checks whether each matrix already has all required input files (`B.txt`, `X_init.txt`, `X_target.txt`), and runs Octave to generate any that are missing.

**Requirements:** GNU Octave must be installed and available on `PATH`.

**Usage:**

```bash
python tools/python/init_matrices.py <matrix_list.txt> [-s <source_dir>]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `matrix_list.txt` | Text file with newline-separated matrix names. Lines starting with `#` are ignored. |
| `-s`, `--source-dir` | Directory containing raw `.mtx` files. Default: `/matrices`. |

**Example:**

```bash
# Create a matrix list
echo -e "cage13\ncircuit5M_dc\nnv2" > my_matrices.txt

# Initialize using default source directory (/matrices)
python tools/python/init_matrices.py my_matrices.txt

# Initialize using a custom source directory
python tools/python/init_matrices.py my_matrices.txt -s /path/to/mtx/files
```

The script prints progress for each matrix (`[SKIP]` if already initialized or source not found, `[INIT]` if processing) and a summary at the end.


Note: Last benchmark was run with the following command:
```bash
python ~/Templates/results_medium/run_benchmarks.py --parts-basedir ~/Templates/results_medium --outdir data/results/ --commands-tsv ~/Templates/results_medium/commands.tsv
```