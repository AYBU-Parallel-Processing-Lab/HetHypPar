import os
from argparse import ArgumentParser
from pathlib import Path

from parallelbar import progress_map
from scipy.io import mminfo


def parse_weight(wpath: Path):
    wlist_str = wpath.read_text().split()
    return [float(wstr) for wstr in wlist_str]


def check_mtx(path_mtx: Path):
    numrows, numcols, nnz, format_type, field, symmetry = mminfo(path_mtx)

    result = True

    result &= numrows == numcols
    result &= symmetry == "general"
    result &= numrows > 0
    result &= nnz > 0
    result &= format_type == "coordinate"
    result &= field == "real"

    return result


def make_parser():
    parser = ArgumentParser()
    parser.add_argument("-w", "--rank-weight-path", type=Path)
    parser.add_argument("-o", "--outdir", type=Path)
    return parser


parser = make_parser()
args = parser.parse_args()

path_weight_dir = args.rank_weight_path
# path_weight_dir = Path("data/weights")
path_weights = list(path_weight_dir.glob("*.txt"))

path_weight = path_weights[0]
list_weight = parse_weight(path_weight)
num_parts = len(list_weight)
print(f"num parts = {num_parts}")

cmd_patpart = "patpart {matrix_path} {npart} {weight_path} {imbal_percent} {seed} {output_path} {log_path}"
seed = 42

path_data_dir: Path = Path("./data/matrices")


def format_cmd(matrix_name: str, path_rank_weight: Path, imbalance: int):
    nparts = len(parse_weight(path_rank_weight))

    path_matrix_dir: Path = path_data_dir / matrix_name
    path_in_dir = path_matrix_dir / "in"
    # path_out_dir = path_matrix_dir / "out"

    path_part_dir = path_in_dir / "part" / args.outdir
    path_part_dir.mkdir(parents=True, exist_ok=True)

    outname = f"{path_rank_weight.stem}_i{imbalance}"

    path_output = path_part_dir / f"{outname}.part"
    path_log = path_part_dir / f"{outname}.log"

    return cmd_patpart.format(
        matrix_path=f"/matrices/{matrix_name}.mtx",
        npart=nparts,
        weight_path=path_rank_weight,
        imbal_percent=imbalance,
        seed=seed,
        output_path=path_output,
        log_path=path_log,
    )


"""
matrices_dir = Path("/matrices")
mtx_files = sorted(matrices_dir.glob("*.mtx"))
nbefore = len(mtx_files)
mtx_files = [path_mtx for path_mtx in mtx_files if check_mtx(path_mtx)]
nafter = len(mtx_files)
print(f"Filtered {nbefore - nafter} out of {nbefore} matrices")
mtx_names = [path_mtx.stem for path_mtx in mtx_files]
"""
mtx_names = [i.name for i in path_data_dir.glob("*")]

# imbals = (0.10, 0.01, 0.03)
imbals = (0.01,)
imbal_ps = [int(i * 100) for i in imbals]

tasks = [
    format_cmd(mtx_name, path_weight, imbal)
    for mtx_name in mtx_names
    for path_weight in path_weights
    for imbal in imbal_ps
]

results = progress_map(os.system, tasks)

failures = {
    Path(task.split()[6]).parent.parent.parent
    for task, result in zip(tasks, results)
    if result != 0
}
print(f"Failed {len(failures)} out of {len(tasks)} tasks")


def rmrf(path: str | Path) -> None:
    """Recursively delete a directory and all its contents (like rm -rf)."""
    path = Path(path)

    if path.is_file() or path.is_symlink():
        path.unlink()
    elif path.is_dir():
        for child in path.iterdir():
            rmrf(child)
        path.rmdir()


# for failure in failures:
#     rmrf(failure)
