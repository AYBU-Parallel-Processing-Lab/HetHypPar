# Variant speedup vs single-GPU `bicgstab-gpu`  (iters=1000, best-of-2)

Best speedup across weights (hybrids/MPI) — see results.tsv for the full per-weight sweep.

| solver | cage12 | cage13 |
|---|---|---|
| `cpu` | 0.15× | 0.10× |
| `gpu` | 1.00× | 1.00× |
| `gpu-dp` | 1.21× | 1.19× |
| `gpu-pipelined` | 0.77× | 0.73× |
| `hybrid-async` | 1.06× | 1.07× |
| `hybrid-async-dp` | 1.29× | 1.28× |
| `hybrid-dist-dp` | 1.10× | 1.14× |
| `hybrid-dist-pipelined` | 0.86× | 0.92× |
| `mpi` | 0.22× | 0.14× |
| `mpi-gpu` | 0.67× | 0.70× |
