# Big benchmark — speedup vs single-GPU `bicgstab-gpu` (iters: 1000 small / 200 for nnz>15M, best-of-2)

## Best speedup per solver (across weights & partition source)

| solver | circuit5M_dc | circuit5M | cage15 | Freescale2 |
|---|---|---|---|---|
| `hybrid-async` | — | — | — | — |
| `hybrid-async-dp` | — | — | — | — |
| `hybrid-dist-dp` | — | — | — | — |
| `hybrid-dist-pipelined` | — | — | — | — |
| `mpi` | — | — | — | — |
| `mpi-gpu` | — | — | — | — |

## PaToH vs naive (best speedup each) — does cut-minimization help?

| matrix | solver | PaToH | naive | PaToH/naive |
|---|---|---|---|---|
