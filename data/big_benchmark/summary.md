# Big benchmark — summary

Best `spmv` loop time (best-of-2, best OMP threads) across weights & partition source. Two baselines: **old `bicgstab-gpu`** and **`bicgstab-gpu-dp`** (the device-pointer pure-GPU solver — the fairer 'is the CPU split worth it?' baseline).


## Speedup vs old `bicgstab-gpu`

| solver | Freescale2 | Hamrle3 | ML_Geer | RM07R | Transport | cage14 | cage15 | circuit5M | circuit5M_dc | dgreen | nv2 | rajat30_mc64_5 | ss | vas_stokes_1M |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `gpu` | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× |
| `gpu-dp` | 1.25× | 1.32× | 1.05× | 1.04× | 1.18× | 1.15× | 1.14× | 1.20× | 1.28× | 1.09× | 1.08× | 1.26× | 1.13× | 1.11× |
| `hybrid-async-dp` | 1.16× | 0.99× | 1.16× | 1.18× | 1.25× | 1.24× | 1.18× | 1.20× | 1.09× | 1.19× | 1.19× | 1.28× | 1.21× | 1.21× |
| `hybrid-dist-dp` | 1.23× | 1.22× | 1.16× | 1.14× | 1.22× | 1.18× | 1.16× | 1.20× | 1.27× | 1.18× | 1.19× | 1.24× | 1.20× | 1.20× |
| `hybrid-async` | 0.94× | 0.79× | 1.09× | 1.12× | 1.06× | 1.06× | 1.03× | 1.00× | 0.87× | 1.07× | 1.09× | 1.01× | 1.06× | 1.09× |
| `hybrid-dist-pipelined` | 0.87× | 0.84× | 0.96× | 0.97× | 0.92× | 0.91× | 0.89× | 0.88× | 0.88× | 0.95× | 0.96× | 0.91× | 0.94× | 0.95× |
| `gpu-pipelined` | 0.66× | 0.64× | 0.82× | 0.83× | 0.72× | 0.73× | 0.73× | 0.69× | 0.64× | 0.77× | 0.79× | 0.69× | 0.75× | 0.78× |
| `mpi-gpu` | 0.51× | 0.48× | 0.99× | 1.03× | 0.68× | 0.67× | 0.61× | 0.22× | 0.51× | 0.78× | 0.70× | 0.32× | 0.73× | 0.79× |
| `mpi` | 0.09× | 0.09× | 0.12× | 0.13× | 0.12× | 0.11× | 0.09× | 0.08× | 0.09× | 0.11× | 0.09× | 0.12× | 0.11× | 0.12× |
| `cpu` | 0.06× | 0.09× | 0.09× | 0.08× | 0.09× | 0.08× | 0.07× | 0.07× | 0.08× | 0.07× | 0.06× | 0.09× | 0.08× | 0.09× |

## Speedup vs `bicgstab-gpu-dp`

| solver | Freescale2 | Hamrle3 | ML_Geer | RM07R | Transport | cage14 | cage15 | circuit5M | circuit5M_dc | dgreen | nv2 | rajat30_mc64_5 | ss | vas_stokes_1M |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `gpu` | 0.80× | 0.76× | 0.95× | 0.96× | 0.85× | 0.87× | 0.87× | 0.83× | 0.78× | 0.91× | 0.93× | 0.80× | 0.89× | 0.90× |
| `gpu-dp` | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× |
| `hybrid-async-dp` | 0.93× | 0.75× | 1.10× | 1.14× | 1.06× | 1.07× | 1.03× | 1.00× | 0.85× | 1.09× | 1.10× | 1.02× | 1.08× | 1.09× |
| `hybrid-dist-dp` | 0.98× | 0.92× | 1.10× | 1.09× | 1.04× | 1.02× | 1.01× | 1.00× | 0.99× | 1.07× | 1.10× | 0.98× | 1.06× | 1.07× |
| `hybrid-async` | 0.75× | 0.60× | 1.04× | 1.08× | 0.90× | 0.92× | 0.90× | 0.83× | 0.68× | 0.98× | 1.01× | 0.81× | 0.94× | 0.98× |
| `hybrid-dist-pipelined` | 0.70× | 0.63× | 0.91× | 0.93× | 0.78× | 0.79× | 0.78× | 0.73× | 0.69× | 0.87× | 0.89× | 0.72× | 0.83× | 0.86× |
| `gpu-pipelined` | 0.53× | 0.48× | 0.78× | 0.79× | 0.61× | 0.64× | 0.64× | 0.57× | 0.50× | 0.70× | 0.73× | 0.55× | 0.66× | 0.70× |
| `mpi-gpu` | 0.41× | 0.36× | 0.94× | 0.99× | 0.58× | 0.58× | 0.53× | 0.18× | 0.39× | 0.71× | 0.65× | 0.26× | 0.65× | 0.71× |
| `mpi` | 0.07× | 0.07× | 0.12× | 0.12× | 0.10× | 0.09× | 0.08× | 0.07× | 0.07× | 0.10× | 0.08× | 0.10× | 0.10× | 0.11× |
| `cpu` | 0.05× | 0.07× | 0.08× | 0.08× | 0.07× | 0.07× | 0.07× | 0.06× | 0.06× | 0.06× | 0.06× | 0.07× | 0.07× | 0.08× |

## PaToH vs naive — best speedup each (vs old-gpu), ratio = patoh/naive

| matrix | solver | PaToH | naive | ratio |
|---|---|---|---|---|
| Freescale2 | `hybrid-async` | 0.94× | 0.92× | 1.02 |
| Freescale2 | `hybrid-async-dp` | 1.16× | 1.13× | 1.03 |
| Freescale2 | `hybrid-dist-dp` | 1.23× | 1.19× | 1.03 |
| Freescale2 | `hybrid-dist-pipelined` | 0.87× | 0.85× | 1.02 |
| Freescale2 | `mpi-gpu` | 0.51× | 0.47× | 1.09 |
| Hamrle3 | `hybrid-async` | 0.79× | 0.79× | 1.01 |
| Hamrle3 | `hybrid-async-dp` | 0.99× | 0.97× | 1.01 |
| Hamrle3 | `hybrid-dist-dp` | 1.18× | 1.22× | 0.97 |
| Hamrle3 | `hybrid-dist-pipelined` | 0.82× | 0.84× | 0.98 |
| Hamrle3 | `mpi-gpu` | 0.47× | 0.48× | 0.97 |
| ML_Geer | `hybrid-async` | 1.09× | 1.09× | 1.00 |
| ML_Geer | `hybrid-async-dp` | 1.16× | 1.15× | 1.01 |
| ML_Geer | `hybrid-dist-dp` | 1.15× | 1.16× | 0.99 |
| ML_Geer | `hybrid-dist-pipelined` | 0.96× | 0.96× | 1.00 |
| ML_Geer | `mpi-gpu` | 0.99× | 0.97× | 1.02 |
| RM07R | `hybrid-async` | 1.11× | 1.12× | 0.99 |
| RM07R | `hybrid-async-dp` | 1.18× | 1.17× | 1.01 |
| RM07R | `hybrid-dist-dp` | 1.06× | 1.14× | 0.93 |
| RM07R | `hybrid-dist-pipelined` | 0.92× | 0.97× | 0.95 |
| RM07R | `mpi-gpu` | 1.03× | 1.03× | 1.00 |
| Transport | `hybrid-async` | 1.06× | 1.05× | 1.00 |
| Transport | `hybrid-async-dp` | 1.25× | 1.25× | 1.00 |
| Transport | `hybrid-dist-dp` | 1.22× | 1.22× | 1.01 |
| Transport | `hybrid-dist-pipelined` | 0.92× | 0.92× | 1.00 |
| Transport | `mpi-gpu` | 0.68× | 0.67× | 1.03 |
| cage14 | `hybrid-async` | 1.06× | 1.06× | 1.00 |
| cage14 | `hybrid-async-dp` | 1.23× | 1.24× | 1.00 |
| cage14 | `hybrid-dist-dp` | 1.18× | 1.17× | 1.01 |
| cage14 | `hybrid-dist-pipelined` | 0.91× | 0.90× | 1.01 |
| cage14 | `mpi-gpu` | 0.67× | 0.65× | 1.03 |
| cage15 | `hybrid-async` | 1.03× | 1.03× | 1.00 |
| cage15 | `hybrid-async-dp` | 1.17× | 1.18× | 0.99 |
| cage15 | `hybrid-dist-dp` | 1.16× | 1.15× | 1.00 |
| cage15 | `hybrid-dist-pipelined` | 0.89× | 0.89× | 1.01 |
| cage15 | `mpi-gpu` | 0.61× | 0.57× | 1.06 |
| circuit5M | `hybrid-async` | 1.00× | 1.00× | 1.00 |
| circuit5M | `hybrid-async-dp` | 1.20× | 1.20× | 1.00 |
| circuit5M | `hybrid-dist-dp` | 1.20× | 1.18× | 1.02 |
| circuit5M | `hybrid-dist-pipelined` | 0.88× | 0.87× | 1.02 |
| circuit5M | `mpi-gpu` | 0.22× | 0.21× | 1.06 |
| circuit5M_dc | `hybrid-async` | 0.87× | 0.86× | 1.01 |
| circuit5M_dc | `hybrid-async-dp` | 1.09× | 1.08× | 1.01 |
| circuit5M_dc | `hybrid-dist-dp` | 1.27× | 1.23× | 1.04 |
| circuit5M_dc | `hybrid-dist-pipelined` | 0.88× | 0.85× | 1.03 |
| circuit5M_dc | `mpi-gpu` | 0.51× | 0.48× | 1.05 |
| dgreen | `hybrid-async` | 1.07× | 1.07× | 1.00 |
| dgreen | `hybrid-async-dp` | 1.19× | 1.19× | 1.00 |
| dgreen | `hybrid-dist-dp` | 1.18× | 1.10× | 1.06 |
| dgreen | `hybrid-dist-pipelined` | 0.95× | 0.89× | 1.07 |
| dgreen | `mpi-gpu` | 0.78× | 0.76× | 1.03 |
| nv2 | `hybrid-async` | 1.09× | 1.07× | 1.02 |
| nv2 | `hybrid-async-dp` | 1.19× | 1.17× | 1.02 |
| nv2 | `hybrid-dist-dp` | 1.19× | 1.11× | 1.08 |
| nv2 | `hybrid-dist-pipelined` | 0.96× | 0.90× | 1.07 |
| nv2 | `mpi-gpu` | 0.70× | 0.65× | 1.09 |
| rajat30_mc64_5 | `hybrid-async` | 1.01× | 1.00× | 1.01 |
| rajat30_mc64_5 | `hybrid-async-dp` | 1.28× | 1.25× | 1.02 |
| rajat30_mc64_5 | `hybrid-dist-dp` | 1.24× | 1.22× | 1.01 |
| rajat30_mc64_5 | `hybrid-dist-pipelined` | 0.91× | 0.91× | 1.00 |
| rajat30_mc64_5 | `mpi-gpu` | 0.32× | 0.27× | 1.17 |
| ss | `hybrid-async` | 1.06× | 1.05× | 1.01 |
| ss | `hybrid-async-dp` | 1.21× | 1.21× | 1.00 |
| ss | `hybrid-dist-dp` | 1.20× | 1.13× | 1.06 |
| ss | `hybrid-dist-pipelined` | 0.94× | 0.88× | 1.06 |
| ss | `mpi-gpu` | 0.73× | 0.68× | 1.07 |
| vas_stokes_1M | `hybrid-async` | 1.09× | 1.08× | 1.00 |
| vas_stokes_1M | `hybrid-async-dp` | 1.21× | 1.20× | 1.01 |
| vas_stokes_1M | `hybrid-dist-dp` | 1.20× | 1.14× | 1.05 |
| vas_stokes_1M | `hybrid-dist-pipelined` | 0.95× | 0.92× | 1.04 |
| vas_stokes_1M | `mpi-gpu` | 0.79× | 0.77× | 1.04 |

### Mean PaToH/naive ratio per solver

| solver | mean ratio | n matrices |
|---|---|---|
| `hybrid-async` | 1.004 | 14 |
| `hybrid-async-dp` | 1.008 | 14 |
| `hybrid-dist-dp` | 1.019 | 14 |
| `hybrid-dist-pipelined` | 1.019 | 14 |
| `mpi-gpu` | 1.050 | 14 |

## Notes

- Usable matrices: 14. Fully-failed (excluded): ['nlpkkt120'] (solver-level crash).
- `speedup_vs_gpudp` is the honest 'does the heterogeneous CPU split beat just running pure GPU device-pointer?' baseline.
