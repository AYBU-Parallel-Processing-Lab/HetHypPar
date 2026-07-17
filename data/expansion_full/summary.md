# Big benchmark — summary

Best `spmv` loop time (best-of-2, best OMP threads) across weights & partition source. Two baselines: **old `bicgstab-gpu`** and **`bicgstab-gpu-dp`** (the device-pointer pure-GPU solver — the fairer 'is the CPU split worth it?' baseline).


## Speedup vs old `bicgstab-gpu`

| solver | FEM_3D_thermal2 | Freescale2 | FullChip | Goodwin_127 | Hamrle3 | Ill_Stokes | ML_Geer | ML_Laplace | RM07R | TSOPF_RS_b2383_c1 | Transport | Zd_Jac3 | appu | bauru5727 | bayer02 | cage13 | cage14 | cage15 | circuit5M | circuit5M_dc | coupled | dgreen | inlet | k3plates | largebasis | mac_econ_fwd500 | marine1 | nv2 | nxp1_mc64_1 | para-5 | para-9 | poli4 | powersim | pre2 | rajat30_mc64_5 | shar_te2-b3 | ss | ss1 | test1 | tmt_unsym | torso1 | trans4 | trans5 | vas_stokes_1M | venkat01 | venkat50 | wang3 | webbase-1M | xenon2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `gpu` | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× |
| `gpu-dp` | 1.18× | 1.25× | 1.22× | 1.14× | 1.32× | 1.49× | 1.05× | 1.04× | 1.04× | 1.04× | 1.18× | 1.22× | 1.22× | 1.50× | 1.65× | 1.19× | 1.15× | 1.14× | 1.20× | 1.28× | 1.66× | 1.09× | 1.48× | 1.48× | 1.22× | 1.26× | 1.17× | 1.08× | 1.26× | 1.14× | 1.15× | 1.57× | 1.67× | 1.27× | 1.26× | 1.30× | 1.13× | 1.28× | 1.12× | 1.33× | 1.12× | 1.35× | 1.34× | 1.11× | 1.15× | 1.15× | 1.55× | 1.38× | 1.16× |
| `hybrid-async-dp` | 1.33× | 1.16× | 1.17× | 1.29× | 0.99× | 1.66× | 1.16× | 1.20× | 1.18× | 1.21× | 1.25× | 1.37× | 1.42× | 1.31× | 1.58× | 1.28× | 1.24× | 1.18× | 1.20× | 1.09× | 1.80× | 1.19× | 1.58× | 1.65× | 1.30× | 1.13× | 1.28× | 1.19× | 1.10× | 1.30× | 1.30× | 1.39× | 1.58× | 1.27× | 1.28× | 0.94× | 1.21× | 0.97× | 1.26× | 1.05× | 1.32× | 1.22× | 1.21× | 1.21× | 1.32× | 1.30× | 1.57× | 0.91× | 1.31× |
| `hybrid-dist-dp` | 1.15× | 1.23× | 1.22× | 1.14× | 1.22× | 1.12× | 1.16× | 1.15× | 1.14× | 1.12× | 1.22× | 1.06× | 1.07× | 1.07× | 1.08× | 1.15× | 1.18× | 1.16× | 1.20× | 1.27× | 1.18× | 1.18× | 1.07× | 1.12× | 1.19× | 1.12× | 1.18× | 1.19× | 1.17× | 1.16× | 1.14× | 1.31× | 1.09× | 1.25× | 1.24× | 1.08× | 1.20× | 1.11× | 1.15× | 1.29× | 1.17× | 1.16× | 1.16× | 1.20× | 1.03× | 1.03× | 1.10× | 1.30× | 1.13× |
| `hybrid-async` | 1.12× | 0.94× | 0.97× | 1.14× | 0.79× | 1.02× | 1.09× | 1.13× | 1.12× | 1.14× | 1.06× | 1.07× | 1.14× | 0.90× | 0.95× | 1.07× | 1.06× | 1.03× | 1.00× | 0.87× | 1.01× | 1.07× | 1.09× | 1.12× | 1.05× | 0.87× | 1.06× | 1.09× | 0.89× | 1.13× | 1.13× | 0.90× | 0.95× | 1.01× | 1.01× | 0.74× | 1.06× | 0.76× | 1.12× | 0.82× | 1.17× | 0.89× | 0.89× | 1.09× | 1.07× | 1.06× | 0.95× | 0.73× | 1.11× |
| `hybrid-dist-pipelined` | 0.94× | 0.87× | 0.89× | 0.95× | 0.84× | 0.97× | 0.96× | 0.97× | 0.97× | 0.97× | 0.92× | 0.92× | 0.92× | 0.90× | 0.94× | 0.91× | 0.91× | 0.89× | 0.88× | 0.88× | 1.03× | 0.95× | 0.95× | 0.98× | 0.92× | 0.81× | 0.92× | 0.96× | 0.85× | 0.96× | 0.96× | 0.77× | 0.93× | 0.92× | 0.91× | 0.76× | 0.94× | 0.80× | 0.96× | 0.90× | 0.98× | 0.88× | 0.87× | 0.95× | 0.87× | 0.87× | 0.94× | 0.88× | 0.93× |
| `gpu-pipelined` | 0.79× | 0.66× | 0.68× | 0.80× | 0.64× | 0.76× | 0.82× | 0.82× | 0.83× | 0.86× | 0.72× | 0.80× | 0.81× | 0.76× | 0.74× | 0.73× | 0.73× | 0.73× | 0.69× | 0.64× | 0.75× | 0.77× | 0.77× | 0.77× | 0.71× | 0.75× | 0.72× | 0.79× | 0.66× | 0.81× | 0.81× | 0.77× | 0.74× | 0.68× | 0.69× | 0.73× | 0.75× | 0.74× | 0.78× | 0.65× | 0.82× | 0.74× | 0.74× | 0.78× | 0.79× | 0.78× | 0.78× | 0.63× | 0.79× |
| `mpi-gpu` | 0.82× | 0.51× | 0.21× | 0.88× | 0.48× | 0.69× | 0.99× | 1.07× | 1.03× | 0.68× | 0.68× | 0.80× | 0.85× | 0.62× | 0.61× | 0.69× | 0.67× | 0.61× | 0.22× | 0.51× | 0.66× | 0.78× | 0.77× | 0.77× | 0.72× | 0.59× | 0.73× | 0.70× | 0.53× | 0.85× | 0.85× | 0.59× | 0.64× | 0.62× | 0.32× | 0.44× | 0.73× | 0.57× | 0.83× | 0.55× | 0.72× | 0.29× | 0.29× | 0.79× | 0.81× | 0.81× | 0.66× | 0.49× | 0.82× |
| `mpi` | 0.13× | 0.09× | 0.07× | 0.12× | 0.09× | 0.61× | 0.12× | 0.10× | 0.13× | 0.11× | 0.12× | 0.21× | 0.17× | 0.45× | 1.18× | 0.10× | 0.11× | 0.09× | 0.08× | 0.09× | 0.72× | 0.11× | 0.62× | 0.51× | 0.12× | 0.18× | 0.10× | 0.09× | 0.10× | 0.10× | 0.10× | 0.60× | 1.15× | 0.10× | 0.12× | 0.17× | 0.11× | 0.27× | 0.09× | 0.10× | 0.09× | 0.23× | 0.24× | 0.12× | 0.20× | 0.20× | 0.67× | 0.10× | 0.13× |
| `cpu` | 0.13× | 0.06× | 0.07× | 0.11× | 0.09× | 0.55× | 0.09× | 0.09× | 0.08× | 0.07× | 0.09× | 0.16× | 0.17× | 0.41× | 1.30× | 0.10× | 0.08× | 0.07× | 0.07× | 0.08× | 0.64× | 0.07× | 0.68× | 0.58× | 0.11× | 0.17× | 0.10× | 0.06× | 0.09× | 0.10× | 0.10× | 0.54× | 1.06× | 0.10× | 0.09× | 0.20× | 0.08× | 0.18× | 0.08× | 0.10× | 0.08× | 0.25× | 0.25× | 0.09× | 0.20× | 0.21× | 0.66× | 0.09× | 0.12× |

## Speedup vs `bicgstab-gpu-dp`

| solver | FEM_3D_thermal2 | Freescale2 | FullChip | Goodwin_127 | Hamrle3 | Ill_Stokes | ML_Geer | ML_Laplace | RM07R | TSOPF_RS_b2383_c1 | Transport | Zd_Jac3 | appu | bauru5727 | bayer02 | cage13 | cage14 | cage15 | circuit5M | circuit5M_dc | coupled | dgreen | inlet | k3plates | largebasis | mac_econ_fwd500 | marine1 | nv2 | nxp1_mc64_1 | para-5 | para-9 | poli4 | powersim | pre2 | rajat30_mc64_5 | shar_te2-b3 | ss | ss1 | test1 | tmt_unsym | torso1 | trans4 | trans5 | vas_stokes_1M | venkat01 | venkat50 | wang3 | webbase-1M | xenon2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `gpu` | 0.85× | 0.80× | 0.82× | 0.88× | 0.76× | 0.67× | 0.95× | 0.96× | 0.96× | 0.96× | 0.85× | 0.82× | 0.82× | 0.67× | 0.61× | 0.84× | 0.87× | 0.87× | 0.83× | 0.78× | 0.60× | 0.91× | 0.68× | 0.68× | 0.82× | 0.80× | 0.85× | 0.93× | 0.80× | 0.88× | 0.87× | 0.64× | 0.60× | 0.79× | 0.80× | 0.77× | 0.89× | 0.78× | 0.90× | 0.75× | 0.90× | 0.74× | 0.74× | 0.90× | 0.87× | 0.87× | 0.65× | 0.72× | 0.87× |
| `gpu-dp` | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× |
| `hybrid-async-dp` | 1.13× | 0.93× | 0.96× | 1.13× | 0.75× | 1.12× | 1.10× | 1.16× | 1.14× | 1.16× | 1.06× | 1.12× | 1.17× | 0.87× | 0.96× | 1.07× | 1.07× | 1.03× | 1.00× | 0.85× | 1.09× | 1.09× | 1.07× | 1.11× | 1.07× | 0.90× | 1.09× | 1.10× | 0.88× | 1.14× | 1.13× | 0.89× | 0.95× | 0.99× | 1.02× | 0.73× | 1.08× | 0.75× | 1.13× | 0.79× | 1.18× | 0.91× | 0.90× | 1.09× | 1.15× | 1.14× | 1.01× | 0.66× | 1.14× |
| `hybrid-dist-dp` | 0.97× | 0.98× | 1.00× | 1.00× | 0.92× | 0.76× | 1.10× | 1.10× | 1.09× | 1.07× | 1.04× | 0.87× | 0.88× | 0.71× | 0.66× | 0.96× | 1.02× | 1.01× | 1.00× | 0.99× | 0.71× | 1.07× | 0.72× | 0.76× | 0.98× | 0.90× | 1.01× | 1.10× | 0.93× | 1.02× | 0.99× | 0.84× | 0.65× | 0.98× | 0.98× | 0.83× | 1.06× | 0.87× | 1.03× | 0.97× | 1.05× | 0.86× | 0.86× | 1.07× | 0.90× | 0.90× | 0.71× | 0.94× | 0.98× |
| `hybrid-async` | 0.95× | 0.75× | 0.80× | 1.00× | 0.60× | 0.69× | 1.04× | 1.08× | 1.08× | 1.10× | 0.90× | 0.88× | 0.93× | 0.60× | 0.58× | 0.90× | 0.92× | 0.90× | 0.83× | 0.68× | 0.61× | 0.98× | 0.73× | 0.76× | 0.87× | 0.69× | 0.90× | 1.01× | 0.71× | 0.99× | 0.98× | 0.57× | 0.57× | 0.79× | 0.81× | 0.57× | 0.94× | 0.59× | 1.00× | 0.62× | 1.04× | 0.66× | 0.66× | 0.98× | 0.93× | 0.93× | 0.61× | 0.53× | 0.96× |
| `hybrid-dist-pipelined` | 0.80× | 0.70× | 0.73× | 0.83× | 0.63× | 0.65× | 0.91× | 0.93× | 0.93× | 0.94× | 0.78× | 0.75× | 0.76× | 0.60× | 0.57× | 0.77× | 0.79× | 0.78× | 0.73× | 0.69× | 0.62× | 0.87× | 0.64× | 0.66× | 0.75× | 0.65× | 0.79× | 0.89× | 0.68× | 0.84× | 0.83× | 0.49× | 0.56× | 0.72× | 0.72× | 0.59× | 0.83× | 0.63× | 0.86× | 0.68× | 0.88× | 0.65× | 0.65× | 0.86× | 0.76× | 0.76× | 0.61× | 0.63× | 0.80× |
| `gpu-pipelined` | 0.67× | 0.53× | 0.56× | 0.71× | 0.48× | 0.51× | 0.78× | 0.79× | 0.79× | 0.82× | 0.61× | 0.65× | 0.66× | 0.50× | 0.45× | 0.61× | 0.64× | 0.64× | 0.57× | 0.50× | 0.45× | 0.70× | 0.52× | 0.52× | 0.58× | 0.60× | 0.62× | 0.73× | 0.53× | 0.71× | 0.70× | 0.49× | 0.45× | 0.54× | 0.55× | 0.57× | 0.66× | 0.58× | 0.70× | 0.49× | 0.74× | 0.55× | 0.55× | 0.70× | 0.69× | 0.68× | 0.50× | 0.45× | 0.69× |
| `mpi-gpu` | 0.70× | 0.41× | 0.18× | 0.77× | 0.36× | 0.46× | 0.94× | 1.03× | 0.99× | 0.65× | 0.58× | 0.65× | 0.70× | 0.41× | 0.37× | 0.58× | 0.58× | 0.53× | 0.18× | 0.39× | 0.40× | 0.71× | 0.52× | 0.52× | 0.60× | 0.47× | 0.63× | 0.65× | 0.42× | 0.75× | 0.74× | 0.37× | 0.39× | 0.49× | 0.26× | 0.34× | 0.65× | 0.44× | 0.74× | 0.41× | 0.65× | 0.22× | 0.22× | 0.71× | 0.71× | 0.71× | 0.43× | 0.36× | 0.71× |
| `mpi` | 0.11× | 0.07× | 0.06× | 0.10× | 0.07× | 0.41× | 0.12× | 0.10× | 0.12× | 0.11× | 0.10× | 0.17× | 0.14× | 0.30× | 0.72× | 0.09× | 0.09× | 0.08× | 0.07× | 0.07× | 0.43× | 0.10× | 0.42× | 0.35× | 0.10× | 0.15× | 0.09× | 0.08× | 0.08× | 0.09× | 0.09× | 0.38× | 0.69× | 0.08× | 0.10× | 0.13× | 0.10× | 0.21× | 0.08× | 0.08× | 0.08× | 0.17× | 0.18× | 0.11× | 0.18× | 0.18× | 0.43× | 0.07× | 0.12× |
| `cpu` | 0.11× | 0.05× | 0.06× | 0.09× | 0.07× | 0.37× | 0.08× | 0.09× | 0.08× | 0.07× | 0.07× | 0.14× | 0.14× | 0.27× | 0.79× | 0.08× | 0.07× | 0.07× | 0.06× | 0.06× | 0.39× | 0.06× | 0.46× | 0.39× | 0.09× | 0.13× | 0.08× | 0.06× | 0.07× | 0.09× | 0.09× | 0.34× | 0.64× | 0.08× | 0.07× | 0.15× | 0.07× | 0.14× | 0.07× | 0.07× | 0.07× | 0.18× | 0.18× | 0.08× | 0.18× | 0.18× | 0.42× | 0.07× | 0.11× |

## PaToH vs naive — best speedup each (vs old-gpu), ratio = patoh/naive

| matrix | solver | PaToH | naive | ratio |
|---|---|---|---|---|
| FEM_3D_thermal2 | `hybrid-async` | 1.12× | 1.12× | 1.00 |
| FEM_3D_thermal2 | `hybrid-async-dp` | 1.32× | 1.33× | 1.00 |
| FEM_3D_thermal2 | `hybrid-dist-dp` | 1.15× | 1.14× | 1.01 |
| FEM_3D_thermal2 | `hybrid-dist-pipelined` | 0.94× | 0.94× | 0.99 |
| FEM_3D_thermal2 | `mpi-gpu` | 0.82× | 0.82× | 1.00 |
| Freescale2 | `hybrid-async` | 0.94× | 0.92× | 1.02 |
| Freescale2 | `hybrid-async-dp` | 1.16× | 1.13× | 1.03 |
| Freescale2 | `hybrid-dist-dp` | 1.23× | 1.19× | 1.03 |
| Freescale2 | `hybrid-dist-pipelined` | 0.87× | 0.85× | 1.02 |
| Freescale2 | `mpi-gpu` | 0.51× | 0.47× | 1.09 |
| FullChip | `hybrid-async` | 0.97× | 0.97× | 1.01 |
| FullChip | `hybrid-async-dp` | 1.17× | 1.17× | 1.00 |
| FullChip | `hybrid-dist-dp` | 1.22× | 1.20× | 1.02 |
| FullChip | `hybrid-dist-pipelined` | 0.89× | 0.88× | 1.01 |
| FullChip | `mpi-gpu` | 0.21× | 0.21× | 0.99 |
| Goodwin_127 | `hybrid-async` | 1.14× | 1.13× | 1.01 |
| Goodwin_127 | `hybrid-async-dp` | 1.29× | 1.27× | 1.01 |
| Goodwin_127 | `hybrid-dist-dp` | 1.14× | 1.14× | 1.00 |
| Goodwin_127 | `hybrid-dist-pipelined` | 0.94× | 0.95× | 0.99 |
| Goodwin_127 | `mpi-gpu` | 0.87× | 0.88× | 0.99 |
| Hamrle3 | `hybrid-async` | 0.79× | 0.79× | 1.01 |
| Hamrle3 | `hybrid-async-dp` | 0.99× | 0.97× | 1.01 |
| Hamrle3 | `hybrid-dist-dp` | 1.18× | 1.22× | 0.97 |
| Hamrle3 | `hybrid-dist-pipelined` | 0.82× | 0.84× | 0.98 |
| Hamrle3 | `mpi-gpu` | 0.47× | 0.48× | 0.97 |
| Ill_Stokes | `hybrid-async` | 1.02× | 0.97× | 1.06 |
| Ill_Stokes | `hybrid-async-dp` | 1.66× | 1.55× | 1.07 |
| Ill_Stokes | `hybrid-dist-dp` | 1.12× | 1.12× | 1.00 |
| Ill_Stokes | `hybrid-dist-pipelined` | 0.97× | 0.95× | 1.01 |
| Ill_Stokes | `mpi-gpu` | 0.69× | 0.64× | 1.08 |
| ML_Geer | `hybrid-async` | 1.09× | 1.09× | 1.00 |
| ML_Geer | `hybrid-async-dp` | 1.16× | 1.15× | 1.01 |
| ML_Geer | `hybrid-dist-dp` | 1.15× | 1.16× | 0.99 |
| ML_Geer | `hybrid-dist-pipelined` | 0.96× | 0.96× | 1.00 |
| ML_Geer | `mpi-gpu` | 0.99× | 0.97× | 1.02 |
| ML_Laplace | `hybrid-async` | 1.13× | 1.13× | 1.00 |
| ML_Laplace | `hybrid-async-dp` | 1.19× | 1.20× | 0.99 |
| ML_Laplace | `hybrid-dist-dp` | 1.15× | 1.14× | 1.01 |
| ML_Laplace | `hybrid-dist-pipelined` | 0.97× | 0.96× | 1.01 |
| ML_Laplace | `mpi-gpu` | 1.07× | 1.07× | 1.00 |
| RM07R | `hybrid-async` | 1.11× | 1.12× | 0.99 |
| RM07R | `hybrid-async-dp` | 1.18× | 1.17× | 1.01 |
| RM07R | `hybrid-dist-dp` | 1.06× | 1.14× | 0.93 |
| RM07R | `hybrid-dist-pipelined` | 0.92× | 0.97× | 0.95 |
| RM07R | `mpi-gpu` | 1.03× | 1.03× | 1.00 |
| TSOPF_RS_b2383_c1 | `hybrid-async` | 1.12× | 1.14× | 0.98 |
| TSOPF_RS_b2383_c1 | `hybrid-async-dp` | 1.18× | 1.21× | 0.98 |
| TSOPF_RS_b2383_c1 | `hybrid-dist-dp` | 1.10× | 1.12× | 0.98 |
| TSOPF_RS_b2383_c1 | `hybrid-dist-pipelined` | 0.96× | 0.97× | 0.98 |
| TSOPF_RS_b2383_c1 | `mpi-gpu` | 0.68× | 0.68× | 1.00 |
| Transport | `hybrid-async` | 1.06× | 1.05× | 1.00 |
| Transport | `hybrid-async-dp` | 1.25× | 1.25× | 1.00 |
| Transport | `hybrid-dist-dp` | 1.22× | 1.22× | 1.01 |
| Transport | `hybrid-dist-pipelined` | 0.92× | 0.92× | 1.00 |
| Transport | `mpi-gpu` | 0.68× | 0.67× | 1.03 |
| Zd_Jac3 | `hybrid-async` | 1.06× | 1.07× | 0.99 |
| Zd_Jac3 | `hybrid-async-dp` | 1.33× | 1.37× | 0.97 |
| Zd_Jac3 | `hybrid-dist-dp` | 1.06× | 1.05× | 1.01 |
| Zd_Jac3 | `hybrid-dist-pipelined` | 0.92× | 0.90× | 1.02 |
| Zd_Jac3 | `mpi-gpu` | 0.77× | 0.80× | 0.96 |
| appu | `hybrid-async` | 1.13× | 1.14× | 0.99 |
| appu | `hybrid-async-dp` | 1.42× | 1.41× | 1.01 |
| appu | `hybrid-dist-dp` | 1.07× | 1.06× | 1.02 |
| appu | `hybrid-dist-pipelined` | 0.92× | 0.91× | 1.02 |
| appu | `mpi-gpu` | 0.85× | 0.82× | 1.03 |
| bauru5727 | `hybrid-async` | 0.90× | 0.90× | 1.00 |
| bauru5727 | `hybrid-async-dp` | 1.26× | 1.31× | 0.96 |
| bauru5727 | `hybrid-dist-dp` | 1.04× | 1.07× | 0.97 |
| bauru5727 | `hybrid-dist-pipelined` | 0.89× | 0.90× | 0.98 |
| bauru5727 | `mpi-gpu` | 0.62× | 0.57× | 1.09 |
| bayer02 | `hybrid-async` | 0.94× | 0.95× | 0.99 |
| bayer02 | `hybrid-async-dp` | 1.57× | 1.58× | 0.99 |
| bayer02 | `hybrid-dist-dp` | 1.08× | 1.08× | 1.00 |
| bayer02 | `hybrid-dist-pipelined` | 0.94× | 0.94× | 1.00 |
| bayer02 | `mpi-gpu` | 0.61× | 0.61× | 1.00 |
| cage13 | `hybrid-async` | 1.06× | 1.07× | 0.99 |
| cage13 | `hybrid-async-dp` | 1.28× | 1.28× | 1.00 |
| cage13 | `hybrid-dist-dp` | 1.15× | 1.14× | 1.01 |
| cage13 | `hybrid-dist-pipelined` | 0.91× | 0.89× | 1.02 |
| cage13 | `mpi-gpu` | 0.69× | 0.68× | 1.03 |
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
| coupled | `hybrid-async` | 1.01× | 0.96× | 1.05 |
| coupled | `hybrid-async-dp` | 1.80× | 1.72× | 1.04 |
| coupled | `hybrid-dist-dp` | 1.17× | 1.18× | 0.99 |
| coupled | `hybrid-dist-pipelined` | 1.03× | 1.02× | 1.01 |
| coupled | `mpi-gpu` | 0.66× | 0.64× | 1.04 |
| dgreen | `hybrid-async` | 1.07× | 1.07× | 1.00 |
| dgreen | `hybrid-async-dp` | 1.19× | 1.19× | 1.00 |
| dgreen | `hybrid-dist-dp` | 1.18× | 1.10× | 1.06 |
| dgreen | `hybrid-dist-pipelined` | 0.95× | 0.89× | 1.07 |
| dgreen | `mpi-gpu` | 0.78× | 0.76× | 1.03 |
| inlet | `hybrid-async` | 1.09× | 1.09× | 1.00 |
| inlet | `hybrid-async-dp` | 1.58× | 1.58× | 1.00 |
| inlet | `hybrid-dist-dp` | 1.07× | 1.07× | 1.00 |
| inlet | `hybrid-dist-pipelined` | 0.94× | 0.95× | 0.99 |
| inlet | `mpi-gpu` | 0.77× | 0.76× | 1.01 |
| k3plates | `hybrid-async` | 1.12× | 1.12× | 0.99 |
| k3plates | `hybrid-async-dp` | 1.65× | 1.64× | 1.01 |
| k3plates | `hybrid-dist-dp` | 1.12× | 1.12× | 1.00 |
| k3plates | `hybrid-dist-pipelined` | 0.98× | 0.97× | 1.01 |
| k3plates | `mpi-gpu` | 0.77× | 0.77× | 1.00 |
| largebasis | `hybrid-async` | 1.05× | 1.05× | 1.00 |
| largebasis | `hybrid-async-dp` | 1.30× | 1.29× | 1.01 |
| largebasis | `hybrid-dist-dp` | 1.19× | 1.13× | 1.05 |
| largebasis | `hybrid-dist-pipelined` | 0.92× | 0.85× | 1.07 |
| largebasis | `mpi-gpu` | 0.72× | 0.61× | 1.18 |
| mac_econ_fwd500 | `hybrid-async` | 0.87× | 0.87× | 1.01 |
| mac_econ_fwd500 | `hybrid-async-dp` | 1.13× | 1.13× | 1.00 |
| mac_econ_fwd500 | `hybrid-dist-dp` | 1.12× | 1.11× | 1.01 |
| mac_econ_fwd500 | `hybrid-dist-pipelined` | 0.81× | 0.81× | 0.99 |
| mac_econ_fwd500 | `mpi-gpu` | 0.59× | 0.59× | 1.00 |
| marine1 | `hybrid-async` | 1.06× | 1.06× | 1.00 |
| marine1 | `hybrid-async-dp` | 1.28× | 1.26× | 1.01 |
| marine1 | `hybrid-dist-dp` | 1.18× | 1.18× | 0.99 |
| marine1 | `hybrid-dist-pipelined` | 0.92× | 0.91× | 1.02 |
| marine1 | `mpi-gpu` | 0.73× | 0.68× | 1.07 |
| nv2 | `hybrid-async` | 1.09× | 1.07× | 1.02 |
| nv2 | `hybrid-async-dp` | 1.19× | 1.17× | 1.02 |
| nv2 | `hybrid-dist-dp` | 1.19× | 1.11× | 1.08 |
| nv2 | `hybrid-dist-pipelined` | 0.96× | 0.90× | 1.07 |
| nv2 | `mpi-gpu` | 0.70× | 0.65× | 1.09 |
| nxp1_mc64_1 | `hybrid-async` | 0.89× | 0.81× | 1.09 |
| nxp1_mc64_1 | `hybrid-async-dp` | 1.10× | 1.01× | 1.09 |
| nxp1_mc64_1 | `hybrid-dist-dp` | 1.17× | 1.12× | 1.05 |
| nxp1_mc64_1 | `hybrid-dist-pipelined` | 0.85× | 0.81× | 1.05 |
| nxp1_mc64_1 | `mpi-gpu` | 0.50× | 0.53× | 0.94 |
| para-5 | `hybrid-async` | 1.13× | 1.13× | 1.00 |
| para-5 | `hybrid-async-dp` | 1.30× | 1.30× | 1.00 |
| para-5 | `hybrid-dist-dp` | 1.16× | 1.08× | 1.08 |
| para-5 | `hybrid-dist-pipelined` | 0.96× | 0.88× | 1.09 |
| para-5 | `mpi-gpu` | 0.85× | 0.69× | 1.25 |
| para-9 | `hybrid-async` | 1.12× | 1.13× | 0.99 |
| para-9 | `hybrid-async-dp` | 1.29× | 1.30× | 0.99 |
| para-9 | `hybrid-dist-dp` | 1.14× | 1.07× | 1.06 |
| para-9 | `hybrid-dist-pipelined` | 0.96× | 0.88× | 1.09 |
| para-9 | `mpi-gpu` | 0.85× | 0.68× | 1.26 |
| poli4 | `hybrid-async` | 0.90× | 0.88× | 1.03 |
| poli4 | `hybrid-async-dp` | 1.39× | 1.24× | 1.12 |
| poli4 | `hybrid-dist-dp` | 1.31× | 1.29× | 1.02 |
| poli4 | `hybrid-dist-pipelined` | 0.77× | 0.75× | 1.02 |
| poli4 | `mpi-gpu` | 0.59× | 0.56× | 1.05 |
| powersim | `hybrid-async` | 0.95× | 0.95× | 1.01 |
| powersim | `hybrid-async-dp` | 1.58× | 1.54× | 1.02 |
| powersim | `hybrid-dist-dp` | 1.08× | 1.09× | 1.00 |
| powersim | `hybrid-dist-pipelined` | 0.93× | 0.93× | 1.00 |
| powersim | `mpi-gpu` | 0.64× | 0.61× | 1.06 |
| pre2 | `hybrid-async` | 1.01× | 0.99× | 1.02 |
| pre2 | `hybrid-async-dp` | 1.27× | 1.25× | 1.01 |
| pre2 | `hybrid-dist-dp` | 1.25× | 1.24× | 1.01 |
| pre2 | `hybrid-dist-pipelined` | 0.92× | 0.91× | 1.00 |
| pre2 | `mpi-gpu` | 0.62× | 0.61× | 1.02 |
| rajat30_mc64_5 | `hybrid-async` | 1.01× | 1.00× | 1.01 |
| rajat30_mc64_5 | `hybrid-async-dp` | 1.28× | 1.25× | 1.02 |
| rajat30_mc64_5 | `hybrid-dist-dp` | 1.24× | 1.22× | 1.01 |
| rajat30_mc64_5 | `hybrid-dist-pipelined` | 0.91× | 0.91× | 1.00 |
| rajat30_mc64_5 | `mpi-gpu` | 0.32× | 0.27× | 1.17 |
| shar_te2-b3 | `hybrid-async` | 0.72× | 0.74× | 0.98 |
| shar_te2-b3 | `hybrid-async-dp` | 0.92× | 0.94× | 0.98 |
| shar_te2-b3 | `hybrid-dist-dp` | 1.08× | 1.07× | 1.01 |
| shar_te2-b3 | `hybrid-dist-pipelined` | 0.76× | 0.75× | 1.02 |
| shar_te2-b3 | `mpi-gpu` | 0.43× | 0.44× | 0.99 |
| ss | `hybrid-async` | 1.06× | 1.05× | 1.01 |
| ss | `hybrid-async-dp` | 1.21× | 1.21× | 1.00 |
| ss | `hybrid-dist-dp` | 1.20× | 1.13× | 1.06 |
| ss | `hybrid-dist-pipelined` | 0.94× | 0.88× | 1.06 |
| ss | `mpi-gpu` | 0.73× | 0.68× | 1.07 |
| ss1 | `hybrid-async` | 0.72× | 0.76× | 0.96 |
| ss1 | `hybrid-async-dp` | 0.93× | 0.97× | 0.96 |
| ss1 | `hybrid-dist-dp` | 1.06× | 1.11× | 0.96 |
| ss1 | `hybrid-dist-pipelined` | 0.80× | 0.78× | 1.03 |
| ss1 | `mpi-gpu` | 0.57× | 0.48× | 1.18 |
| test1 | `hybrid-async` | 1.12× | 1.11× | 1.00 |
| test1 | `hybrid-async-dp` | 1.26× | 1.25× | 1.01 |
| test1 | `hybrid-dist-dp` | 1.15× | 1.09× | 1.05 |
| test1 | `hybrid-dist-pipelined` | 0.96× | 0.89× | 1.08 |
| test1 | `mpi-gpu` | 0.83× | 0.77× | 1.07 |
| tmt_unsym | `hybrid-async` | 0.82× | 0.82× | 1.01 |
| tmt_unsym | `hybrid-async-dp` | 1.03× | 1.05× | 0.97 |
| tmt_unsym | `hybrid-dist-dp` | 1.29× | 1.29× | 1.00 |
| tmt_unsym | `hybrid-dist-pipelined` | 0.90× | 0.90× | 1.00 |
| tmt_unsym | `mpi-gpu` | 0.55× | 0.55× | 1.00 |
| torso1 | `hybrid-async` | 1.17× | 1.16× | 1.01 |
| torso1 | `hybrid-async-dp` | 1.32× | 1.30× | 1.02 |
| torso1 | `hybrid-dist-dp` | 1.17× | 1.16× | 1.01 |
| torso1 | `hybrid-dist-pipelined` | 0.98× | 0.97× | 1.01 |
| torso1 | `mpi-gpu` | 0.72× | 0.70× | 1.03 |
| trans4 | `hybrid-async` | 0.76× | 0.89× | 0.86 |
| trans4 | `hybrid-async-dp` | 0.95× | 1.22× | 0.78 |
| trans4 | `hybrid-dist-dp` | 0.97× | 1.16× | 0.84 |
| trans4 | `hybrid-dist-pipelined` | 0.78× | 0.88× | 0.89 |
| trans4 | `mpi-gpu` | 0.29× | 0.29× | 1.00 |
| trans5 | `hybrid-async` | 0.78× | 0.89× | 0.87 |
| trans5 | `hybrid-async-dp` | 0.96× | 1.21× | 0.80 |
| trans5 | `hybrid-dist-dp` | 0.96× | 1.16× | 0.83 |
| trans5 | `hybrid-dist-pipelined` | 0.77× | 0.87× | 0.88 |
| trans5 | `mpi-gpu` | 0.29× | 0.29× | 0.99 |
| vas_stokes_1M | `hybrid-async` | 1.09× | 1.08× | 1.00 |
| vas_stokes_1M | `hybrid-async-dp` | 1.21× | 1.20× | 1.01 |
| vas_stokes_1M | `hybrid-dist-dp` | 1.20× | 1.14× | 1.05 |
| vas_stokes_1M | `hybrid-dist-pipelined` | 0.95× | 0.92× | 1.04 |
| vas_stokes_1M | `mpi-gpu` | 0.79× | 0.77× | 1.04 |
| venkat01 | `hybrid-async` | 1.07× | 1.06× | 1.01 |
| venkat01 | `hybrid-async-dp` | 1.32× | 1.29× | 1.02 |
| venkat01 | `hybrid-dist-dp` | 1.03× | 1.03× | 1.00 |
| venkat01 | `hybrid-dist-pipelined` | 0.87× | 0.86× | 1.01 |
| venkat01 | `mpi-gpu` | 0.81× | 0.76× | 1.07 |
| venkat50 | `hybrid-async` | 1.06× | 1.06× | 1.01 |
| venkat50 | `hybrid-async-dp` | 1.30× | 1.29× | 1.01 |
| venkat50 | `hybrid-dist-dp` | 1.03× | 1.03× | 1.00 |
| venkat50 | `hybrid-dist-pipelined` | 0.87× | 0.86× | 1.01 |
| venkat50 | `mpi-gpu` | 0.81× | 0.76× | 1.07 |
| wang3 | `hybrid-async` | 0.95× | 0.94× | 1.01 |
| wang3 | `hybrid-async-dp` | 1.55× | 1.57× | 0.98 |
| wang3 | `hybrid-dist-dp` | 1.10× | 1.09× | 1.00 |
| wang3 | `hybrid-dist-pipelined` | 0.94× | 0.94× | 1.01 |
| wang3 | `mpi-gpu` | 0.66× | 0.66× | 1.00 |
| webbase-1M | `hybrid-async` | 0.66× | 0.73× | 0.91 |
| webbase-1M | `hybrid-async-dp` | 0.81× | 0.91× | 0.88 |
| webbase-1M | `hybrid-dist-dp` | 1.30× | 1.29× | 1.00 |
| webbase-1M | `hybrid-dist-pipelined` | 0.87× | 0.88× | 0.99 |
| webbase-1M | `mpi-gpu` | 0.49× | 0.49× | 0.99 |
| xenon2 | `hybrid-async` | 1.11× | 1.11× | 1.00 |
| xenon2 | `hybrid-async-dp` | 1.31× | 1.30× | 1.01 |
| xenon2 | `hybrid-dist-dp` | 1.13× | 1.12× | 1.01 |
| xenon2 | `hybrid-dist-pipelined` | 0.92× | 0.93× | 0.99 |
| xenon2 | `mpi-gpu` | 0.82× | 0.81× | 1.00 |

### Mean PaToH/naive ratio per solver

| solver | mean ratio | n matrices |
|---|---|---|
| `hybrid-async` | 0.997 | 49 |
| `hybrid-async-dp` | 0.996 | 49 |
| `hybrid-dist-dp` | 1.005 | 49 |
| `hybrid-dist-pipelined` | 1.012 | 49 |
| `mpi-gpu` | 1.043 | 49 |

## Notes

- Usable matrices: 49. Fully-failed (excluded): ['nlpkkt120'] (solver-level crash).
- `speedup_vs_gpudp` is the honest 'does the heterogeneous CPU split beat just running pure GPU device-pointer?' baseline.
