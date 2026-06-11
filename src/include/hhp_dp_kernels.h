#ifndef HHP_DP_KERNELS_H
#define HHP_DP_KERNELS_H

// Device-side scalar updates for device-pointer-mode BiCGStab. All pointer
// arguments are DEVICE pointers; `s` is the stream the updates are queued on.
// Each launches a single-thread kernel (the work is one scalar op).

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// beta = (rho_new / rho_old) * (alpha / omega);  rho_old <- rho_new.
void hhp_dp_update_beta(double *beta, double *rho_old, const double *rho_new,
                        const double *alpha, const double *omega, cudaStream_t s);

// alpha = rho / rv;  neg_alpha = -alpha.
void hhp_dp_update_alpha(double *alpha, double *neg_alpha,
                         const double *rho, const double *rv, cudaStream_t s);

// omega = ts / tt;  neg_omega = -omega.
void hhp_dp_update_omega(double *omega, double *neg_omega,
                         const double *ts, const double *tt, cudaStream_t s);

#ifdef __cplusplus
}
#endif

#endif // HHP_DP_KERNELS_H
