// Tiny single-thread device kernels for device-pointer-mode BiCGStab.
//
// In device-pointer mode (cublasSetPointerMode(.., CUBLAS_POINTER_MODE_DEVICE))
// dot results and axpy/scal scale factors all live in device memory, so the
// per-iteration scalar updates (beta, alpha, omega) must run on the device
// instead of the host. Keeping them on-device removes the host<->device sync
// that host-pointer-mode cublasDdot forces after every dot -- the whole point
// of the optimization. Each update is a single scalar op, so one thread is fine.

#include <cuda_runtime.h>
#include "hhp_dp_kernels.h"

// beta = (rho_new / rho_old) * (alpha / omega);  then commit rho_old <- rho_new.
// Uses the OLD alpha/omega (not yet updated this iteration), matching the
// host-pointer reference loop.
__global__ static void k_update_beta(double *beta, double *rho_old,
                                     const double *rho_new,
                                     const double *alpha, const double *omega) {
    *beta = (*rho_new / *rho_old) * (*alpha / *omega);
    *rho_old = *rho_new;
}

// alpha = rho / rv;  neg_alpha = -alpha.
__global__ static void k_update_alpha(double *alpha, double *neg_alpha,
                                      const double *rho, const double *rv) {
    double a = *rho / *rv;
    *alpha = a;
    *neg_alpha = -a;
}

// omega = ts / tt;  neg_omega = -omega.
__global__ static void k_update_omega(double *omega, double *neg_omega,
                                      const double *ts, const double *tt) {
    double w = *ts / *tt;
    *omega = w;
    *neg_omega = -w;
}

// total = a + b. Combines a GPU partial dot with a CPU partial dot (already
// copied to device) for the distributed solver.
__global__ static void k_add(double *total, const double *a, const double *b) {
    *total = *a + *b;
}

extern "C" {

void hhp_dp_add(double *total, const double *a, const double *b, cudaStream_t s) {
    k_add<<<1, 1, 0, s>>>(total, a, b);
}

void hhp_dp_update_beta(double *beta, double *rho_old, const double *rho_new,
                        const double *alpha, const double *omega, cudaStream_t s) {
    k_update_beta<<<1, 1, 0, s>>>(beta, rho_old, rho_new, alpha, omega);
}

void hhp_dp_update_alpha(double *alpha, double *neg_alpha,
                         const double *rho, const double *rv, cudaStream_t s) {
    k_update_alpha<<<1, 1, 0, s>>>(alpha, neg_alpha, rho, rv);
}

void hhp_dp_update_omega(double *omega, double *neg_omega,
                         const double *ts, const double *tt, cudaStream_t s) {
    k_update_omega<<<1, 1, 0, s>>>(omega, neg_omega, ts, tt);
}

} // extern "C"
