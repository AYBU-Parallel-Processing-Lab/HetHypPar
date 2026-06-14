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

// --- Fused element-wise vector-op kernels (device-scalar coefficients) ---
// One kernel launch / one memory pass replaces a sequence of cuBLAS scal/axpy
// calls. Coefficients are read from device memory (dp mode). n is the slice
// length (per-rank), so these are local ops -- multi-node friendly.

// P = (P - omega*V)*beta + R
__global__ static void k_vecop_P(double *P, const double *V, const double *R,
                                 const double *omega, const double *beta, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) P[i] = (P[i] - (*omega) * V[i]) * (*beta) + R[i];
}

// S = R + neg_alpha*V   (i.e. R - alpha*V)
__global__ static void k_vecop_S(double *S, const double *R, const double *V,
                                 const double *neg_alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) S[i] = R[i] + (*neg_alpha) * V[i];
}

// X += alpha*P + omega*S ;  R = S + neg_omega*T
__global__ static void k_vecop_XR(double *X, double *R, const double *P,
                                  const double *S, const double *T,
                                  const double *alpha, const double *omega,
                                  const double *neg_omega, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        X[i] += (*alpha) * P[i] + (*omega) * S[i];
        R[i]  = S[i] + (*neg_omega) * T[i];
    }
}

// Fused combine+update: take GPU and CPU partial dots directly, sum them, apply
// the scalar update, AND publish the result(s) to mapped host memory + bump a
// flag, so the host can spin-wait (~1-2us) instead of cudaStreamSynchronize
// (~10-15us). hm = mapped-host scalar slots (device ptr), flag = mapped-host
// flag (device ptr); the host polls flag>=seq then reads hm. __threadfence_system
// orders the scalar writes before the flag so the host sees consistent data.
__global__ static void k_cu_beta(double *beta, double *rho_old, const double *g,
                                 const double *c, const double *alpha, const double *omega,
                                 double *hm, unsigned long long *flag, unsigned long long seq) {
    double rn = *g + *c;
    double b = (rn / *rho_old) * (*alpha / *omega);
    *beta = b; *rho_old = rn;
    hm[0] = b;
    __threadfence_system();
    *flag = seq;
}
__global__ static void k_cu_alpha(double *alpha, double *neg_a, const double *rho,
                                  const double *g, const double *c,
                                  double *hm, unsigned long long *flag, unsigned long long seq) {
    double a = *rho / (*g + *c);
    *alpha = a; *neg_a = -a;
    hm[0] = a; hm[1] = -a;
    __threadfence_system();
    *flag = seq;
}
__global__ static void k_cu_omega(double *omega, double *neg_w,
                                  const double *gts, const double *cts,
                                  const double *gtt, const double *ctt,
                                  double *hm, unsigned long long *flag, unsigned long long seq) {
    double w = (*gts + *cts) / (*gtt + *ctt);
    *omega = w; *neg_w = -w;
    hm[0] = w; hm[1] = -w;
    __threadfence_system();
    *flag = seq;
}

// Set a mapped-host flag after prior stream work (e.g. a D->H copy) so the host
// can spin-wait on completion instead of cudaStreamSynchronize.
__global__ static void k_set_flag(unsigned long long *flag, unsigned long long seq) {
    __threadfence_system();
    *flag = seq;
}

extern "C" {

void hhp_dp_add(double *total, const double *a, const double *b, cudaStream_t s) {
    k_add<<<1, 1, 0, s>>>(total, a, b);
}

void hhp_dp_set_flag(unsigned long long *flag, unsigned long long seq, cudaStream_t s) {
    k_set_flag<<<1, 1, 0, s>>>(flag, seq);
}

void hhp_dp_cu_beta(double *beta, double *rho_old, const double *g, const double *c,
                    const double *alpha, const double *omega,
                    double *hm, unsigned long long *flag, unsigned long long seq, cudaStream_t s) {
    k_cu_beta<<<1, 1, 0, s>>>(beta, rho_old, g, c, alpha, omega, hm, flag, seq);
}
void hhp_dp_cu_alpha(double *alpha, double *neg_a, const double *rho,
                     const double *g, const double *c,
                     double *hm, unsigned long long *flag, unsigned long long seq, cudaStream_t s) {
    k_cu_alpha<<<1, 1, 0, s>>>(alpha, neg_a, rho, g, c, hm, flag, seq);
}
void hhp_dp_cu_omega(double *omega, double *neg_w, const double *gts, const double *cts,
                     const double *gtt, const double *ctt,
                     double *hm, unsigned long long *flag, unsigned long long seq, cudaStream_t s) {
    k_cu_omega<<<1, 1, 0, s>>>(omega, neg_w, gts, cts, gtt, ctt, hm, flag, seq);
}

void hhp_dp_vecop_P(double *P, const double *V, const double *R,
                    const double *omega, const double *beta, int n, cudaStream_t s) {
    if (n <= 0) return;
    k_vecop_P<<<(n + 255) / 256, 256, 0, s>>>(P, V, R, omega, beta, n);
}

void hhp_dp_vecop_S(double *S, const double *R, const double *V,
                    const double *neg_alpha, int n, cudaStream_t s) {
    if (n <= 0) return;
    k_vecop_S<<<(n + 255) / 256, 256, 0, s>>>(S, R, V, neg_alpha, n);
}

void hhp_dp_vecop_XR(double *X, double *R, const double *P, const double *S,
                     const double *T, const double *alpha, const double *omega,
                     const double *neg_omega, int n, cudaStream_t s) {
    if (n <= 0) return;
    k_vecop_XR<<<(n + 255) / 256, 256, 0, s>>>(X, R, P, S, T, alpha, omega, neg_omega, n);
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
