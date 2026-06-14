#ifndef HHP_DP_KERNELS_H
#define HHP_DP_KERNELS_H

// Device-side scalar updates for device-pointer-mode BiCGStab. All pointer
// arguments are DEVICE pointers; `s` is the stream the updates are queued on.
// Each launches a single-thread kernel (the work is one scalar op).

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// total = a + b  (combine a GPU partial dot with a device-resident CPU partial).
void hhp_dp_add(double *total, const double *a, const double *b, cudaStream_t s);

// Set mapped-host *flag = seq after prior stream work, for a host spin-wait.
void hhp_dp_set_flag(unsigned long long *flag, unsigned long long seq, cudaStream_t s);

// out[k] = in[idx[k]] for k in [0,nh) -- gather a halo before a D->H transfer.
void hhp_dp_gather(double *out, const double *in, const int *idx, int nh, cudaStream_t s);

// Fused combine+update (distributed): sum GPU partial g + CPU partial c, update
// scalar, and publish result(s) to mapped host `hm` + set `flag`=seq (for a host
// spin-wait that replaces cudaStreamSynchronize). hm/flag are device pointers to
// mapped host memory.
void hhp_dp_cu_beta(double *beta, double *rho_old, const double *g, const double *c,
                    const double *alpha, const double *omega,
                    double *hm, unsigned long long *flag, unsigned long long seq, cudaStream_t s);
void hhp_dp_cu_alpha(double *alpha, double *neg_a, const double *rho,
                     const double *g, const double *c,
                     double *hm, unsigned long long *flag, unsigned long long seq, cudaStream_t s);
void hhp_dp_cu_omega(double *omega, double *neg_w, const double *gts, const double *cts,
                     const double *gtt, const double *ctt,
                     double *hm, unsigned long long *flag, unsigned long long seq, cudaStream_t s);

// Fused element-wise vector ops (device-scalar coefficients), n = slice length.
// P = (P - omega*V)*beta + R
void hhp_dp_vecop_P(double *P, const double *V, const double *R,
                    const double *omega, const double *beta, int n, cudaStream_t s);
// S = R + neg_alpha*V
void hhp_dp_vecop_S(double *S, const double *R, const double *V,
                    const double *neg_alpha, int n, cudaStream_t s);
// X += alpha*P + omega*S ; R = S + neg_omega*T
void hhp_dp_vecop_XR(double *X, double *R, const double *P, const double *S,
                     const double *T, const double *alpha, const double *omega,
                     const double *neg_omega, int n, cudaStream_t s);

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
