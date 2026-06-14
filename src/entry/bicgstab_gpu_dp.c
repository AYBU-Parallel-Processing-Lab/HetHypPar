#include "stdio.h"
#include <omp.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>

#include "hhp_common.h"
#include "hhp_matrix.h"
#include "hhp_cuda.h"
#include "hhp_cpu.h"
#include "hhp_util.h"
#include "hhp_dp_kernels.h"

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

#include "argp.h"
#include "unistd.h"

// Pure-GPU BiCGStab using cuBLAS DEVICE-pointer mode for the dot products.
//
// Baseline (bicgstab_gpu.c) runs cublasDdot in host-pointer mode: every dot
// blocks the host on a device->host scalar copy, i.e. 5 host<->device stalls
// per iteration. Profiling (docs/dot-product-profiling.md) showed those stalls
// make dots 23-41% of the loop.
//
// Here the dot results and the axpy/scal scale factors all live on the device.
// The per-iteration scalar updates (beta, alpha, omega) run in tiny one-thread
// kernels (hhp_dp_kernels.cu), so the entire iteration is one uninterrupted
// stream of GPU kernels with NO host sync. The host only synchronizes after the
// whole loop. Math is identical to the host-pointer reference.

struct arguments {
    char *input_matrix, *input_x, *input_y, *output_x;
    int n_iters;
};

#define OPT_INPUT_MATRIX 'm'
#define OPT_OUTPUT 'o'
#define OPT_INPUT_X 'x'
#define OPT_INPUT_Y 'y'
#define OPT_N_ITERS 'n'

static struct argp_option options[] = {
    {"input-matrix", OPT_INPUT_MATRIX, "FILE", 0, "Input matrix market file path"},
    {"input-x", OPT_INPUT_X, "FILE", 0, "Input X vector file path"},
    {"input-y", OPT_INPUT_Y, "FILE", 0, "Input Y target vector file path"},
    {"output-x", OPT_OUTPUT, "FILE", 0, "Output X vector file path"},
    {"n-iters", OPT_N_ITERS, "POSITIVE-INTEGER", 0, "Number of iterations"},
    {0}
};

static error_t parse_opt(int key, char *arg, struct argp_state *state) {
    struct arguments *a = state->input;
    char *buf = NULL;
    switch (key) {
        case OPT_INPUT_MATRIX: a->input_matrix = arg; break;
        case OPT_INPUT_X:      a->input_x = arg; break;
        case OPT_INPUT_Y:      a->input_y = arg; break;
        case OPT_OUTPUT:       a->output_x = arg; break;
        case OPT_N_ITERS:      a->n_iters = strtol(arg, &buf, 10); break;
        case ARGP_KEY_ARG:     return 0;
        default:               return ARGP_ERR_UNKNOWN;
    }
    return 0;
}

static char doc[] = "Pure-GPU BiCGStab with cuBLAS device-pointer-mode dots";
static char args_doc[] = "";
static struct argp argp = {options, parse_opt, args_doc, doc};

// Allocate a single device double and set it from a host value.
static double *dscalar(double init) {
    double *p;
    if (cudaMalloc((void **)&p, sizeof(double)) != cudaSuccess)
        ABORT("cudaMalloc failed for device scalar")
    if (cudaMemcpy(p, &init, sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess)
        ABORT("cudaMemcpy failed for device scalar")
    return p;
}

int main(int argc, char *argv[]) {
    struct arguments arguments = {};
    arguments.n_iters = 1;
    argp_parse(&argp, argc, argv, 0, 0, &arguments);

    if (!arguments.input_matrix || !arguments.input_x || !arguments.input_y
        || !arguments.output_x) {
        fprintf(stderr, "Error: -m, -x, -y, -o must all be specified.\n");
        return EXIT_FAILURE;
    }
    if (access(arguments.input_matrix, F_OK) == -1)
        ABORT("Input matrix '%s' does not exist", arguments.input_matrix)
    if (access(arguments.input_x, F_OK) == -1)
        ABORT("Input X '%s' does not exist", arguments.input_x)
    if (access(arguments.input_y, F_OK) == -1)
        ABORT("Input B '%s' does not exist", arguments.input_y)

    int niters = arguments.n_iters;
    if (niters < 1) niters = 1;

    double t_begin = omp_get_wtime();

    CHECK_CUDA(cudaSetDevice(0))
    cublasHandle_t bh;   CHECK_CUBLAS(cublasCreate(&bh))
    cusparseHandle_t ch; CHECK_CUSPARSE(cusparseCreate(&ch))

    double t_read0 = omp_get_wtime();
    Device_CSR dA;
    {
        CSR cA = buReadSparseMatrix(arguments.input_matrix);
        CHECK_CUSPARSE(device_csr_create(cA, &dA))
        freeSparseMatrix(&cA);
    }
    printf("Matrix name : %s\n", arguments.input_matrix);

    Device_Vector X, B;
    {
        Vector Xt = vector_read(arguments.input_x, dA.data.n);
        Vector Bt = vector_read(arguments.input_y, dA.data.m);
        CHECK_CUSPARSE(device_vector_init(Xt.nvals, &X))
        CHECK_CUSPARSE(device_vector_init(Bt.nvals, &B))
        CHECK_CUSPARSE(device_vector_toGPU(Xt, X))
        CHECK_CUSPARSE(device_vector_toGPU(Bt, B))
        vector_destroy(&Xt);
        vector_destroy(&Bt);
    }
    double t_read1 = omp_get_wtime();

    int n = dA.data.m;
    Device_Vector Y, V, P, R, R_0, S, T;
    CHECK_CUSPARSE(device_vector_init(n, &Y))   CHECK_CUSPARSE(device_vector_zero(Y))
    CHECK_CUSPARSE(device_vector_init(n, &V))   CHECK_CUSPARSE(device_vector_zero(V))
    CHECK_CUSPARSE(device_vector_init(n, &P))   CHECK_CUSPARSE(device_vector_zero(P))
    CHECK_CUSPARSE(device_vector_init(n, &R))
    CHECK_CUSPARSE(device_vector_init(n, &R_0))
    CHECK_CUSPARSE(device_vector_init(n, &S))
    CHECK_CUSPARSE(device_vector_init(n, &T))

    const double h_one = 1.0, h_neg = -1.0, h_zero = 0.0;

    Device_Buffer_SpMV dA_buf;
    CHECK_CUSPARSE(device_buffer_spmv_create(ch, dA.desc, X, B, &h_one, &h_zero, &dA_buf))

    // R = B - A*X   (host-pointer mode is fine here, outside the loop)
    CHECK_CUDA(device_vector_GPUtoGPU(B, R))
    CHECK_CUSPARSE(device_csr_spmv(ch, dA, X, Y, h_one, h_zero, dA_buf))
    CHECK_CUBLAS(device_vector_axpy(bh, Y, h_neg, R))     // R -= A*X
    CHECK_CUSPARSE(device_vector_zero(Y))
    CHECK_CUDA(device_vector_GPUtoGPU(R, R_0))            // R_0 = R

    // --- device-resident scalars ---
    double *d_rho     = dscalar(1.0);
    double *d_alpha   = dscalar(1.0);
    double *d_omega   = dscalar(1.0);
    double *d_beta    = dscalar(0.0);
    double *d_neg_a   = dscalar(0.0);   // -alpha
    double *d_neg_w   = dscalar(0.0);   // -omega
    double *d_one     = dscalar(1.0);
    double *d_neg_one = dscalar(-1.0);
    double *d_rho_new = dscalar(0.0);
    double *d_rv      = dscalar(0.0);
    double *d_ts      = dscalar(0.0);
    double *d_tt      = dscalar(0.0);
    double *d_tol     = dscalar(0.0);

    cudaStream_t s = 0;  // default stream (cuBLAS/cuSPARSE default to it)

    double t_loop0 = omp_get_wtime();
    CHECK_CUBLAS(cublasSetPointerMode(bh, CUBLAS_POINTER_MODE_DEVICE))

    for (int it = 0; it < niters; it++) {
        // rho_new = R . R_0 ;  beta = (rho_new/rho)*(alpha/omega) ; rho = rho_new
        CHECK_CUBLAS(cublasDdot(bh, n, R.vals, 1, R_0.vals, 1, d_rho_new))
        hhp_dp_update_beta(d_beta, d_rho, d_rho_new, d_alpha, d_omega, s);

        // P = (P - omega*V)*beta + R   (uses OLD omega) -- one fused kernel
        hhp_dp_vecop_P(P.vals, V.vals, R.vals, d_omega, d_beta, n, s);

        // V = A*P
        CHECK_CUSPARSE(device_csr_spmv(ch, dA, P, V, h_one, h_zero, dA_buf))

        // rv = R_0 . V ;  alpha = rho/rv ;  neg_a = -alpha
        CHECK_CUBLAS(cublasDdot(bh, n, R_0.vals, 1, V.vals, 1, d_rv))
        hhp_dp_update_alpha(d_alpha, d_neg_a, d_rho, d_rv, s);

        // S = R - alpha*V   -- one fused kernel
        hhp_dp_vecop_S(S.vals, R.vals, V.vals, d_neg_a, n, s);

        // T = A*S
        CHECK_CUSPARSE(device_csr_spmv(ch, dA, S, T, h_one, h_zero, dA_buf))

        // ts = T.S ; tt = T.T ; omega = ts/tt ; neg_w = -omega
        CHECK_CUBLAS(cublasDdot(bh, n, T.vals, 1, S.vals, 1, d_ts))
        CHECK_CUBLAS(cublasDdot(bh, n, T.vals, 1, T.vals, 1, d_tt))
        hhp_dp_update_omega(d_omega, d_neg_w, d_ts, d_tt, s);

        // X += alpha*P + omega*S ; R = S - omega*T   -- one fused kernel
        hhp_dp_vecop_XR(X.vals, R.vals, P.vals, S.vals, T.vals, d_alpha, d_omega, d_neg_w, n, s);

        // tol = S.S  (computed for parity with the reference; result unused)
        CHECK_CUBLAS(cublasDdot(bh, n, S.vals, 1, S.vals, 1, d_tol))
    }

    CHECK_CUBLAS(cublasSetPointerMode(bh, CUBLAS_POINTER_MODE_HOST))
    cudaDeviceSynchronize();
    double t_loop1 = omp_get_wtime();

    // --- relative residual: ||A*X - B|| / ||B|| ---
    CHECK_CUSPARSE(device_csr_spmv(ch, dA, X, Y, h_one, h_zero, dA_buf))
    CHECK_CUBLAS(device_vector_axpy(bh, B, h_neg, Y))   // Y = A*X - B
    double sy, sb;
    CHECK_CUBLAS(device_vector_dot(bh, Y, Y, &sy))
    CHECK_CUBLAS(device_vector_dot(bh, B, B, &sb))
    double relative_residual = sqrt(sy / sb);

    printf("n_iters : %d \n", niters);
    printf("spmv : %lf \n", t_loop1 - t_loop0);
    printf("file_read : %lf \n", t_read1 - t_read0);
    printf("relative_residual : %E\n", relative_residual);
    printf("everything_total : %lf\n", omp_get_wtime() - t_begin);
    printf("\n----------------------------------------------------------------------\n");

    Vector Xh = vector_init(X.nvals);
    CHECK_CUSPARSE(device_vector_toCPU(X, Xh))
    vector_write(arguments.output_x, Xh);
    vector_destroy(&Xh);

    return EXIT_SUCCESS;
}
