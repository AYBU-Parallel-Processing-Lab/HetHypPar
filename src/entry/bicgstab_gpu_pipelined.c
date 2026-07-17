#include "stdio.h"
#include <math.h>
#include <omp.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "hhp_common.h"
#include "hhp_matrix.h"
#include "hhp_cuda.h"
#include "hhp_cpu.h"
#include "hhp_util.h"
#include "hhp_prof.h"

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

#include "argp.h"
#include "unistd.h"

// Single-process GPU pipelined BiCGStab -- Algorithm 9 (unpreconditioned) of
// Cools & Vanroose, "The communication-hiding pipelined BiCGstab method", Parallel
// Computing 65 (2017). This is a SINGLE-NODE validation harness: it implements the
// pipelined recurrences exactly so we can check they converge to the same solution
// as standard BiCGStab before porting the MPI_Iallreduce overlap. No comm to hide
// here, so no speedup is expected -- correctness only. Host-pointer cuBLAS for
// debuggability.
//
// Per iteration (rhat = fixed shadow residual = r_0):
//   p = r + b(p - w*s);  s = w + b(s - w*z);  z = t + b(z - w*v)     [b=beta_{i-1}, w=omega_{i-1}]
//   q = r - a*s;  y = w - a*z                                         [a=alpha_i]
//   v = A z
//   omega = (q,y)/(y,y)
//   x += a*p + omega*q;  r = q - omega*y;  w = y - omega*(t - a*v)
//   t = A w
//   beta  = (a/omega) (rhat,r_new)/(rhat,r_old)
//   alpha = (rhat,r_new) / ((rhat,w_new) + beta*(rhat,s) - beta*omega*(rhat,z))
// Vectors here named: rhat=R0. The cuBLAS helpers: axpy(v1,a,out): out = a*v1+out.

struct arguments { char *input_matrix, *input_x, *input_y, *output_x; int n_iters; };

#define OPT_INPUT_MATRIX 'm'
#define OPT_OUTPUT 'o'
#define OPT_INPUT_X 'x'
#define OPT_INPUT_Y 'y'
#define OPT_N_ITERS 'n'

static struct argp_option options[] = {
    {"input-matrix", OPT_INPUT_MATRIX, "FILE", 0, "Input matrix market file"},
    {"input-x", OPT_INPUT_X, "FILE", 0, "Input X (initial guess) vector file"},
    {"input-y", OPT_INPUT_Y, "FILE", 0, "Target B vector file"},
    {"output-x", OPT_OUTPUT, "FILE", 0, "Output X (solution) vector file"},
    {"n-iters", OPT_N_ITERS, "POSITIVE-INTEGER", 0, "Iterations"},
    {0}
};
static error_t parse_opt(int key, char *arg, struct argp_state *state) {
    struct arguments *a = state->input; char *buf = NULL;
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
// prof_sync callback: default-stream solver, so sync the whole device.
static void prof_sync_dev(void *unused) { (void)unused; cudaDeviceSynchronize(); }

static char doc[] = "Single-process GPU pipelined BiCGStab (Alg. 9, unpreconditioned)";
static char args_doc[] = "";
static struct argp argp = {options, parse_opt, args_doc, doc};

int main(int argc, char *argv[]) {
    struct arguments arguments = {}; arguments.n_iters = 1;
    argp_parse(&argp, argc, argv, 0, 0, &arguments);
    if (!arguments.input_matrix || !arguments.input_x || !arguments.input_y || !arguments.output_x) {
        fprintf(stderr, "Error: -m, -x, -y, -o must all be specified.\n"); return EXIT_FAILURE;
    }
    if (access(arguments.input_matrix, F_OK) == -1) ABORT("matrix '%s' missing", arguments.input_matrix)
    if (access(arguments.input_x, F_OK) == -1) ABORT("X '%s' missing", arguments.input_x)
    if (access(arguments.input_y, F_OK) == -1) ABORT("B '%s' missing", arguments.input_y)
    int niters = arguments.n_iters; if (niters < 1) niters = 1;
    int trace = (getenv("HHP_TRACE") != NULL);
    int replace_k = getenv("HHP_REPLACE") ? atoi(getenv("HHP_REPLACE")) : 0; // residual replacement period (0=off)

    double t_begin = omp_get_wtime();
    CHECK_CUDA(cudaSetDevice(0))
    cublasHandle_t bh;   CHECK_CUBLAS(cublasCreate(&bh))
    cusparseHandle_t ch; CHECK_CUSPARSE(cusparseCreate(&ch))

    double t_read0 = omp_get_wtime();
    Device_CSR dA;
    { CSR cA = buReadSparseMatrix(arguments.input_matrix); CHECK_CUSPARSE(device_csr_create(cA, &dA)) freeSparseMatrix(&cA); }
    printf("Matrix name : %s\n", arguments.input_matrix);
    int n = dA.data.m;

    Device_Vector X, B;
    { Vector Xt = vector_read(arguments.input_x, dA.data.n);
      Vector Bt = vector_read(arguments.input_y, dA.data.m);
      CHECK_CUSPARSE(device_vector_init(Xt.nvals, &X)) CHECK_CUSPARSE(device_vector_init(Bt.nvals, &B))
      CHECK_CUSPARSE(device_vector_toGPU(Xt, X)) CHECK_CUSPARSE(device_vector_toGPU(Bt, B))
      vector_destroy(&Xt); vector_destroy(&Bt); }
    double t_read1 = omp_get_wtime();

    // pipelined work vectors
    Device_Vector R, R0, W, T, P, S, Z, Q, Y, V, TMP;
    CHECK_CUSPARSE(device_vector_init(n, &R))  CHECK_CUSPARSE(device_vector_init(n, &R0))
    CHECK_CUSPARSE(device_vector_init(n, &W))  CHECK_CUSPARSE(device_vector_init(n, &T))
    CHECK_CUSPARSE(device_vector_init(n, &P))  CHECK_CUSPARSE(device_vector_init(n, &S))
    CHECK_CUSPARSE(device_vector_init(n, &Z))  CHECK_CUSPARSE(device_vector_init(n, &Q))
    CHECK_CUSPARSE(device_vector_init(n, &Y))  CHECK_CUSPARSE(device_vector_init(n, &V))
    CHECK_CUSPARSE(device_vector_init(n, &TMP))

    Device_Buffer_SpMV buf;
    const double one = 1.0, zero = 0.0, m_one = -1.0;
    CHECK_CUSPARSE(device_buffer_spmv_create(ch, dA.desc, X, B, &one, &zero, &buf))

    // --- init:  r = B - A x ;  R0 = r ;  w = A r ;  t = A w ;  p=s=z=v=0 ---
    CHECK_CUSPARSE(device_csr_spmv(ch, dA, X, TMP, one, zero, buf))   // TMP = A x
    CHECK_CUDA(device_vector_GPUtoGPU(B, R))                          // R = B
    CHECK_CUBLAS(device_vector_axpy(bh, TMP, m_one, R))               // R = B - A x
    CHECK_CUDA(device_vector_GPUtoGPU(R, R0))                         // R0 = R
    CHECK_CUSPARSE(device_csr_spmv(ch, dA, R, W, one, zero, buf))     // W = A r
    CHECK_CUSPARSE(device_csr_spmv(ch, dA, W, T, one, zero, buf))     // T = A w
    CHECK_CUSPARSE(device_vector_zero(P)) CHECK_CUSPARSE(device_vector_zero(S))
    CHECK_CUSPARSE(device_vector_zero(Z)) CHECK_CUSPARSE(device_vector_zero(V))

    double rho_old, r0w, bnorm;
    CHECK_CUBLAS(device_vector_dot(bh, R0, R, &rho_old))   // (r0, r)
    CHECK_CUBLAS(device_vector_dot(bh, R0, W, &r0w))       // (r0, w)
    CHECK_CUBLAS(device_vector_dot(bh, B, B, &bnorm)) bnorm = sqrt(bnorm);
    double alpha = rho_old / r0w;
    double beta_prev = 0.0, omega_prev = 1.0, omega = 0.0;

    // Optional detailed profiling (env HHP_PROF: 1=stderr summary, 2=+per-iter TSV).
    Prof g_P; prof_init(&g_P, niters, prof_sync_dev, NULL);
    prof_set_preprocess(&g_P, t_read1 - t_begin);

    double t_loop0 = omp_get_wtime();
    for (int i = 0; i < niters; i++) {
        prof_tick(&g_P);
        double bw = beta_prev * omega_prev;
        // p = r + beta(p - omega s) = beta*p + r - beta*omega*s
        CHECK_CUBLAS(device_vector_scale(bh, beta_prev, P))
        CHECK_CUBLAS(device_vector_axpy(bh, R, one, P))
        CHECK_CUBLAS(device_vector_axpy(bh, S, -bw, P))
        // s = beta*s + w - beta*omega*z
        CHECK_CUBLAS(device_vector_scale(bh, beta_prev, S))
        CHECK_CUBLAS(device_vector_axpy(bh, W, one, S))
        CHECK_CUBLAS(device_vector_axpy(bh, Z, -bw, S))
        // z = beta*z + t - beta*omega*v
        CHECK_CUBLAS(device_vector_scale(bh, beta_prev, Z))
        CHECK_CUBLAS(device_vector_axpy(bh, T, one, Z))
        CHECK_CUBLAS(device_vector_axpy(bh, V, -bw, Z))
        // q = r - alpha s ;  y = w - alpha z
        CHECK_CUDA(device_vector_GPUtoGPU(R, Q)) CHECK_CUBLAS(device_vector_axpy(bh, S, -alpha, Q))
        CHECK_CUDA(device_vector_GPUtoGPU(W, Y)) CHECK_CUBLAS(device_vector_axpy(bh, Z, -alpha, Y))
        prof_lap(&g_P, PF_VECOPS);
        // v = A z
        CHECK_CUSPARSE(device_csr_spmv(ch, dA, Z, V, one, zero, buf))
        prof_lap(&g_P, PF_SPMV);
        // omega = (q,y)/(y,y)
        double qy, yy; CHECK_CUBLAS(device_vector_dot(bh, Q, Y, &qy)) CHECK_CUBLAS(device_vector_dot(bh, Y, Y, &yy))
        omega = qy / yy;
        prof_lap(&g_P, PF_DOT);
        // x += alpha p + omega q
        CHECK_CUBLAS(device_vector_axpy(bh, P, alpha, X))
        CHECK_CUBLAS(device_vector_axpy(bh, Q, omega, X))
        // r = q - omega y
        CHECK_CUDA(device_vector_GPUtoGPU(Q, R)) CHECK_CUBLAS(device_vector_axpy(bh, Y, -omega, R))
        // w = y - omega(t - alpha v) = y - omega*t + omega*alpha*v
        CHECK_CUDA(device_vector_GPUtoGPU(Y, W))
        CHECK_CUBLAS(device_vector_axpy(bh, T, -omega, W))
        CHECK_CUBLAS(device_vector_axpy(bh, V, omega * alpha, W))
        prof_lap(&g_P, PF_VECOPS);
        // t = A w
        CHECK_CUSPARSE(device_csr_spmv(ch, dA, W, T, one, zero, buf))
        // --- residual replacement (Sec 4.2): every k iters, reset the recurrence-
        // maintained vectors to their true definitions (using the primary iterates
        // x and p) to negate accumulated rounding error.  Costs 6 SpMVs.
        if (replace_k > 0 && (i + 1) % replace_k == 0) {
            CHECK_CUSPARSE(device_csr_spmv(ch, dA, X, TMP, one, zero, buf))   // TMP = A x
            CHECK_CUDA(device_vector_GPUtoGPU(B, R))
            CHECK_CUBLAS(device_vector_axpy(bh, TMP, m_one, R))              // r := b - A x
            CHECK_CUSPARSE(device_csr_spmv(ch, dA, R, W, one, zero, buf))     // w := A r
            CHECK_CUSPARSE(device_csr_spmv(ch, dA, W, T, one, zero, buf))     // t := A w
            CHECK_CUSPARSE(device_csr_spmv(ch, dA, P, S, one, zero, buf))     // s := A p
            CHECK_CUSPARSE(device_csr_spmv(ch, dA, S, Z, one, zero, buf))     // z := A s
            CHECK_CUSPARSE(device_csr_spmv(ch, dA, Z, V, one, zero, buf))     // v := A z
        }
        prof_lap(&g_P, PF_SPMV);
        // reduction: (r0,r), (r0,w), (r0,s), (r0,z)
        double rho_new, r0w2, r0s, r0z;
        CHECK_CUBLAS(device_vector_dot(bh, R0, R, &rho_new))
        CHECK_CUBLAS(device_vector_dot(bh, R0, W, &r0w2))
        CHECK_CUBLAS(device_vector_dot(bh, R0, S, &r0s))
        CHECK_CUBLAS(device_vector_dot(bh, R0, Z, &r0z))
        prof_lap(&g_P, PF_DOT);
        double beta = (alpha / omega) * (rho_new / rho_old);
        double denom = r0w2 + beta * r0s - beta * omega * r0z;
        double alpha_next = rho_new / denom;

        if (trace && (i < 5 || i % 50 == 0)) {
            double rr; CHECK_CUBLAS(device_vector_dot(bh, R, R, &rr))
            printf("  it=%d  recursive ||r||/||b|| = %.6E  (alpha=%.3e beta=%.3e omega=%.3e)\n",
                   i, sqrt(rr) / bnorm, alpha_next, beta, omega);
        }
        rho_old = rho_new; beta_prev = beta; omega_prev = omega; alpha = alpha_next;
        prof_iter_end(&g_P);
    }
    cudaDeviceSynchronize();
    double t_loop1 = omp_get_wtime();
    prof_report(&g_P, "gpu-pipelined", arguments.input_matrix);
    prof_free(&g_P);

    // --- final TRUE relative residual ||A x - B|| / ||B|| ---
    CHECK_CUSPARSE(device_csr_spmv(ch, dA, X, TMP, one, zero, buf))   // TMP = A x
    CHECK_CUBLAS(device_vector_axpy(bh, B, m_one, TMP))               // TMP = A x - B
    double sy, sb; CHECK_CUBLAS(device_vector_dot(bh, TMP, TMP, &sy)) CHECK_CUBLAS(device_vector_dot(bh, B, B, &sb))
    double relative_residual = sqrt(sy / sb);

    printf("n_iters : %d \n", niters);
    printf("spmv : %lf \n", t_loop1 - t_loop0);
    printf("file_read : %lf \n", t_read1 - t_read0);
    printf("relative_residual : %E\n", relative_residual);
    printf("everything_total : %lf\n", omp_get_wtime() - t_begin);
    printf("\n----------------------------------------------------------------------\n");

    Vector Xh = vector_init(X.nvals);
    CHECK_CUSPARSE(device_vector_toCPU(X, Xh)) vector_write(arguments.output_x, Xh); vector_destroy(&Xh);
    return EXIT_SUCCESS;
}
