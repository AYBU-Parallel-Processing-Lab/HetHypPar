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
#include "hhp_dp_kernels.h"
#include "hhp_dp_helpers.h"

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

#include "argp.h"
#include "unistd.h"

// Single-process GPU+CPU hybrid BiCGStab -- NO MPI -- with cuBLAS DEVICE-pointer
// mode for the dot products.
//
// This is bicgstab_hybrid_async.c plus the device-pointer-dots optimization
// already proven on the pure-GPU solver (bicgstab_gpu_dp.c, +1.10..1.21x). The
// hybrid still pays 2 unavoidable host syncs/iter inside hybrid_spmv (the CPU
// must read each SpMV input), which device-pointer mode cannot remove. What it
// DOES remove is the 5 host<->device scalar round-trips the dots forced in
// host-pointer mode: results stay on the device and the beta/alpha/omega
// updates run in tiny kernels (hhp_dp_kernels.cu). Math is identical to the
// host-pointer hybrid. The matrix layout, slicing, and hybrid_spmv overlap are
// unchanged from bicgstab_hybrid_async.c.

struct arguments {
    char *input_matrix, *input_x, *input_y, *input_part, *input_gpu, *output_x;
    int n_iters;
};

#define OPT_INPUT_MATRIX 'm'
#define OPT_OUTPUT 'o'
#define OPT_INPUT_X 'x'
#define OPT_INPUT_Y 'y'
#define OPT_INPUT_PART 'p'
#define OPT_INPUT_GPU 'g'
#define OPT_N_ITERS 'n'

static struct argp_option options[] = {
    {"input-matrix", OPT_INPUT_MATRIX, "FILE", 0, "Input matrix market file"},
    {"input-x", OPT_INPUT_X, "FILE", 0, "Input X (initial guess) vector file"},
    {"input-y", OPT_INPUT_Y, "FILE", 0, "Target B vector file"},
    {"input-part", OPT_INPUT_PART, "FILE", 0, "2-rank partition vector file"},
    {"is-gpu", OPT_INPUT_GPU, "FILE", 0, "is_gpu file (2 lines, exactly one '1')"},
    {"output-x", OPT_OUTPUT, "FILE", 0, "Output X (solution) vector file"},
    {"n-iters", OPT_N_ITERS, "POSITIVE-INTEGER", 0, "BiCGStab iterations"},
    {0}
};

static error_t parse_opt(int key, char *arg, struct argp_state *state) {
    struct arguments *a = state->input;
    char *buf = NULL;
    switch (key) {
        case OPT_INPUT_MATRIX: a->input_matrix = arg; break;
        case OPT_INPUT_X:      a->input_x = arg; break;
        case OPT_INPUT_Y:      a->input_y = arg; break;
        case OPT_INPUT_PART:   a->input_part = arg; break;
        case OPT_INPUT_GPU:    a->input_gpu = arg; break;
        case OPT_OUTPUT:       a->output_x = arg; break;
        case OPT_N_ITERS:      a->n_iters = strtol(arg, &buf, 10); break;
        case ARGP_KEY_ARG:     return 0;
        default:               return ARGP_ERR_UNKNOWN;
    }
    return 0;
}

static char doc[] = "Hybrid GPU+CPU BiCGStab (no MPI) with device-pointer-mode dots";
static char args_doc[] = "";
static struct argp argp = {options, parse_opt, args_doc, doc};

// read_ints, dscalar, csr_permute, csr_row_slice live in hhp_dp_helpers.h.

// One overlapped hybrid SpMV. Identical to bicgstab_hybrid_async.c: cuSPARSE
// runs with host alpha/beta constants, independent of the cuBLAS pointer mode.
static cusparseStatus_t hybrid_spmv(
    cusparseHandle_t   ch,
    cusparseSpMatDescr_t mat_desc,
    cusparseDnVecDescr_t in_desc,
    cusparseDnVecDescr_t out_gpu_desc,
    CSR                cpuA,
    Device_Vector      in_dev,
    double            *in_pin,
    Device_Vector      out_dev,
    double            *cpu_out,
    int                n_gpu,
    int                n_cpu,
    int                n_full,
    cudaStream_t       compute_s,
    cudaStream_t       copy_s,
    cudaEvent_t        in_ready_event,
    cudaEvent_t        scatter_done_event,
    Device_Buffer_SpMV buf)
{
    const double alpha = 1.0, beta = 0.0;

    CHECK_CUDA(cudaEventRecord(in_ready_event, compute_s))
    CHECK_CUDA(cudaStreamWaitEvent(copy_s, in_ready_event, 0))

    CHECK_CUDA(cudaMemcpyAsync(in_pin, in_dev.vals, n_full * sizeof(double),
                               cudaMemcpyDeviceToHost, copy_s))

    CHECK_CUSPARSE(cusparseSpMV(ch, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, mat_desc, in_desc, &beta, out_gpu_desc,
                                CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buf))

    CHECK_CUDA(cudaStreamSynchronize(copy_s))

    if (n_cpu > 0) {
        Vector y_view = {.vals = cpu_out, .nvals = n_cpu};
        Vector x_view = {.vals = in_pin,  .nvals = n_full};
        CSR_spmxv_omp(cpuA, x_view, y_view);
    }

    if (n_cpu > 0) {
        CHECK_CUDA(cudaMemcpyAsync(out_dev.vals + n_gpu, cpu_out,
                                   n_cpu * sizeof(double),
                                   cudaMemcpyHostToDevice, copy_s))
        CHECK_CUDA(cudaEventRecord(scatter_done_event, copy_s))
        CHECK_CUDA(cudaStreamWaitEvent(compute_s, scatter_done_event, 0))
    }

    return CUSPARSE_STATUS_SUCCESS;
}

int main(int argc, char *argv[]) {
    struct arguments arguments = {};
    arguments.n_iters = 1;
    argp_parse(&argp, argc, argv, 0, 0, &arguments);

    if (!arguments.input_matrix || !arguments.input_x || !arguments.input_y
        || !arguments.input_part || !arguments.input_gpu || !arguments.output_x) {
        fprintf(stderr, "Error: -m, -x, -y, -p, -g, -o must all be specified.\n");
        return EXIT_FAILURE;
    }
    if (access(arguments.input_matrix, F_OK) == -1)
        ABORT("Input matrix '%s' does not exist", arguments.input_matrix)
    if (access(arguments.input_x, F_OK) == -1)
        ABORT("Input X '%s' does not exist", arguments.input_x)
    if (access(arguments.input_y, F_OK) == -1)
        ABORT("Input B '%s' does not exist", arguments.input_y)
    if (access(arguments.input_part, F_OK) == -1)
        ABORT("Partition '%s' does not exist", arguments.input_part)
    if (access(arguments.input_gpu, F_OK) == -1)
        ABORT("is_gpu '%s' does not exist", arguments.input_gpu)

    int niters = arguments.n_iters;
    if (niters < 1) niters = 1;

    double t_begin = omp_get_wtime();

    // --- Read inputs ---
    double t_read0 = omp_get_wtime();
    CSR A = buReadSparseMatrix(arguments.input_matrix);
    int n = A.n;
    Vector Xin = vector_read(arguments.input_x, n);
    Vector Bin = vector_read(arguments.input_y, A.m);
    printf("Matrix name : %s\n", arguments.input_matrix);

    int isgpu[8];
    int nrank = read_ints(arguments.input_gpu, isgpu, 8);
    if (nrank != 2) ABORT("is_gpu must have 2 entries (got %d)", nrank)
    int gpu_rank = -1, gpu_count = 0;
    for (int r = 0; r < 2; r++) if (isgpu[r] == 1) { gpu_rank = r; gpu_count++; }
    if (gpu_count != 1) ABORT("is_gpu must mark exactly one rank as GPU (got %d)", gpu_count)
    int cpu_rank = 1 - gpu_rank;

    int *part; ALLOC_ARRAY(part, A.m);
    int np = read_ints(arguments.input_part, part, A.m);
    if (np != A.m) ABORT("Partition has %d entries, expected %d", np, A.m)

    // --- Build permutation: GPU rows first, CPU rows after ---
    int *perm; ALLOC_ARRAY(perm, A.m);
    int *perm_inv; ALLOC_ARRAY(perm_inv, A.m);
    int n_gpu = 0, n_cpu = 0;
    for (int i = 0; i < A.m; i++) if (part[i] == gpu_rank) perm[n_gpu++] = i;
    int gpu_end = n_gpu;
    for (int i = 0; i < A.m; i++) if (part[i] == cpu_rank) perm[gpu_end + n_cpu++] = i;
    if (n_gpu + n_cpu != A.m) ABORT("Partition has %d unmapped rows", A.m - n_gpu - n_cpu)
    for (int i = 0; i < A.m; i++) perm_inv[perm[i]] = i;
    printf("n_gpu=%d  n_cpu=%d  total=%d\n", n_gpu, n_cpu, A.m);

    // --- Permute matrix, X, B ---
    CSR Aperm = csr_permute(A, perm, perm_inv);
    freeSparseMatrix(&A);

    Vector Xperm = vector_init(n);
    Vector Bperm = vector_init(n);
    for (int i = 0; i < n; i++) Xperm.vals[i] = Xin.vals[perm[i]];
    for (int i = 0; i < n; i++) Bperm.vals[i] = Bin.vals[perm[i]];
    vector_destroy(&Xin);
    vector_destroy(&Bin);

    // Slice for GPU and CPU.
    CSR Agpu = csr_row_slice(Aperm, 0, n_gpu);
    CSR Acpu = csr_row_slice(Aperm, n_gpu, n_gpu + n_cpu);
    double t_read1 = omp_get_wtime();

    // --- CUDA / cuBLAS / cuSPARSE setup ---
    CHECK_CUDA(cudaSetDevice(0))
    cublasHandle_t   bh; CHECK_CUBLAS(cublasCreate(&bh))
    cusparseHandle_t ch; CHECK_CUSPARSE(cusparseCreate(&ch))
    cudaStream_t compute_s, copy_s;
    CHECK_CUDA(cudaStreamCreate(&compute_s))
    CHECK_CUDA(cudaStreamCreate(&copy_s))
    CHECK_CUBLAS(cublasSetStream(bh, compute_s))
    CHECK_CUSPARSE(cusparseSetStream(ch, compute_s))
    cudaEvent_t in_ready_event, scatter_done_event;
    CHECK_CUDA(cudaEventCreateWithFlags(&in_ready_event, cudaEventDisableTiming))
    CHECK_CUDA(cudaEventCreateWithFlags(&scatter_done_event, cudaEventDisableTiming))

    Device_CSR dAgpu; CHECK_CUSPARSE(device_csr_create(Agpu, &dAgpu))

    Device_Vector X, B, Y, V, P, R, R_0, S, T;
    CHECK_CUSPARSE(device_vector_init(n, &X))
    CHECK_CUSPARSE(device_vector_init(n, &B))
    CHECK_CUSPARSE(device_vector_init(n, &Y))
    CHECK_CUSPARSE(device_vector_init(n, &V))
    CHECK_CUSPARSE(device_vector_init(n, &P))
    CHECK_CUSPARSE(device_vector_init(n, &R))
    CHECK_CUSPARSE(device_vector_init(n, &R_0))
    CHECK_CUSPARSE(device_vector_init(n, &S))
    CHECK_CUSPARSE(device_vector_init(n, &T))
    CHECK_CUSPARSE(device_vector_toGPU(Xperm, X))
    CHECK_CUSPARSE(device_vector_toGPU(Bperm, B))
    CHECK_CUDA(cudaMemset(V.vals, 0, n * sizeof(double)))
    CHECK_CUDA(cudaMemset(P.vals, 0, n * sizeof(double)))

    cusparseDnVecDescr_t Y_gpu_desc, V_gpu_desc, T_gpu_desc;
    CHECK_CUSPARSE(cusparseCreateDnVec(&Y_gpu_desc, n_gpu > 0 ? n_gpu : 1, Y.vals, CUDA_R_64F))
    CHECK_CUSPARSE(cusparseCreateDnVec(&V_gpu_desc, n_gpu > 0 ? n_gpu : 1, V.vals, CUDA_R_64F))
    CHECK_CUSPARSE(cusparseCreateDnVec(&T_gpu_desc, n_gpu > 0 ? n_gpu : 1, T.vals, CUDA_R_64F))

    Device_Buffer_SpMV spmv_buf;
    {
        size_t bsize = 0;
        const double a = 1.0, b = 0.0;
        CHECK_CUSPARSE(cusparseSpMV_bufferSize(ch, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &a, dAgpu.desc, X.desc, &b, Y_gpu_desc, CUDA_R_64F,
            CUSPARSE_SPMV_ALG_DEFAULT, &bsize))
        CHECK_CUDA(cudaMalloc(&spmv_buf, bsize))
    }

    double *pin_input;  CHECK_CUDA(cudaMallocHost((void**)&pin_input, n * sizeof(double)))
    double *pin_cpu_out;
    CHECK_CUDA(cudaMallocHost((void**)&pin_cpu_out,
                              (n_cpu > 0 ? n_cpu : 1) * sizeof(double)))

    printf("LOG: Setup done\n");

    // --- Initial residual: R = B - A*X  (host-pointer mode is fine here) ---
    CHECK_CUSPARSE(hybrid_spmv(ch, dAgpu.desc, X.desc, Y_gpu_desc, Acpu,
                               X, pin_input, Y, pin_cpu_out,
                               n_gpu, n_cpu, n, compute_s, copy_s,
                               in_ready_event, scatter_done_event, spmv_buf))
    CHECK_CUDA(cudaMemcpyAsync(R.vals, B.vals, n * sizeof(double),
                               cudaMemcpyDeviceToDevice, compute_s))
    const double m_one = -1.0;
    CHECK_CUBLAS(cublasDaxpy(bh, n, &m_one, Y.vals, 1, R.vals, 1)) // R -= Y
    CHECK_CUDA(cudaMemcpyAsync(R_0.vals, R.vals, n * sizeof(double),
                               cudaMemcpyDeviceToDevice, compute_s))

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

    double t_loop0 = omp_get_wtime();
    CHECK_CUBLAS(cublasSetPointerMode(bh, CUBLAS_POINTER_MODE_DEVICE))

    for (int it = 0; it < niters; it++) {
        // rho_new = R . R_0 ; beta = (rho_new/rho)*(alpha/omega) ; rho = rho_new
        CHECK_CUBLAS(cublasDdot(bh, n, R.vals, 1, R_0.vals, 1, d_rho_new))
        hhp_dp_update_beta(d_beta, d_rho, d_rho_new, d_alpha, d_omega, compute_s);

        // P = (P - omega*V)*beta + R   (uses OLD omega) -- one fused kernel
        hhp_dp_vecop_P(P.vals, V.vals, R.vals, d_omega, d_beta, n, compute_s);

        // V = A * P  (overlapped hybrid SpMV)
        CHECK_CUSPARSE(hybrid_spmv(ch, dAgpu.desc, P.desc, V_gpu_desc, Acpu,
                                   P, pin_input, V, pin_cpu_out,
                                   n_gpu, n_cpu, n, compute_s, copy_s,
                                   in_ready_event, scatter_done_event, spmv_buf))

        // rv = R_0 . V ; alpha = rho/rv ; neg_a = -alpha
        CHECK_CUBLAS(cublasDdot(bh, n, R_0.vals, 1, V.vals, 1, d_rv))
        hhp_dp_update_alpha(d_alpha, d_neg_a, d_rho, d_rv, compute_s);

        // S = R - alpha*V   -- one fused kernel
        hhp_dp_vecop_S(S.vals, R.vals, V.vals, d_neg_a, n, compute_s);

        // T = A * S  (overlapped hybrid SpMV)
        CHECK_CUSPARSE(hybrid_spmv(ch, dAgpu.desc, S.desc, T_gpu_desc, Acpu,
                                   S, pin_input, T, pin_cpu_out,
                                   n_gpu, n_cpu, n, compute_s, copy_s,
                                   in_ready_event, scatter_done_event, spmv_buf))

        // ts = T.S ; tt = T.T ; omega = ts/tt ; neg_w = -omega
        CHECK_CUBLAS(cublasDdot(bh, n, T.vals, 1, S.vals, 1, d_ts))
        CHECK_CUBLAS(cublasDdot(bh, n, T.vals, 1, T.vals, 1, d_tt))
        hhp_dp_update_omega(d_omega, d_neg_w, d_ts, d_tt, compute_s);

        // X += alpha*P + omega*S ; R = S - omega*T   -- one fused kernel
        hhp_dp_vecop_XR(X.vals, R.vals, P.vals, S.vals, T.vals, d_alpha, d_omega, d_neg_w, n, compute_s);

        // tol = S.S  (parity with reference; result unused)
        CHECK_CUBLAS(cublasDdot(bh, n, S.vals, 1, S.vals, 1, d_tol))
    }

    CHECK_CUBLAS(cublasSetPointerMode(bh, CUBLAS_POINTER_MODE_HOST))
    cudaDeviceSynchronize();
    double t_loop1 = omp_get_wtime();

    // --- Final relative residual: ||A*X - B|| / ||B|| ---
    CHECK_CUSPARSE(hybrid_spmv(ch, dAgpu.desc, X.desc, Y_gpu_desc, Acpu,
                               X, pin_input, Y, pin_cpu_out,
                               n_gpu, n_cpu, n, compute_s, copy_s,
                               in_ready_event, scatter_done_event, spmv_buf))
    CHECK_CUBLAS(cublasDaxpy(bh, n, &m_one, B.vals, 1, Y.vals, 1)) // Y -= B
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

    // --- Copy X back and unpermute to original ordering before writing ---
    Vector Xh_perm = vector_init(n);
    CHECK_CUSPARSE(device_vector_toCPU(X, Xh_perm))
    Vector Xout = vector_init(n);
    for (int i = 0; i < n; i++) Xout.vals[perm[i]] = Xh_perm.vals[i];
    vector_write(arguments.output_x, Xout);

    // --- Cleanup ---
    cudaFree(d_rho); cudaFree(d_alpha); cudaFree(d_omega); cudaFree(d_beta);
    cudaFree(d_neg_a); cudaFree(d_neg_w); cudaFree(d_one); cudaFree(d_neg_one);
    cudaFree(d_rho_new); cudaFree(d_rv); cudaFree(d_ts); cudaFree(d_tt); cudaFree(d_tol);
    vector_destroy(&Xh_perm);
    vector_destroy(&Xout);
    vector_destroy(&Xperm);
    vector_destroy(&Bperm);
    CHECK_CUSPARSE(device_vector_destroy(&X))
    CHECK_CUSPARSE(device_vector_destroy(&B))
    CHECK_CUSPARSE(device_vector_destroy(&Y))
    CHECK_CUSPARSE(device_vector_destroy(&V))
    CHECK_CUSPARSE(device_vector_destroy(&P))
    CHECK_CUSPARSE(device_vector_destroy(&R))
    CHECK_CUSPARSE(device_vector_destroy(&R_0))
    CHECK_CUSPARSE(device_vector_destroy(&S))
    CHECK_CUSPARSE(device_vector_destroy(&T))
    cusparseDestroyDnVec(Y_gpu_desc);
    cusparseDestroyDnVec(V_gpu_desc);
    cusparseDestroyDnVec(T_gpu_desc);
    CHECK_CUDA(cudaFree(spmv_buf))
    CHECK_CUDA(cudaFreeHost(pin_input))
    CHECK_CUDA(cudaFreeHost(pin_cpu_out))
    cudaEventDestroy(in_ready_event);
    cudaEventDestroy(scatter_done_event);
    cudaStreamDestroy(compute_s);
    cudaStreamDestroy(copy_s);
    CHECK_CUSPARSE(device_csr_destroy(&dAgpu))
    CHECK_CUBLAS(cublasDestroy(bh))
    CHECK_CUSPARSE(cusparseDestroy(ch))
    freeSparseMatrix(&Aperm);
    freeSparseMatrix(&Agpu);
    freeSparseMatrix(&Acpu);
    FREE_AND_NULL(part);
    FREE_AND_NULL(perm);
    FREE_AND_NULL(perm_inv);

    return EXIT_SUCCESS;
}
