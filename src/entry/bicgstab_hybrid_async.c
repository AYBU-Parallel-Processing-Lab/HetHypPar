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

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

#include "argp.h"
#include "unistd.h"

// Single-process GPU+CPU hybrid BiCGStab -- NO MPI.
//
// Architecture (the cleanest way to actually win):
//   1. Row-permute the matrix once at setup so GPU rows occupy positions
//      0..n_gpu and CPU rows occupy positions n_gpu..n. Column indices are
//      renumbered with the inverse permutation so the math stays equivalent.
//   2. All BiCGStab vectors live full-size on the device in the permuted
//      layout; vector ops and dot products use cuBLAS at full GPU speed.
//   3. Two CUDA streams: compute_stream (cuBLAS / cuSPARSE) and copy_stream
//      (host<->device transfers). Pinned host buffers enable real async
//      copies.
//   4. For each SpMV: the input vector is copied D->H on copy_stream at the
//      same moment GPU SpMV (its row slice) launches on compute_stream --
//      the copy hides inside the GPU work. CPU SpMV on its row slice runs
//      on the host as soon as the copy lands, writing into a pinned host
//      buffer. After both finish, that pinned buffer is copied H->D into
//      the tail (positions n_gpu..n) of the output vector -- a single
//      contiguous transfer.

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

static char doc[] = "Single-process GPU+CPU hybrid BiCGStab (no MPI), overlapped async SpMV";
static char args_doc[] = "";
static struct argp argp = {options, parse_opt, args_doc, doc};

// --- Per-category profiling (enabled by env HHP_PROFILE=1) ---
// When on, we drain compute_s at every category boundary and accumulate
// host wall-clock into dot / spmv / vecops buckets. This PERTURBS the async
// overlap (the loop total grows), so the unperturbed total/speedup must be
// taken from a separate clean run with HHP_PROFILE unset. The bucket *ratios*
// remain a faithful breakdown of where time goes.
static int    g_prof = 0;
static double g_t_dot = 0.0, g_t_spmv = 0.0, g_t_vec = 0.0;
static double g_pt = 0.0;
#define PROF_BEGIN(s) do { if (g_prof) { cudaStreamSynchronize(s); g_pt = omp_get_wtime(); } } while (0)
#define PROF_END(s, acc) do { if (g_prof) { cudaStreamSynchronize(s); (acc) += omp_get_wtime() - g_pt; } } while (0)

// --- Small helpers ---

static int read_ints(const char *path, int *out, int max) {
    FILE *f = fopen(path, "r");
    if (!f) ABORT("Could not open file: %s", path)
    int i = 0, v;
    while (i < max && fscanf(f, "%d", &v) == 1) out[i++] = v;
    fclose(f);
    return i;
}

// Permute rows of A by perm and renumber columns by perm_inv:
// Aperm[i,j] = A[perm[i], perm[j]].
static CSR csr_permute(CSR A, const int *perm, const int *perm_inv) {
    CSR out = {.m = A.m, .n = A.n, .nnz = A.nnz};
    CALLOC_ARRAY(out.I, A.m + 1);
    ALLOC_ARRAY(out.J, A.nnz);
    ALLOC_ARRAY(out.val, A.nnz);
    out.I[0] = 0;
    for (int i = 0; i < A.m; i++) {
        int old_row = perm[i];
        out.I[i + 1] = out.I[i] + (A.I[old_row + 1] - A.I[old_row]);
    }
    for (int i = 0; i < A.m; i++) {
        int old_row = perm[i];
        int old_start = A.I[old_row];
        int cnt = A.I[old_row + 1] - old_start;
        int new_start = out.I[i];
        for (int k = 0; k < cnt; k++) {
            out.J[new_start + k] = perm_inv[A.J[old_start + k]];
            out.val[new_start + k] = A.val[old_start + k];
        }
    }
    return out;
}

// Extract contiguous rows [row_start, row_end) as a CSR slice.
static CSR csr_row_slice(CSR A, int row_start, int row_end) {
    int m_new = row_end - row_start;
    int offset = A.I[row_start];
    int nnz_new = A.I[row_end] - offset;
    CSR out = {.m = m_new, .n = A.n, .nnz = nnz_new};
    CALLOC_ARRAY(out.I, m_new + 1);
    ALLOC_ARRAY(out.J, nnz_new > 0 ? nnz_new : 1);
    ALLOC_ARRAY(out.val, nnz_new > 0 ? nnz_new : 1);
    for (int i = 0; i <= m_new; i++) out.I[i] = A.I[row_start + i] - offset;
    for (int k = 0; k < nnz_new; k++) {
        out.J[k] = A.J[offset + k];
        out.val[k] = A.val[offset + k];
    }
    return out;
}

// One overlapped hybrid SpMV.
//
// Only ONE host sync remains (waiting for D->H so the CPU can read the
// input). The wait between GPU SpMV and the H->D scatter is dropped because
// the two write disjoint memory regions ([0,n_gpu) vs [n_gpu,n_full)). The
// wait between the H->D and the next cuBLAS op on compute_s is dropped in
// favor of a device-side stream wait, so the host returns immediately.
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

    // copy_s must wait for compute_s's pending writes to in_dev before
    // starting the D->H copy.
    CHECK_CUDA(cudaEventRecord(in_ready_event, compute_s))
    CHECK_CUDA(cudaStreamWaitEvent(copy_s, in_ready_event, 0))

    // copy_s: async D->H of full input vector into pinned host buffer.
    CHECK_CUDA(cudaMemcpyAsync(in_pin, in_dev.vals, n_full * sizeof(double),
                               cudaMemcpyDeviceToHost, copy_s))

    // compute_s: GPU SpMV on its row slice. Writes out_dev.vals[0..n_gpu).
    CHECK_CUSPARSE(cusparseSpMV(ch, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, mat_desc, in_desc, &beta, out_gpu_desc,
                                CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buf))

    // SYNC (host-required): the CPU is about to read in_pin -- wait for D->H.
    // This is the only host sync left in the SpMV.
    CHECK_CUDA(cudaStreamSynchronize(copy_s))

    // CPU SpMV (OpenMP) on its row slice, into pinned host buffer.
    if (n_cpu > 0) {
        Vector y_view = {.vals = cpu_out, .nvals = n_cpu};
        Vector x_view = {.vals = in_pin,  .nvals = n_full};
        CSR_spmxv_omp(cpuA, x_view, y_view);
    }

    // H->D scatter on copy_s. No need to wait for GPU SpMV -- it writes
    // out_dev[0..n_gpu), this writes out_dev[n_gpu..n_full); disjoint.
    if (n_cpu > 0) {
        CHECK_CUDA(cudaMemcpyAsync(out_dev.vals + n_gpu, cpu_out,
                                   n_cpu * sizeof(double),
                                   cudaMemcpyHostToDevice, copy_s))
        // Tell compute_s to wait for the scatter device-side -- no host sync.
        CHECK_CUDA(cudaEventRecord(scatter_done_event, copy_s))
        CHECK_CUDA(cudaStreamWaitEvent(compute_s, scatter_done_event, 0))
    }
    // GPU SpMV's writes to out_dev[0..n_gpu) are naturally ordered before
    // subsequent compute_s ops because they share compute_s -- no extra wait.

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

    g_prof = (getenv("HHP_PROFILE") != NULL);

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

    // Device GPU-slice matrix
    Device_CSR dAgpu; CHECK_CUSPARSE(device_csr_create(Agpu, &dAgpu))

    // Full-size device vectors (in permuted layout)
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

    // SpMV input descriptor (size n, pointing at the current input device buf).
    // We create one per "input vector identity" lazily by binding cusparseDnVecDescr_t
    // to each of X/P/S since they are reused. Simpler: just use the descriptors
    // already in Device_Vector for inputs (they are size n).
    // For SpMV outputs, we need n_gpu-sized descriptors that write into the
    // device output buffer's head -- create those manually.
    cusparseDnVecDescr_t Y_gpu_desc, V_gpu_desc, T_gpu_desc;
    CHECK_CUSPARSE(cusparseCreateDnVec(&Y_gpu_desc, n_gpu > 0 ? n_gpu : 1, Y.vals, CUDA_R_64F))
    CHECK_CUSPARSE(cusparseCreateDnVec(&V_gpu_desc, n_gpu > 0 ? n_gpu : 1, V.vals, CUDA_R_64F))
    CHECK_CUSPARSE(cusparseCreateDnVec(&T_gpu_desc, n_gpu > 0 ? n_gpu : 1, T.vals, CUDA_R_64F))

    // SpMV buffer (config: Agpu * Xn -> Yn_gpu).
    Device_Buffer_SpMV spmv_buf;
    {
        size_t bsize = 0;
        const double a = 1.0, b = 0.0;
        CHECK_CUSPARSE(cusparseSpMV_bufferSize(ch, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &a, dAgpu.desc, X.desc, &b, Y_gpu_desc, CUDA_R_64F,
            CUSPARSE_SPMV_ALG_DEFAULT, &bsize))
        CHECK_CUDA(cudaMalloc(&spmv_buf, bsize))
    }

    // Pinned host buffers for the overlap.
    double *pin_input;  CHECK_CUDA(cudaMallocHost((void**)&pin_input, n * sizeof(double)))
    double *pin_cpu_out;
    CHECK_CUDA(cudaMallocHost((void**)&pin_cpu_out,
                              (n_cpu > 0 ? n_cpu : 1) * sizeof(double)))

    printf("LOG: Setup done\n");

    // --- Initial residual: R = B - A*X ---
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

    double rho = 1.0, alpha_s = 1.0, omega = 1.0;
    double t_loop0 = omp_get_wtime();

    for (int it = 0; it < niters; it++) {
        // rho_{n+1} = R . R_0
        double tmp_rho;
        PROF_BEGIN(compute_s);
        CHECK_CUBLAS(device_vector_dot(bh, R, R_0, &tmp_rho))
        PROF_END(compute_s, g_t_dot);
        double beta = (tmp_rho / rho) * (alpha_s / omega);
        rho = tmp_rho;

        // P = (P - omega*V)*beta + R   (vector ops)
        const double one = 1.0;
        PROF_BEGIN(compute_s);
        CHECK_CUBLAS(device_vector_scale(bh, omega, V))            // V *= omega
        CHECK_CUBLAS(device_vector_axpy(bh, V, m_one, P))          // P = P - V
        CHECK_CUBLAS(device_vector_scale(bh, beta, P))             // P *= beta
        CHECK_CUBLAS(cublasDaxpy(bh, n, &one, R.vals, 1, P.vals, 1)) // P = P + R
        PROF_END(compute_s, g_t_vec);

        // V = A * P  (overlapped)
        PROF_BEGIN(compute_s);
        CHECK_CUSPARSE(hybrid_spmv(ch, dAgpu.desc, P.desc, V_gpu_desc, Acpu,
                                   P, pin_input, V, pin_cpu_out,
                                   n_gpu, n_cpu, n, compute_s, copy_s,
                                   in_ready_event, scatter_done_event, spmv_buf))
        PROF_END(compute_s, g_t_spmv);

        double tmp_rv;
        PROF_BEGIN(compute_s);
        CHECK_CUBLAS(device_vector_dot(bh, R_0, V, &tmp_rv))
        PROF_END(compute_s, g_t_dot);
        alpha_s = rho / tmp_rv;

        // S = R; S -= alpha*V   (vector ops)
        const double m_alpha = -alpha_s;
        PROF_BEGIN(compute_s);
        CHECK_CUDA(cudaMemcpyAsync(S.vals, R.vals, n * sizeof(double),
                                   cudaMemcpyDeviceToDevice, compute_s))
        CHECK_CUBLAS(cublasDaxpy(bh, n, &m_alpha, V.vals, 1, S.vals, 1))
        PROF_END(compute_s, g_t_vec);

        // T = A * S  (overlapped)
        PROF_BEGIN(compute_s);
        CHECK_CUSPARSE(hybrid_spmv(ch, dAgpu.desc, S.desc, T_gpu_desc, Acpu,
                                   S, pin_input, T, pin_cpu_out,
                                   n_gpu, n_cpu, n, compute_s, copy_s,
                                   in_ready_event, scatter_done_event, spmv_buf))
        PROF_END(compute_s, g_t_spmv);

        double tmp_ts, tmp_tt;
        PROF_BEGIN(compute_s);
        CHECK_CUBLAS(device_vector_dot(bh, T, S, &tmp_ts))
        CHECK_CUBLAS(device_vector_dot(bh, T, T, &tmp_tt))
        PROF_END(compute_s, g_t_dot);
        omega = tmp_ts / tmp_tt;

        // X += alpha*P + omega*S ; R = S - omega*T   (vector ops)
        const double m_omega = -omega;
        PROF_BEGIN(compute_s);
        CHECK_CUBLAS(cublasDaxpy(bh, n, &alpha_s, P.vals, 1, X.vals, 1))
        CHECK_CUBLAS(cublasDaxpy(bh, n, &omega,   S.vals, 1, X.vals, 1))
        CHECK_CUDA(cudaMemcpyAsync(R.vals, S.vals, n * sizeof(double),
                                   cudaMemcpyDeviceToDevice, compute_s))
        CHECK_CUBLAS(cublasDaxpy(bh, n, &m_omega, T.vals, 1, R.vals, 1))
        PROF_END(compute_s, g_t_vec);

        double tol;
        PROF_BEGIN(compute_s);
        CHECK_CUBLAS(device_vector_dot(bh, S, S, &tol))
        PROF_END(compute_s, g_t_dot);
        (void)tol;
    }
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
    if (g_prof) {
        double psum = g_t_dot + g_t_spmv + g_t_vec;
        printf("PROFILE_enabled : 1\n");
        printf("PROFILE_loop_total : %lf\n", t_loop1 - t_loop0);
        printf("PROFILE_dot : %lf\n", g_t_dot);
        printf("PROFILE_spmv : %lf\n", g_t_spmv);
        printf("PROFILE_vec : %lf\n", g_t_vec);
        printf("PROFILE_sum : %lf\n", psum);
        printf("PROFILE_dot_pct : %.2f\n",  psum > 0 ? 100.0 * g_t_dot  / psum : 0.0);
        printf("PROFILE_spmv_pct : %.2f\n", psum > 0 ? 100.0 * g_t_spmv / psum : 0.0);
        printf("PROFILE_vec_pct : %.2f\n",  psum > 0 ? 100.0 * g_t_vec  / psum : 0.0);
    }
    printf("\n----------------------------------------------------------------------\n");

    // --- Copy X back and unpermute to original ordering before writing ---
    Vector Xh_perm = vector_init(n);
    CHECK_CUSPARSE(device_vector_toCPU(X, Xh_perm))
    Vector Xout = vector_init(n);
    for (int i = 0; i < n; i++) Xout.vals[perm[i]] = Xh_perm.vals[i];
    vector_write(arguments.output_x, Xout);

    // --- Cleanup ---
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
