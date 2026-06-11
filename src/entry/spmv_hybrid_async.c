#include "stdio.h"
#include <math.h>
#include <omp.h>
#include <stddef.h>
#include <stdlib.h>

#include "hhp_common.h"
#include "hhp_matrix.h"
#include "hhp_cuda.h"
#include "hhp_cpu.h"
#include "hhp_util.h"

#include <cuda_runtime_api.h>
#include <cusparse.h>

#include "argp.h"
#include "unistd.h"

// Single-process GPU+CPU hybrid SpMV -- NO MPI.
// The matrix rows are split by a 2-rank partition file + an is_gpu file (exactly
// one rank marked GPU). Rows whose partition id == the GPU rank go to the GPU;
// the rest go to the CPU. X is shared (one host copy, one device copy), so there
// is no halo exchange. Each iteration launches the GPU SpMV asynchronously, then
// immediately runs the CPU SpMV with OpenMP, then synchronizes -- the two devices
// compute concurrently. Compared against GPU-alone on the full matrix.

struct arguments {
    char *input_matrix;
    char *input_x;
    char *input_part;
    char *input_gpu;
    char *output_x;
    int n_iters;
};

#define OPT_INPUT_MATRIX 'm'
#define OPT_OUTPUT 'o'
#define OPT_INPUT_X 'x'
#define OPT_INPUT_PART 'p'
#define OPT_INPUT_GPU 'g'
#define OPT_N_ITERS 'n'

static struct argp_option options[] = {
    {"input-matrix", OPT_INPUT_MATRIX, "FILE", 0, "Input matrix market file path"},
    {"input-x", OPT_INPUT_X, "FILE", 0, "Input X vector file path"},
    {"input-part", OPT_INPUT_PART, "FILE", 0, "2-rank partition vector file path"},
    {"is-gpu", OPT_INPUT_GPU, "FILE", 0, "is_gpu file (2 lines, exactly one '1')"},
    {"output-y", OPT_OUTPUT, "FILE", 0, "Output Y vector file path"},
    {"n-iters", OPT_N_ITERS, "POSITIVE-INTEGER", 0, "Number of SpMV repetitions"},
    {0}
};

static error_t parse_opt(int key, char *arg, struct argp_state *state) {
    struct arguments *arguments = state->input;
    char *buf = NULL;
    switch (key) {
        case OPT_INPUT_MATRIX: arguments->input_matrix = arg; break;
        case OPT_INPUT_X:      arguments->input_x = arg; break;
        case OPT_INPUT_PART:   arguments->input_part = arg; break;
        case OPT_INPUT_GPU:    arguments->input_gpu = arg; break;
        case OPT_OUTPUT:       arguments->output_x = arg; break;
        case OPT_N_ITERS:      arguments->n_iters = strtol(arg, &buf, 10); break;
        case ARGP_KEY_ARG:     return 0;
        default:               return ARGP_ERR_UNKNOWN;
    }
    return 0;
}

static char doc[] = "Single-process GPU+CPU hybrid SpMV (no MPI) -- async GPU overlapped with OpenMP CPU";
static char args_doc[] = "";
static struct argp argp = {options, parse_opt, args_doc, doc};

// Read up to `max` whitespace-separated ints from a file. Returns count read.
static int read_ints(const char *path, int *out, int max) {
    FILE *f = fopen(path, "r");
    if (!f) ABORT("Could not open file: %s", path)
    int i = 0, v;
    while (i < max && fscanf(f, "%d", &v) == 1) out[i++] = v;
    fclose(f);
    return i;
}

// Gather the rows listed in rows[0..k-1] from A into a new CSR (global columns
// preserved, so it multiplies against the full X). Caller frees with freeSparseMatrix.
static CSR csr_row_gather(CSR A, const int *rows, int k) {
    CSR out = {.m = k, .n = A.n, .nnz = 0};
    for (int j = 0; j < k; j++) out.nnz += A.I[rows[j] + 1] - A.I[rows[j]];
    CALLOC_ARRAY(out.I, k + 1);
    ALLOC_ARRAY(out.J, out.nnz > 0 ? out.nnz : 1);
    ALLOC_ARRAY(out.val, out.nnz > 0 ? out.nnz : 1);
    int pos = 0;
    for (int j = 0; j < k; j++) {
        int r = rows[j];
        out.I[j] = pos;
        for (int e = A.I[r]; e < A.I[r + 1]; e++) {
            out.J[pos] = A.J[e];
            out.val[pos] = A.val[e];
            pos++;
        }
    }
    out.I[k] = pos;
    return out;
}

int main(int argc, char *argv[]) {
    struct arguments arguments = {};
    arguments.n_iters = 1;
    argp_parse(&argp, argc, argv, 0, 0, &arguments);

    if (!arguments.input_matrix || !arguments.input_x || !arguments.input_part
        || !arguments.input_gpu || !arguments.output_x) {
        fprintf(stderr, "Error: -m, -x, -p, -g, and -o must all be specified.\n");
        return EXIT_FAILURE;
    }
    if (access(arguments.input_matrix, F_OK) == -1)
        ABORT("Input matrix file '%s' does not exist\n", arguments.input_matrix)
    if (access(arguments.input_x, F_OK) == -1)
        ABORT("Input vector file '%s' does not exist\n", arguments.input_x)
    if (access(arguments.input_part, F_OK) == -1)
        ABORT("Partition file '%s' does not exist\n", arguments.input_part)
    if (access(arguments.input_gpu, F_OK) == -1)
        ABORT("is_gpu file '%s' does not exist\n", arguments.input_gpu)

    int niters = arguments.n_iters;
    if (niters < 1) niters = 1;

    double t_begin = omp_get_wtime();

    CSR A = buReadSparseMatrix(arguments.input_matrix);
    Vector X = vector_read(arguments.input_x, A.n);
    printf("Matrix name : %s\n", arguments.input_matrix);

    // --- is_gpu: exactly 2 ranks, exactly one marked GPU ---
    int isgpu[8];
    int nrank = read_ints(arguments.input_gpu, isgpu, 8);
    if (nrank != 2)
        ABORT("is_gpu file must have exactly 2 ranks (got %d)", nrank)
    int gpu_rank = -1, gpu_count = 0;
    for (int r = 0; r < 2; r++) if (isgpu[r] == 1) { gpu_rank = r; gpu_count++; }
    if (gpu_count != 1)
        ABORT("is_gpu file must mark exactly one rank as GPU (got %d)", gpu_count)
    int cpu_rank = 1 - gpu_rank;

    // --- partition: one entry per row ---
    int *part; ALLOC_ARRAY(part, A.m);
    int np = read_ints(arguments.input_part, part, A.m);
    if (np != A.m)
        ABORT("Partition file has %d entries, expected %d (matrix rows)", np, A.m)

    // --- build GPU / CPU row lists ---
    int *gpu_rows; ALLOC_ARRAY(gpu_rows, A.m);
    int *cpu_rows; ALLOC_ARRAY(cpu_rows, A.m);
    int n_gpu = 0, n_cpu = 0;
    for (int i = 0; i < A.m; i++) {
        if (part[i] == gpu_rank) gpu_rows[n_gpu++] = i;
        else if (part[i] == cpu_rank) cpu_rows[n_cpu++] = i;
        else ABORT("Partition entry %d = %d not in {%d,%d}", i, part[i], gpu_rank, cpu_rank)
    }
    printf("gpu_rows : %d  cpu_rows : %d  (total %d)\n", n_gpu, n_cpu, A.m);

    CSR gpuA = csr_row_gather(A, gpu_rows, n_gpu);
    CSR cpuA = csr_row_gather(A, cpu_rows, n_cpu);

    // --- GPU setup ---
    CHECK_CUDA(cudaSetDevice(0))
    cusparseHandle_t h; CHECK_CUSPARSE(cusparseCreate(&h))
    const double alpha = 1.0, beta = 0.0;

    // GPU-alone baseline on the full matrix.
    double gpu_full = 0.0;
    {
        Device_CSR dFull; CHECK_CUSPARSE(device_csr_create(A, &dFull))
        Device_Vector dX, dY;
        CHECK_CUSPARSE(device_vector_init(A.n, &dX))
        CHECK_CUSPARSE(device_vector_toGPU(X, dX))
        CHECK_CUSPARSE(device_vector_init(A.m, &dY))
        Device_Buffer_SpMV buf;
        CHECK_CUSPARSE(device_buffer_spmv_create(h, dFull.desc, dX, dY, &alpha, &beta, &buf))
        CHECK_CUSPARSE(device_csr_spmv(h, dFull, dX, dY, alpha, beta, buf))
        double t0 = omp_get_wtime();
        for (int i = 0; i < niters; i++)
            CHECK_CUSPARSE(device_csr_spmv(h, dFull, dX, dY, alpha, beta, buf))
        cudaDeviceSynchronize();
        gpu_full = (omp_get_wtime() - t0) / niters;
        CHECK_CUSPARSE(device_vector_destroy(&dX))
        CHECK_CUSPARSE(device_vector_destroy(&dY))
        CHECK_CUSPARSE(device_csr_destroy(&dFull))
    }

    // --- hybrid setup: GPU slice on device, CPU slice on host, shared X ---
    Device_CSR dG; CHECK_CUSPARSE(device_csr_create(gpuA, &dG))
    Device_Vector dX, dYg;
    CHECK_CUSPARSE(device_vector_init(A.n, &dX))
    CHECK_CUSPARSE(device_vector_toGPU(X, dX))
    CHECK_CUSPARSE(device_vector_init(n_gpu > 0 ? n_gpu : 1, &dYg))
    Device_Buffer_SpMV bufG;
    CHECK_CUSPARSE(device_buffer_spmv_create(h, dG.desc, dX, dYg, &alpha, &beta, &bufG))

    Vector Yc = vector_init_const(n_cpu > 0 ? n_cpu : 1, 0);

    // warm-up
    CHECK_CUSPARSE(device_csr_spmv(h, dG, dX, dYg, alpha, beta, bufG))
    if (n_cpu > 0) CSR_spmxv_omp(cpuA, X, Yc);

    // --- timed hybrid loop: launch GPU async, run CPU concurrently, sync ---
    double t0 = omp_get_wtime();
    for (int i = 0; i < niters; i++) {
        CHECK_CUSPARSE(device_csr_spmv_async(h, dG, dX, dYg, alpha, beta, bufG))
        if (n_cpu > 0) CSR_spmxv_omp(cpuA, X, Yc);
        cudaDeviceSynchronize();
    }
    double hybrid = (omp_get_wtime() - t0) / niters;

    // --- breakdown: each device's slice alone ---
    double t1 = omp_get_wtime();
    for (int i = 0; i < niters; i++)
        CHECK_CUSPARSE(device_csr_spmv(h, dG, dX, dYg, alpha, beta, bufG))
    double gpu_slice = (omp_get_wtime() - t1) / niters;

    double cpu_slice = 0.0;
    if (n_cpu > 0) {
        double t2 = omp_get_wtime();
        for (int i = 0; i < niters; i++) CSR_spmxv_omp(cpuA, X, Yc);
        cpu_slice = (omp_get_wtime() - t2) / niters;
    }

    // --- stitch full Y and verify vs [1..n] ---
    Vector Ygpu = vector_init(n_gpu > 0 ? n_gpu : 1);
    CHECK_CUSPARSE(device_vector_toCPU(dYg, Ygpu))
    Vector Yfull = vector_init(A.m);
    for (int j = 0; j < n_gpu; j++) Yfull.vals[gpu_rows[j]] = Ygpu.vals[j];
    for (int j = 0; j < n_cpu; j++) Yfull.vals[cpu_rows[j]] = Yc.vals[j];

    double max_abs_err = 0.0;
    for (int i = 0; i < A.m; i++) {
        double err = fabs(Yfull.vals[i] - (double)(i + 1));
        if (err > max_abs_err) max_abs_err = err;
    }
    vector_write(arguments.output_x, Yfull);

    printf("gpu_full_per_iter   : %e\n", gpu_full);
    printf("gpu_slice_per_iter  : %e\n", gpu_slice);
    printf("cpu_slice_per_iter  : %e\n", cpu_slice);
    printf("hybrid_per_iter     : %e\n", hybrid);
    printf("speedup_vs_gpu_full : %.4f\n", gpu_full / hybrid);
    printf("max_abs_err_vs_1ton : %e\n", max_abs_err);
    printf("everything_total : %lf\n", omp_get_wtime() - t_begin);
    printf("\n----------------------------------------------------------------------\n");

    vector_destroy(&Ygpu);
    vector_destroy(&Yfull);
    vector_destroy(&Yc);
    vector_destroy(&X);
    CHECK_CUSPARSE(device_vector_destroy(&dX))
    CHECK_CUSPARSE(device_vector_destroy(&dYg))
    CHECK_CUSPARSE(device_csr_destroy(&dG))
    CHECK_CUSPARSE(cusparseDestroy(h))
    freeSparseMatrix(&A);
    freeSparseMatrix(&gpuA);
    freeSparseMatrix(&cpuA);
    FREE_AND_NULL(part);
    FREE_AND_NULL(gpu_rows);
    FREE_AND_NULL(cpu_rows);
    return EXIT_SUCCESS;
}
