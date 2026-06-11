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

#include "mmio.h"
#include "mpi.h"

#include "argp.h"
#include "unistd.h"

// Hybrid SpMV ceiling test (2 ranks).
//   rank 0 = GPU (cuSPARSE), holds the top `f` fraction of rows
//   rank 1 = CPU (OpenMP, OMP_NUM_THREADS), holds the remaining rows
// X is replicated on both ranks (no halo exchange) so this measures the pure
// concurrent-multiply ceiling: can GPU + multicore-CPU beat GPU-alone?
//
// Self-contained: rank 0 also times GPU SpMV on the FULL matrix as the baseline,
// so the reported speedup is hybrid_wall vs gpu_full directly.

struct arguments {
    char *input_matrix;
    char *input_x;
    char *output_x;
    int n_iters;
    double gpu_frac;
};

#define OPT_INPUT_MATRIX 'm'
#define OPT_OUTPUT 'o'
#define OPT_INPUT_X 'x'
#define OPT_N_ITERS 'n'
#define OPT_GPU_FRAC 'f'

static struct argp_option options[] = {
    {"input-matrix", OPT_INPUT_MATRIX, "FILE", 0, "Input matrix market file path"},
    {"input-x", OPT_INPUT_X, "FILE", 0, "Input X vector file path"},
    {"output-y", OPT_OUTPUT, "FILE", 0, "Output Y vector file path"},
    {"n-iters", OPT_N_ITERS, "POSITIVE-INTEGER", 0, "Number of SpMV repetitions"},
    {"gpu-frac", OPT_GPU_FRAC, "FLOAT", 0, "Fraction of rows assigned to the GPU rank (0..1)"},
    {0}
};

static error_t parse_opt(int key, char *arg, struct argp_state *state) {
    struct arguments *arguments = state->input;
    char *buf = NULL;
    switch (key) {
        case OPT_INPUT_MATRIX: arguments->input_matrix = arg; break;
        case OPT_INPUT_X:      arguments->input_x = arg; break;
        case OPT_OUTPUT:       arguments->output_x = arg; break;
        case OPT_N_ITERS:      arguments->n_iters = strtol(arg, &buf, 10); break;
        case OPT_GPU_FRAC:     arguments->gpu_frac = strtod(arg, &buf); break;
        case ARGP_KEY_ARG:     return 0;
        default:               return ARGP_ERR_UNKNOWN;
    }
    return 0;
}

static char doc[] = "Hybrid SpMV ceiling test -- GPU rank + multi-threaded CPU rank, replicated X";
static char args_doc[] = "";
static struct argp argp = {options, parse_opt, args_doc, doc};

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    if (mpi_size != 2) {
        if (mpi_rank == 0) fprintf(stderr, "spmv-hybrid requires exactly 2 ranks (rank0=GPU, rank1=CPU)\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    struct arguments arguments = {};
    arguments.n_iters = 1;
    arguments.gpu_frac = 0.87;
    argp_parse(&argp, argc, argv, 0, 0, &arguments);

    if (!arguments.input_matrix || !arguments.input_x || !arguments.output_x) {
        if (mpi_rank == 0) fprintf(stderr, "Error: -m, -x, and -o must all be specified.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int niters = arguments.n_iters;
    if (niters < 1) niters = 1;

    int n = 0, total_m = 0, k = 0;       // k = number of GPU rows
    Vector Xfull = {};
    double gpu_full = 0.0;

    // CPU-rank's local CSR slice (filled on rank 0, sent to rank 1)
    CSR cpuA = {};

    if (mpi_rank == 0) {
        CSR A = buReadSparseMatrix(arguments.input_matrix);
        Xfull = vector_read(arguments.input_x, A.n);
        n = A.n;
        total_m = A.m;
        k = (int)(arguments.gpu_frac * total_m);
        if (k < 0) k = 0;
        if (k > total_m) k = total_m;
        printf("Matrix name : %s\n", arguments.input_matrix);
        printf("gpu_frac : %.4f  gpu_rows : %d  cpu_rows : %d  (total %d)\n",
               arguments.gpu_frac, k, total_m - k, total_m);

        // Broadcast scalars + X to rank 1.
        int dims[3] = {n, total_m, k};
        MPI_Send(dims, 3, MPI_INT, 1, 100, MPI_COMM_WORLD);
        MPI_Send(Xfull.vals, n, MPI_DOUBLE, 1, 101, MPI_COMM_WORLD);

        // Build + send rank 1's CPU slice (rows k..total_m-1), global col indices.
        int cpu_m = total_m - k;
        int cpu_off = A.I[k];
        int cpu_nnz = A.I[total_m] - cpu_off;
        int *cpu_I; ALLOC_ARRAY(cpu_I, cpu_m + 1);
        for (int i = 0; i <= cpu_m; i++) cpu_I[i] = A.I[k + i] - cpu_off;
        MPI_Send(&cpu_m, 1, MPI_INT, 1, 102, MPI_COMM_WORLD);
        MPI_Send(&cpu_nnz, 1, MPI_INT, 1, 103, MPI_COMM_WORLD);
        MPI_Send(cpu_I, cpu_m + 1, MPI_INT, 1, 104, MPI_COMM_WORLD);
        MPI_Send(A.J + cpu_off, cpu_nnz, MPI_INT, 1, 105, MPI_COMM_WORLD);
        MPI_Send(A.val + cpu_off, cpu_nnz, MPI_DOUBLE, 1, 106, MPI_COMM_WORLD);
        FREE_AND_NULL(cpu_I);

        // --- GPU baseline: full matrix ---
        CHECK_CUDA(cudaSetDevice(0))
        cusparseHandle_t h; CHECK_CUSPARSE(cusparseCreate(&h))
        const double a = 1.0, b = 0.0;
        {
            Device_CSR dFull;
            CHECK_CUSPARSE(device_csr_create(A, &dFull))
            Device_Vector dX, dYf;
            CHECK_CUSPARSE(device_vector_init(n, &dX))
            CHECK_CUSPARSE(device_vector_toGPU(Xfull, dX))
            CHECK_CUSPARSE(device_vector_init(total_m, &dYf))
            Device_Buffer_SpMV buf;
            CHECK_CUSPARSE(device_buffer_spmv_create(h, dFull.desc, dX, dYf, &a, &b, &buf))
            CHECK_CUSPARSE(device_csr_spmv(h, dFull, dX, dYf, a, b, buf))
            cudaDeviceSynchronize();
            double t0 = omp_get_wtime();
            for (int i = 0; i < niters; i++)
                CHECK_CUSPARSE(device_csr_spmv(h, dFull, dX, dYf, a, b, buf))
            cudaDeviceSynchronize();
            gpu_full = (omp_get_wtime() - t0) / niters;
            CHECK_CUSPARSE(device_vector_destroy(&dX))
            CHECK_CUSPARSE(device_vector_destroy(&dYf))
            CHECK_CUSPARSE(device_csr_destroy(&dFull))
        }

        // --- GPU slice (rows 0..k-1) for the hybrid run ---
        CSR gpuA = {.I = A.I, .J = A.J, .val = A.val, .m = k, .n = n, .nnz = A.I[k]};
        Device_CSR dG; CHECK_CUSPARSE(device_csr_create(gpuA, &dG))
        Device_Vector dX, dYg;
        CHECK_CUSPARSE(device_vector_init(n, &dX))
        CHECK_CUSPARSE(device_vector_toGPU(Xfull, dX))
        CHECK_CUSPARSE(device_vector_init(k > 0 ? k : 1, &dYg))
        Device_Buffer_SpMV bufG;
        CHECK_CUSPARSE(device_buffer_spmv_create(h, dG.desc, dX, dYg, &a, &b, &bufG))
        // warm-up
        CHECK_CUSPARSE(device_csr_spmv(h, dG, dX, dYg, a, b, bufG))
        cudaDeviceSynchronize();

        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = omp_get_wtime();
        for (int i = 0; i < niters; i++)
            CHECK_CUSPARSE(device_csr_spmv(h, dG, dX, dYg, a, b, bufG))
        cudaDeviceSynchronize();
        double gpu_loop = omp_get_wtime() - t0;
        MPI_Barrier(MPI_COMM_WORLD);
        double wall = omp_get_wtime() - t0;

        // gather rank 1's loop time
        double cpu_loop = 0.0;
        MPI_Recv(&cpu_loop, 1, MPI_DOUBLE, 1, 200, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // gather Y for verification
        Vector Ygpu = vector_init(k > 0 ? k : 1);
        CHECK_CUSPARSE(device_vector_toCPU(dYg, Ygpu))
        Vector Yfull = vector_init(total_m);
        for (int i = 0; i < k; i++) Yfull.vals[i] = Ygpu.vals[i];
        if (total_m - k > 0)
            MPI_Recv(Yfull.vals + k, total_m - k, MPI_DOUBLE, 1, 201, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        double max_abs_err = 0.0;
        for (int i = 0; i < total_m; i++) {
            double err = fabs(Yfull.vals[i] - (double)(i + 1));
            if (err > max_abs_err) max_abs_err = err;
        }
        vector_write(arguments.output_x, Yfull);

        double hybrid = wall / niters;
        printf("gpu_full_per_iter   : %e\n", gpu_full);
        printf("gpu_slice_per_iter  : %e\n", gpu_loop / niters);
        printf("cpu_slice_per_iter  : %e\n", cpu_loop / niters);
        printf("hybrid_wall_per_iter: %e\n", hybrid);
        printf("speedup_vs_gpu_full : %.4f\n", gpu_full / hybrid);
        printf("max_abs_err_vs_1ton : %e\n", max_abs_err);
        printf("\n----------------------------------------------------------------------\n");

        vector_destroy(&Ygpu);
        vector_destroy(&Yfull);
        vector_destroy(&Xfull);
        CHECK_CUSPARSE(device_vector_destroy(&dX))
        CHECK_CUSPARSE(device_vector_destroy(&dYg))
        CHECK_CUSPARSE(device_csr_destroy(&dG))
        CHECK_CUSPARSE(cusparseDestroy(h))
        freeSparseMatrix(&A);
    } else {
        // rank 1 = CPU
        int dims[3];
        MPI_Recv(dims, 3, MPI_INT, 0, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        n = dims[0]; total_m = dims[1]; k = dims[2];
        Xfull = vector_init(n);
        MPI_Recv(Xfull.vals, n, MPI_DOUBLE, 0, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        int cpu_m, cpu_nnz;
        MPI_Recv(&cpu_m, 1, MPI_INT, 0, 102, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&cpu_nnz, 1, MPI_INT, 0, 103, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cpuA.m = cpu_m; cpuA.n = n; cpuA.nnz = cpu_nnz;
        ALLOC_ARRAY(cpuA.I, cpu_m + 1);
        ALLOC_ARRAY(cpuA.J, cpu_nnz > 0 ? cpu_nnz : 1);
        ALLOC_ARRAY(cpuA.val, cpu_nnz > 0 ? cpu_nnz : 1);
        MPI_Recv(cpuA.I, cpu_m + 1, MPI_INT, 0, 104, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(cpuA.J, cpu_nnz, MPI_INT, 0, 105, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(cpuA.val, cpu_nnz, MPI_DOUBLE, 0, 106, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        Vector Yc = vector_init_const(cpu_m > 0 ? cpu_m : 1, 0);
        // warm-up
        if (cpu_m > 0) CSR_spmxv_omp(cpuA, Xfull, Yc);

        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = omp_get_wtime();
        for (int i = 0; i < niters; i++)
            if (cpu_m > 0) CSR_spmxv_omp(cpuA, Xfull, Yc);
        double cpu_loop = omp_get_wtime() - t0;
        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Send(&cpu_loop, 1, MPI_DOUBLE, 0, 200, MPI_COMM_WORLD);
        if (cpu_m > 0)
            MPI_Send(Yc.vals, cpu_m, MPI_DOUBLE, 0, 201, MPI_COMM_WORLD);

        vector_destroy(&Yc);
        vector_destroy(&Xfull);
        freeSparseMatrix(&cpuA);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
