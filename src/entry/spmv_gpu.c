#include "stdio.h"
#include <omp.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>

#include "hhp_common.h"
#include "hhp_matrix.h"
#include "hhp_cuda.h"
#include "hhp_cpu.h"
#include "hhp_util.h"

#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <cublas_v2.h>

#include "argp.h"
#include "unistd.h"

// SpMV-only GPU benchmark.
// Reads matrix A and input vector X, repeatedly computes Y = A*X on the GPU
// (cuSPARSE), and times the multiplication. If X was prepared via
// prepare_spmv_input.m, Y should equal [1, 2, ..., n].

struct arguments {
    char *input_matrix;
    char *input_x;
    char *output_x;
    int n_iters;
};

#define OPT_INPUT_MATRIX 'm'
#define OPT_OUTPUT 'o'
#define OPT_INPUT_X 'x'
#define OPT_N_ITERS 'n'

static struct argp_option options[] = {
    {"input-matrix", OPT_INPUT_MATRIX, "FILE", 0, "Input matrix market file path"},
    {"input-x", OPT_INPUT_X, "FILE", 0, "Input X vector file path"},
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
        case OPT_OUTPUT:       arguments->output_x = arg; break;
        case OPT_N_ITERS:      arguments->n_iters = strtol(arg, &buf, 10); break;
        case ARGP_KEY_ARG:     return 0;
        default:               return ARGP_ERR_UNKNOWN;
    }
    return 0;
}

static char doc[] = "SpMV-only GPU benchmark -- repeatedly computes Y = A*X on the GPU and times it";
static char args_doc[] = "";
static struct argp argp = {options, parse_opt, args_doc, doc};

int main(int argc, char *argv[]) {
    struct arguments arguments = {};
    arguments.n_iters = 1;
    argp_parse(&argp, argc, argv, 0, 0, &arguments);

    if (!arguments.input_matrix || !arguments.input_x || !arguments.output_x) {
        fprintf(stderr, "Error: -m, -x, and -o must all be specified.\n");
        return EXIT_FAILURE;
    }
    if (access(arguments.input_matrix, F_OK) == -1)
        ABORT("Input matrix file '%s' does not exist\n", arguments.input_matrix)
    if (access(arguments.input_x, F_OK) == -1)
        ABORT("Input vector file '%s' does not exist\n", arguments.input_x)

    int niters = arguments.n_iters;
    if (niters < 1) niters = 1;

    double t_begin = omp_get_wtime();

    CHECK_CUDA(cudaSetDevice(0))
    cusparseHandle_t cusparseHandle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle))

    const double mx_alpha = 1.00;
    const double mx_beta = 0.00;

    double t_read0 = omp_get_wtime();
    Device_CSR dA;
    int m, n;
    {
        CSR cA = buReadSparseMatrix(arguments.input_matrix);
        m = cA.m;
        n = cA.n;
        CHECK_CUSPARSE(device_csr_create(cA, &dA));
        freeSparseMatrix(&cA);
    }

    Device_Vector X, Y;
    {
        Vector Xt = vector_read(arguments.input_x, n);
        CHECK_CUSPARSE(device_vector_init(Xt.nvals, &X));
        CHECK_CUSPARSE(device_vector_toGPU(Xt, X));
        vector_destroy(&Xt);
    }
    CHECK_CUSPARSE(device_vector_init(m, &Y));
    CHECK_CUSPARSE(device_vector_zero(Y));

    Device_Buffer_SpMV dA_buf;
    CHECK_CUSPARSE(device_buffer_spmv_create(cusparseHandle, dA.desc, X, Y, &mx_alpha, &mx_beta, &dA_buf))
    double t_read1 = omp_get_wtime();

    printf("Matrix name : %s\n", arguments.input_matrix);

    // Warm-up SpMV (first cuSPARSE call pays kernel-launch / plan-setup cost)
    CHECK_CUSPARSE(device_csr_spmv(cusparseHandle, dA, X, Y, mx_alpha, mx_beta, dA_buf))
    cudaDeviceSynchronize();

    // Timed loop: repeat the identical SpMV niters times.
    double t_spmv0 = omp_get_wtime();
    for (int i = 0; i < niters; i++) {
        CHECK_CUSPARSE(device_csr_spmv(cusparseHandle, dA, X, Y, mx_alpha, mx_beta, dA_buf))
    }
    cudaDeviceSynchronize();
    double t_spmv1 = omp_get_wtime();

    // Copy result back and verify against [1, 2, ..., m].
    Vector Yh = vector_init(m);
    CHECK_CUSPARSE(device_vector_toCPU(Y, Yh))

    double max_abs_err = 0.0;
    for (size_t i = 0; i < Yh.nvals; i++) {
        double err = fabs(Yh.vals[i] - (double)(i + 1));
        if (err > max_abs_err) max_abs_err = err;
    }

    double spmv_total = t_spmv1 - t_spmv0;
    printf(
        "n_iters : %d \n"
        "spmv : %lf \n"
        "spmv_per_iter : %e \n"
        "file_read : %lf \n"
        "max_abs_err_vs_1ton : %e \n",
        niters,
        spmv_total,
        spmv_total / niters,
        t_read1 - t_read0,
        max_abs_err
    );
    printf("everything_total : %lf\n", omp_get_wtime() - t_begin);
    printf("\n----------------------------------------------------------------------\n");

    vector_write(arguments.output_x, Yh);

    vector_destroy(&Yh);
    CHECK_CUSPARSE(cusparseDestroy(cusparseHandle))
    return EXIT_SUCCESS;
}
