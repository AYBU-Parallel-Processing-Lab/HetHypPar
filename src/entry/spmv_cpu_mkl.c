#include "stdio.h"
#include <omp.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>

#include "hhp_common.h"
#include "hhp_matrix.h"
#include "hhp_cpu.h"
#include "hhp_util.h"

#include <mkl.h>
#include <mkl_spblas.h>

#include "argp.h"
#include "unistd.h"

// MKL-based SpMV-only CPU benchmark.
// Uses MKL's inspector-executor sparse BLAS (mkl_sparse_d_mv) which is far
// better at saturating memory bandwidth than a naive OpenMP loop.
// Thread count is controlled by MKL_NUM_THREADS (or OMP_NUM_THREADS).

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

static char doc[] = "MKL SpMV-only CPU benchmark -- repeatedly computes Y = A*X via mkl_sparse_d_mv and times it";
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

    double t_read0 = omp_get_wtime();
    CSR A = buReadSparseMatrix(arguments.input_matrix);
    Vector X = vector_read(arguments.input_x, A.n);
    Vector Y = vector_init_const(A.m, 0);
    double t_read1 = omp_get_wtime();

    printf("Matrix name : %s\n", arguments.input_matrix);
    printf("mkl_threads : %d\n", mkl_get_max_threads());

    // Build MKL CSR handle. create_csr references (does not copy) A's arrays,
    // so A must stay alive for the duration. Row pointers: start=I, end=I+1.
    sparse_matrix_t mklA;
    sparse_status_t st = mkl_sparse_d_create_csr(
        &mklA, SPARSE_INDEX_BASE_ZERO, A.m, A.n,
        A.I, A.I + 1, A.J, A.val);
    if (st != SPARSE_STATUS_SUCCESS)
        ABORT("mkl_sparse_d_create_csr failed (status %d)", st)

    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    // Inspector-executor: hint many calls + optimize for best mv performance.
    mkl_sparse_set_mv_hint(mklA, SPARSE_OPERATION_NON_TRANSPOSE, descr, niters + 1);
    mkl_sparse_optimize(mklA);

    // Warm-up SpMV
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, mklA, descr, X.vals, 0.0, Y.vals);

    // Timed loop: repeat the identical SpMV niters times.
    double t_spmv0 = omp_get_wtime();
    for (int i = 0; i < niters; i++) {
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, mklA, descr, X.vals, 0.0, Y.vals);
    }
    double t_spmv1 = omp_get_wtime();

    // Verify: Y should equal [1, 2, ..., m] when X was prepared accordingly.
    double max_abs_err = 0.0;
    for (size_t i = 0; i < Y.nvals; i++) {
        double err = fabs(Y.vals[i] - (double)(i + 1));
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

    vector_write(arguments.output_x, Y);

    mkl_sparse_destroy(mklA);
    freeSparseMatrix(&A);
    vector_destroy(&X);
    vector_destroy(&Y);
    return EXIT_SUCCESS;
}
