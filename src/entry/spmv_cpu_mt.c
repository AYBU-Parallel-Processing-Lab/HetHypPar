#include "stdio.h"
#include <omp.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>

#include "hhp_common.h"
#include "hhp_matrix.h"
#include "hhp_cpu.h"
#include "hhp_util.h"

#include "argp.h"
#include "unistd.h"

// Multi-threaded SpMV-only CPU benchmark.
// Same as spmv-cpu but uses the OpenMP-parallel CSR_spmxv_omp kernel.
// Thread count is controlled by the OMP_NUM_THREADS environment variable.

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

static char doc[] = "Multi-threaded SpMV-only CPU benchmark -- repeatedly computes Y = A*X (OpenMP) and times it";
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

    int n_threads = omp_get_max_threads();
    printf("Matrix name : %s\n", arguments.input_matrix);
    printf("omp_threads : %d\n", n_threads);

    // Warm-up SpMV (first call pays page-fault / cache-fill / thread-spawn costs)
    CSR_spmxv_omp(A, X, Y);

    // Timed loop: repeat the identical SpMV niters times.
    double t_spmv0 = omp_get_wtime();
    for (int i = 0; i < niters; i++) {
        CSR_spmxv_omp(A, X, Y);
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

    freeSparseMatrix(&A);
    vector_destroy(&X);
    vector_destroy(&Y);
    return EXIT_SUCCESS;
}
