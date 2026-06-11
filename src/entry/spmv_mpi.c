#include "stdio.h"
#include <math.h>
#include <omp.h>
#include <stddef.h>
#include <stdlib.h>

#include "hhp_common.h"
#include "hhp_matrix.h"
#include "hhp_cpu.h"
#include "hhp_util.h"

#include "mmio.h"
#include "mpi.h"

#include "argp.h"
#include "unistd.h"

// SpMV-only MPI (CPU) benchmark.
// Distributes the matrix row-wise across ranks (one rank per core) and times
// repeated Y = A*X. Intended for heterogeneous P-core/E-core partitioning:
// give P-core ranks more rows than E-core ranks via a weighted partition.
// No solver, no dot products -- pure distributed SpMV with halo exchange.

struct arguments {
    char *input_matrix;
    char *input_x;
    char *input_part;
    char *input_y;
    char *output_x;
    int n_iters;
};

#define OPT_INPUT_MATRIX 'm'
#define OPT_OUTPUT 'o'
#define OPT_INPUT_X 'x'
#define OPT_INPUT_PART 'p'
#define OPT_INPUT_Y 'y'
#define OPT_N_ITERS 'n'

static struct argp_option options[] = {
    {"input-matrix", OPT_INPUT_MATRIX, "FILE", 0, "Input matrix market file path"},
    {"input-x", OPT_INPUT_X, "FILE", 0, "Input X vector file path"},
    {"input-part", OPT_INPUT_PART, "FILE", 0, "Input Partition vector file path"},
    {"input-y", OPT_INPUT_Y, "FILE", 0, "Target Y vector (optional, for verification)"},
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
        case OPT_INPUT_Y:      arguments->input_y = arg; break;
        case OPT_OUTPUT:       arguments->output_x = arg; break;
        case OPT_N_ITERS:      arguments->n_iters = strtol(arg, &buf, 10); break;
        case ARGP_KEY_ARG:     return 0;
        default:               return ARGP_ERR_UNKNOWN;
    }
    return 0;
}

static char doc[] = "SpMV-only MPI CPU benchmark -- distributed Y = A*X, timed, with per-rank profiling";
static char args_doc[] = "";
static struct argp argp = {options, parse_opt, args_doc, doc};

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    struct arguments arguments = {};
    arguments.n_iters = 1;
    argp_parse(&argp, argc, argv, 0, 0, &arguments);

    if (!arguments.input_matrix || !arguments.input_x || !arguments.input_part || !arguments.output_x) {
        fprintf(stderr, "Error: -m, -x, -p, and -o must all be specified.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (mpi_rank == 0) {
        if (access(arguments.input_matrix, F_OK) == -1)
            ABORT("Input matrix file '%s' does not exist\n", arguments.input_matrix)
        if (access(arguments.input_x, F_OK) == -1)
            ABORT("Input vector file '%s' does not exist\n", arguments.input_x)
        if (access(arguments.input_part, F_OK) == -1)
            ABORT("Partition file '%s' does not exist\n", arguments.input_part)
    }

    int niters = arguments.n_iters;
    if (niters < 1) niters = 1;
    int have_target = (arguments.input_y != NULL);

    double t_begin = omp_get_wtime();

    iVector partvec;
    SHARD_CSC A;
    Vector X;
    Vector B = {};
    int ncols = 0;

    {   // READ INPUT DATA AND DISTRIBUTE
        CSR bigmat = {};
        double t_read0 = omp_get_wtime();

        if (mpi_rank == 0) {
            printf("Matrix name : %s\n", arguments.input_matrix);
            bigmat = buReadSparseMatrix(arguments.input_matrix);
            ncols = bigmat.n;
        }
        MPI_Bcast(&ncols, 1, MPI_INT, 0, MPI_COMM_WORLD);

        partvec = MPI_ivector_read_bcast(arguments.input_part, ncols);
        A = MPI_CSR_split_row(bigmat, partvec);
        X = MPI_vector_read_parted(arguments.input_x, ncols, partvec);
        if (have_target)
            B = MPI_vector_read_parted(arguments.input_y, ncols, partvec);

        if (mpi_rank == 0)
            freeSparseMatrix(&bigmat);

        (void)t_read0;
    }

    if (mpi_rank == 0)
        printf("LOG: Finished Creating Shard CSC\n");

    Vector Y = vector_init_const(A.loc.m, 0);

    // Warm-up SpMV (first call pays MPI setup / page-fault costs)
    MPI_SHARD_CSC_mpi_spmxv_seq(A, X, Y, MPI_COMM_WORLD, NULL);

    Iter_Profile *iprof = calloc(niters, sizeof(Iter_Profile));

    MPI_Barrier(MPI_COMM_WORLD);
    double t_spmv0 = omp_get_wtime();
    for (int i = 0; i < niters; i++) {
        SpMV_Profile sp = {0};
        double t0 = omp_get_wtime();
        MPI_SHARD_CSC_mpi_spmxv_seq(A, X, Y, MPI_COMM_WORLD, &sp);
        double t1 = omp_get_wtime();
        iprof[i].spmv = t1 - t0;
        iprof[i].vector_ops = 0.0;
        iprof[i].spmv_detail = sp;
    }
    double t_spmv1 = omp_get_wtime();
    double spmv_total = t_spmv1 - t_spmv0;

    // Verify against [1, 2, ..., n] by gathering Y to rank 0.
    double max_abs_err = -1.0;
    {
        if (mpi_rank == 0) {
            Vector Ybig = vector_init(ncols);
            MPI_vector_gather(Y, Ybig, A.gind, 0, MPI_COMM_WORLD);
            max_abs_err = 0.0;
            for (size_t i = 0; i < Ybig.nvals; i++) {
                double err = fabs(Ybig.vals[i] - (double)(i + 1));
                if (err > max_abs_err) max_abs_err = err;
            }
            vector_write(arguments.output_x, Ybig);
            vector_destroy(&Ybig);
        } else {
            MPI_vector_gather(Y, (Vector){}, A.gind, 0, MPI_COMM_WORLD);
        }
    }

    // Rank 0 prints the headline metrics.
    if (mpi_rank == 0) {
        printf(
            "n_iters : %d \n"
            "ranks : %d \n"
            "spmv : %lf \n"
            "spmv_per_iter : %e \n"
            "max_abs_err_vs_1ton : %e \n",
            niters, mpi_size, spmv_total, spmv_total / niters, max_abs_err
        );
        printf("everything_total : %lf\n", omp_get_wtime() - t_begin);
        printf("\n----------------------------------------------------------------------\n");
    }

    // Per-rank profile (all ranks print their own line).
    {
        SpMV_Profile acc = {0};
        double acc_spmv = 0;
        printf("PROFILE_HEADER rank iter spmv send_fill local_spmv comm_wait shared_spmv send_wait\n");
        for (int i = 0; i < niters; i++) {
            Iter_Profile *p = &iprof[i];
            printf("PROFILE_ITER %d %d %.6e %.6e %.6e %.6e %.6e %.6e\n",
                mpi_rank, i, p->spmv,
                p->spmv_detail.send_fill, p->spmv_detail.local_spmv,
                p->spmv_detail.comm_wait, p->spmv_detail.shared_spmv,
                p->spmv_detail.send_wait);
            acc_spmv += p->spmv;
            acc.send_fill += p->spmv_detail.send_fill;
            acc.local_spmv += p->spmv_detail.local_spmv;
            acc.comm_wait += p->spmv_detail.comm_wait;
            acc.shared_spmv += p->spmv_detail.shared_spmv;
            acc.send_wait += p->spmv_detail.send_wait;
        }
        printf("PROFILE_ACCUM %d local_rows=%d spmv=%.6e send_fill=%.6e local_spmv=%.6e comm_wait=%.6e shared_spmv=%.6e send_wait=%.6e\n",
            mpi_rank, A.loc.m, acc_spmv,
            acc.send_fill, acc.local_spmv, acc.comm_wait, acc.shared_spmv, acc.send_wait);
    }

    free(iprof);
    ivector_destroy(&partvec);
    vector_destroy(&X);
    if (have_target) vector_destroy(&B);
    vector_destroy(&Y);
    SHARD_CSC_destroy(&A);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
