#include "hhp_wrap_bora.h"
#include "stdio.h"
#include <driver_types.h>
#include <math.h>
#include <omp.h>
#include <stddef.h>
#include <string.h> 
#include <stdlib.h>
#include "bora_spmxv.h"

#include "hhp_common.h"
#include "hhp_matrix.h"
#include "hhp_cuda.h"
#include "hhp_cpu.h"
#include "hhp_util.h"

#include <cuda_runtime_api.h>    // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h> 

#include <time.h>

#include "mmio.h"
#include "mpi.h"

#include "argp.h"
#include "unistd.h"

// ========================= ArgP STUFF ===============================================
// Structure to store parsed arguments
struct arguments {
    char *input_matrix;
    char *input_x;
    char *input_part;
    char *input_y;
    char *output_x;
    int n_iters;
};

// Option keys
#define OPT_INPUT_MATRIX 'm'
#define OPT_OUTPUT 'o'
#define OPT_INPUT_X 'x'
#define OPT_INPUT_PART 'p'
#define OPT_INPUT_Y 'y'
#define OPT_N_ITERS 'n'

// Command line options
static struct argp_option options[] = {
    {"input-matrix", OPT_INPUT_MATRIX, "FILE", 0, "Input matrix market file path"},
    {"input-x", OPT_INPUT_X, "FILE", 0, "Input X vector file path"},
    {"input-part", OPT_INPUT_PART, "FILE", 0, "Input Partition vector file path"},
    {"input-y", OPT_INPUT_Y, "FILE", 0, "Input Y target vector file path"},
    {"output-x", OPT_OUTPUT, "FILE", 0, "Output X vector file path"},
    {"n-iters", OPT_N_ITERS, "POSITIVE-INTEGER", 0, "Number of iterations"},
    {0}
};

// Parser function
static error_t parse_opt(int key, char *arg, struct argp_state *state) {
    struct arguments *arguments = state->input;
    char* buf = NULL;

    switch (key) {
        case OPT_INPUT_MATRIX:
            arguments->input_matrix = arg;
            break;
        case OPT_INPUT_X:
            arguments->input_x = arg;
            break;
        case OPT_INPUT_PART:
            arguments->input_part = arg;
            break;
        case OPT_INPUT_Y:
            arguments->input_y = arg;
            break;
        case OPT_OUTPUT:
            arguments->output_x = arg;
            break;
        case OPT_N_ITERS:
            arguments->n_iters = strtol(arg, &buf, 10);
            break;
        case ARGP_KEY_ARG:
            return 0;
        default:
            return ARGP_ERR_UNKNOWN;
    }
    return 0;
}

// Program documentation
static char doc[] = "Matrix-Vector multiplication program -- multiplies a matrix market format file with a vector";
static char args_doc[] = "";  // We don't take non-option arguments

// Our argp parser
static struct argp argp = {options, parse_opt, args_doc, doc};

// ====================================================================================

static struct {
    double 
        begin,
            file_read_begin,
            file_read_end,

            matrix_split_begin,
            matrix_split_end,

            gpu_transfer_begin,
            gpu_transfer_end,

            spmxv_begin,
                cpu_spmxv_begin,
                cpu_spmxv_end,

                gpu_begin,
                gpu_spmxv_begin,
                gpu_spmxv_end,
                gpu_end,
            spmxv_end,
        end
    ;
}time_stamps = {};

static struct {
    double
        cpu_spmxv,
        gpu_spmxv,
        gpu_total,
        total_spmxv
    ;
}times = {};

int main(int argc, char* argv[]) {  // matrix file ve part vector 
    MPI_Init(&argc, &argv);

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);


    struct arguments arguments = {};
    
    // Parse arguments
    argp_parse(&argp, argc, argv, 0, 0, &arguments);
    
    // Check if all required arguments are provided
    if (!arguments.input_matrix || !arguments.input_x || !arguments.input_part || !arguments.output_x) {
        fprintf(stderr, "Error: All input and output files must be specified.\n");
        return EXIT_FAILURE;
    }
    
    if (mpi_rank == 0) {
        // Check file existence
        if (access(arguments.input_matrix, F_OK) == -1)
            ABORT("Input matrix file '%s' does not exist\n", arguments.input_matrix)

        if (access(arguments.input_x, F_OK) == -1)
            ABORT("Input vector file '%s' does not exist\n", arguments.input_x)
        
        if (access(arguments.input_part, F_OK) == -1)
            ABORT("Input vector file '%s' does not exist\n", arguments.input_part)
        
        if (access(arguments.input_y, F_OK) == -1)
            ABORT("Input vector file '%s' does not exist\n", arguments.input_y)
    }

    int niters = arguments.n_iters;
    
    time_stamps.begin = omp_get_wtime(); // The very Beginning timestamp
    
//-------------------------------------------------------------------------------------------------------    
    // [x] IS FREED?
    iVector partvec;

    // [x] IS FREED?
    SHARD_CSC A;
    
    // [x] IS FREED?
    Vector X; // Initial guess, Final result is stored here
    // [x] IS FREED?
    Vector B; // Target result

    int ncols;

    {   // READ INPUT DATA AND DISTRIBUTE
        
        // [x] IS FREED?
        CSR bigmat = {};

        
        // Read the matrix market file and convert it to CSR format
        time_stamps.file_read_begin = omp_get_wtime();

        if (mpi_rank == 0)
        {
            printf("Matrix name : %s\n",arguments.input_matrix);
            bigmat = buReadSparseMatrix(arguments.input_matrix);
            ncols = bigmat.n;
        }

        MPI_Bcast(&ncols, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // read partvec and distribute it to all ranks
        partvec = MPI_ivector_read_bcast(arguments.input_part, ncols);

        A = MPI_CSR_split_row(bigmat, partvec);

        X = MPI_vector_read_parted(arguments.input_x, ncols, partvec);
        B = MPI_vector_read_parted(arguments.input_y, ncols, partvec);

        if (mpi_rank == 0)
        {
            freeSparseMatrix(&bigmat);
        }

        time_stamps.file_read_end = omp_get_wtime();
    }
    
    if (mpi_rank == 0)
        printf("LOG: Finished Creating Shard CSC\n");
    //-------------------------------------------------------------------------------------------------------


    // CPU Objects
    // [x] IS FREED?
    Vector Y = vector_init_const(A.loc.m, 0);

    // BICGStab variables
    double rho = 1.0;
    double alpha = 1.0;
    double omega = 1.0;
    // [x] IS FREED?
    Vector V = vector_init_const(Y.nvals,0); // Add pieces for this
    // [x] IS FREED?
    Vector P = vector_init_const(Y.nvals, 0); // Add pieces for this
    
    // [x] IS FREED?
    Vector R = vector_init_clone(B); // Add pieces for this
    
    MPI_SHARD_CSC_mpi_spmxv_seq(A, X, Y, MPI_COMM_WORLD, NULL);
    vector_sub_seq(R, Y, R);
    vector_zero(Y);

    Vector R_0 = vector_init_clone(R);
    Vector S = vector_init(Y.nvals);
    Vector T = vector_init(Y.nvals);

    if (mpi_rank == 0)
        printf("LOG: Finished Creating Vectors\n");

//-------------------------------------------------------------------------------------------------------
    Iter_Profile *iprof = calloc(niters, sizeof(Iter_Profile));

    time_stamps.spmxv_begin = omp_get_wtime();

    for (size_t i = 0; i < niters; i++)
    {
        SpMV_Profile sp = {0};
        double t_vec0 = omp_get_wtime();

        // calc rho_n+1
        double temp_rho = MPI_vector_dot(R, R_0, MPI_COMM_WORLD);
        double beta = (temp_rho / rho) * (alpha / omega);
        rho = temp_rho;
        // calc P_n+1
        vector_scale_seq(V, omega, V);
        vector_sub_seq(P, V, P);
        vector_scale_seq(P, beta, P);
        vector_add_seq(P, R, P);

        double t_spmv0 = omp_get_wtime();
        MPI_SHARD_CSC_mpi_spmxv_seq(A, P, V, MPI_COMM_WORLD, &sp);
        double t_spmv1 = omp_get_wtime();

        // calc alpha, S
        alpha = rho/MPI_vector_dot(R_0, V, MPI_COMM_WORLD);
        vector_scale_seq(V, alpha, S);
        vector_sub_seq(R, S, S);

        double t_spmv2 = omp_get_wtime();
        MPI_SHARD_CSC_mpi_spmxv_seq(A, S, T, MPI_COMM_WORLD, &sp);
        double t_spmv3 = omp_get_wtime();

        // calc omega, X, R, tol
        omega = MPI_vector_dot(T, S,MPI_COMM_WORLD)/MPI_vector_dot(T, T,MPI_COMM_WORLD);
        vector_scale_seq(P, alpha, Y);
        vector_add_seq(X, Y, X);
        vector_scale_seq(S, omega, Y);
        vector_add_seq(X, Y, X);
        vector_scale_seq(T, omega, R);
        vector_sub_seq(S, R, R);
        double tol = MPI_vector_dot(S, S,MPI_COMM_WORLD);

        double t_vec1 = omp_get_wtime();

        double spmv_total = (t_spmv1 - t_spmv0) + (t_spmv3 - t_spmv2);
        iprof[i].spmv = spmv_total;
        iprof[i].vector_ops = (t_vec1 - t_vec0) - spmv_total;
        iprof[i].spmv_detail = sp;
    }

    time_stamps.spmxv_end = omp_get_wtime();
    time_stamps.end = time_stamps.spmxv_end;

    //-------------------------------------------------------------------------------------------------------
    // Calculate Relative residual (|| Ax - b || / || b ||)

    MPI_SHARD_CSC_mpi_spmxv_seq(A, X, Y, MPI_COMM_WORLD, NULL);
    vector_sub_seq(B, Y, Y);
    double sy = MPI_vector_dot(Y, Y, MPI_COMM_WORLD);
    double sb = MPI_vector_dot(B, B, MPI_COMM_WORLD);
    double relative_residual = sqrt(sy/sb);

    //-------------------------------------------------------------------------------------------------------

    if (mpi_rank == 0){
        printf(
            "n_iters : %d \n"
            "spmv : %lf \n"
            "file_read : %lf \n"
            "relative_residual : %E\n"
            ,
            niters,
            time_stamps.spmxv_end - time_stamps.spmxv_begin,
            time_stamps.file_read_end - time_stamps.file_read_begin,
            relative_residual
            );

        printf("everything_total : %lf\n",time_stamps.end - time_stamps.begin) ;

        printf("\n----------------------------------------------------------------------\n");
    }

    // Per-iteration profile (all ranks print, prefixed with rank)
    {
        SpMV_Profile acc = {0};
        double acc_spmv = 0, acc_vecops = 0;

        printf("PROFILE_HEADER rank iter spmv vecops send_fill local_spmv comm_wait shared_spmv send_wait\n");
        for (size_t i = 0; i < niters; i++) {
            Iter_Profile *p = &iprof[i];
            printf("PROFILE_ITER %d %zu %.6e %.6e %.6e %.6e %.6e %.6e %.6e\n",
                mpi_rank, i, p->spmv, p->vector_ops,
                p->spmv_detail.send_fill, p->spmv_detail.local_spmv,
                p->spmv_detail.comm_wait, p->spmv_detail.shared_spmv,
                p->spmv_detail.send_wait);
            acc_spmv += p->spmv;
            acc_vecops += p->vector_ops;
            acc.send_fill += p->spmv_detail.send_fill;
            acc.local_spmv += p->spmv_detail.local_spmv;
            acc.comm_wait += p->spmv_detail.comm_wait;
            acc.shared_spmv += p->spmv_detail.shared_spmv;
            acc.send_wait += p->spmv_detail.send_wait;
        }
        printf("PROFILE_ACCUM %d spmv=%.6e vecops=%.6e send_fill=%.6e local_spmv=%.6e comm_wait=%.6e shared_spmv=%.6e send_wait=%.6e\n",
            mpi_rank, acc_spmv, acc_vecops,
            acc.send_fill, acc.local_spmv, acc.comm_wait, acc.shared_spmv, acc.send_wait);
    }
    free(iprof);   

//-------------------------------------------------------------------------------------------------------


    if (mpi_rank == 0)
    {
        Vector Xbig = vector_init(ncols);

        MPI_vector_gather(X, Xbig, A.gind, 0, MPI_COMM_WORLD);
        // Write X vector
        vector_write(arguments.output_x, Xbig);
        vector_destroy(&Xbig);
    }else{
        MPI_vector_gather(X, (Vector){}, A.gind, 0, MPI_COMM_WORLD);
    }


    // //TODO: REMOVE AFTER TESTING
    // if (mpi_rank == 0)
    // {
    //     Vector Ybig = vector_init(ncols);
    //     MPI_vector_gather(Y, Ybig, A.gind, 0, MPI_COMM_WORLD);
    //     // Write Y vector
    //     vector_write(arguments.output_x, Ybig);
    //     vector_destroy(&Ybig);
    // }else{
    //     MPI_vector_gather(Y, (Vector){}, A.gind, 0, MPI_COMM_WORLD);
    // }



//-------------------------------------------------------------------------------------------------------
    ivector_destroy(&partvec);
    vector_destroy(&X);
    vector_destroy(&B);
    SHARD_CSC_destroy(&A);

    vector_destroy(&Y);
    vector_destroy(&V);
    vector_destroy(&P);
    vector_destroy(&R);
    vector_destroy(&R_0);
    vector_destroy(&S);
    vector_destroy(&T);

    MPI_Finalize();

    return EXIT_SUCCESS;
}