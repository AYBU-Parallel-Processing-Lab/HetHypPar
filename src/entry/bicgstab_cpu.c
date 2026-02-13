#include "stdio.h"
#include <driver_types.h>
#include <omp.h>
#include <stddef.h>
#include <string.h> 
#include <stdlib.h>

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
    char *input_y;
    char *output_x;
    int n_iters;
};

// Option keys
#define OPT_INPUT_MATRIX 'm'
#define OPT_OUTPUT 'o'
#define OPT_INPUT_X 'x'
#define OPT_INPUT_Y 'y'
#define OPT_N_ITERS 'n'

// Command line options
static struct argp_option options[] = {
    {"input-matrix", OPT_INPUT_MATRIX, "FILE", 0, "Input matrix market file path"},
    {"input-x", OPT_INPUT_X, "FILE", 0, "Input X vector file path"},
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

// TODO: use GNU Octave to prepare some input output samples to test the functions
// TODO: Premake the vectors for CPU and re-implement the spmxv
// TODO: Simplify GPU spmxv
// TODO: Read an actual X vector and output Y

int main(int argc, char* argv[]) {  // matrix file ve part vector 
    
    struct arguments arguments = {};
    
    // Parse arguments
    argp_parse(&argp, argc, argv, 0, 0, &arguments);
    
    // Check if all required arguments are provided
    if (!arguments.input_matrix || !arguments.input_x || !arguments.output_x) {
        fprintf(stderr, "Error: All input and output files must be specified.\n");
        return EXIT_FAILURE;
    }
    
    // Check file existence
    if (access(arguments.input_matrix, F_OK) == -1)
    ABORT("Input matrix file '%s' does not exist\n", arguments.input_matrix)

if (access(arguments.input_x, F_OK) == -1)
        ABORT("Input vector file '%s' does not exist\n", arguments.input_x)
    
    if (access(arguments.input_y, F_OK) == -1)
    ABORT("Input vector file '%s' does not exist\n", arguments.input_y)

int niters = arguments.n_iters;

time_stamps.begin = omp_get_wtime(); // The very Beginning timestamp
//-------------------------------------------------------------------------------------------------------
    // Read the matrix market file and convert it to CSR format
    time_stamps.file_read_begin = omp_get_wtime();

    CSR A = buReadSparseMatrix(arguments.input_matrix);

    printf("Matrix name : %s\n",arguments.input_matrix);
//-------------------------------------------------------------------------------------------------------
    // Read partition vectors mapping rows to GPU(0) CPU(1)
    Vector X = vector_read(arguments.input_x, A.n);
    Vector B = vector_read(arguments.input_y, A.m);
    time_stamps.file_read_end = omp_get_wtime();

//-------------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------------

    // CPU Objects
    Vector Y = vector_init_const(A.m, 0);

    // BICGStab variables
    double rho = 1.0;
    double alpha = 1.0;
    double omega = 1.0;

    Vector V = vector_init_const(Y.nvals,0); // Add pieces for this
    Vector P = vector_init_const(Y.nvals, 0); // Add pieces for this

    Vector R = vector_init_clone(B); // Add pieces for this
    CSR_spmxv_seq(A, X, Y);
    vector_sub_seq(R, Y, R);
    vector_zero(Y);

    Vector R_0 = vector_init_clone(R);
    Vector S = vector_init(Y.nvals);
    Vector T = vector_init(Y.nvals);

//-------------------------------------------------------------------------------------------------------
    time_stamps.gpu_transfer_begin = omp_get_wtime();


    time_stamps.gpu_transfer_end = omp_get_wtime();
//-------------------------------------------------------------------------------------------------------
    // double times[4] = {};
    time_stamps.spmxv_begin = omp_get_wtime();
    for (size_t i = 0; i < niters; i++)
    {
        // calc rho_n+1
        double temp_rho = vector_dot_seq(R, R_0);
        //==============
        // calc Beta
        double beta = (temp_rho / rho) * (alpha / omega);
        rho = temp_rho; // old rho is never used again after here
        //==============
        // calc P_n+1
        vector_scale_seq(V, omega, V); // Changed V
        vector_sub_seq(P, V, P);
        vector_scale_seq(P, beta, P);
        vector_add_seq(P, R, P);
        //==============
        // calc V_n+1
        CSR_spmxv_seq(A, P, V); // result in V
        //==============
        // calc alpha_n+1
        alpha = rho/vector_dot_seq(R_0, V);
        //==============
        // calc S
        vector_scale_seq(V, alpha, S);
        vector_sub_seq(R, S, S);
        //==============
        // calc T
        CSR_spmxv_seq(A, S, T);
        //==============
        // calc omega_n+1
        omega = vector_dot_seq(T, S)/vector_dot_seq(T, T);
        //==============
        // calc X_n+1
        vector_scale_seq(P, alpha, Y);
        vector_add_seq(X, Y, X);
        vector_scale_seq(S, omega, Y);
        vector_add_seq(X, Y, X);
        //==============
        // calc R_n+1
        vector_scale_seq(T, omega, R);
        vector_sub_seq(S, R, R);
        //==============
        // calc tol
        double tol = vector_dot_seq(S, S);
        //==============
    }
    time_stamps.spmxv_end = omp_get_wtime();
    time_stamps.end = time_stamps.spmxv_end;
    
    //-------------------------------------------------------------------------------------------------------
    // Calculate Relavtive residual (|| Ax - b || / || b ||)
    
    CSR_spmxv_seq(A, X, Y);
    vector_sub_seq(B, Y, Y);
    double sy = vector_dot_seq(Y, Y);
    double sb = vector_dot_seq(B, B);
    double relative_residual = sqrt(sy/sb);
    
    
    //-------------------------------------------------------------------------------------------------------


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
   

//-------------------------------------------------------------------------------------------------------
    // Write X vector
    vector_write(arguments.output_x, X);

//-------------------------------------------------------------------------------------------------------
    freeSparseMatrix(&A) ;
    vector_destroy(&X);
    vector_destroy(&Y);
    vector_destroy(&B);
    vector_destroy(&P);
    vector_destroy(&S);
    vector_destroy(&V);
    vector_destroy(&R);
    vector_destroy(&R_0);
    vector_destroy(&T);

    return EXIT_SUCCESS;
}