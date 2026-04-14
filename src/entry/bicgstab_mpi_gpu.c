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
    char *input_gpu;
    char *input_y;
    char *output_x;
    int n_iters;
};

// Option keys
#define OPT_INPUT_MATRIX 'm'
#define OPT_OUTPUT 'o'
#define OPT_INPUT_X 'x'
#define OPT_INPUT_PART 'p'
#define OPT_INPUT_GPU 'g'
#define OPT_INPUT_Y 'y'
#define OPT_N_ITERS 'n'

// Command line options
static struct argp_option options[] = {
    {"input-matrix", OPT_INPUT_MATRIX, "FILE", 0, "Input matrix market file path"},
    {"input-x", OPT_INPUT_X, "FILE", 0, "Input X vector file path"},
    {"input-part", OPT_INPUT_PART, "FILE", 0, "Input Partition vector file path"},
    {"input-gpu", OPT_INPUT_GPU, "FILE", 0, "Input Binary vector indicating whether or not this rank uses a GPU"},
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
        case OPT_INPUT_GPU:
            arguments->input_gpu = arg;
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


int main(int argc, char* argv[]) {  // matrix file ve part vector 
    MPI_Init(&argc, &argv);

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);


    struct arguments arguments = {};
    
    // Parse arguments
    argp_parse(&argp, argc, argv, 0, 0, &arguments);
    
    // Check if all required arguments are provided
    if (!arguments.input_matrix || !arguments.input_x || !arguments.input_gpu || !arguments.input_part || !arguments.output_x) {
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
            ABORT("Input partition vector file '%s' does not exist\n", arguments.input_part)

        if (access(arguments.input_gpu, F_OK) == -1)
            ABORT("Input gpu vector file '%s' does not exist\n", arguments.input_gpu)
        
        if (access(arguments.input_y, F_OK) == -1)
            ABORT("Input vector file '%s' does not exist\n", arguments.input_y)
    }

    int niters = arguments.n_iters;

    time_stamps.begin = omp_get_wtime(); // The very Beginning timestamp
    
//-------------------------------------------------------------------------------------------------------    

    int isgpu;

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

        // Read the boolean vector that indicates whether or not a specified mpi rank is using GPU OR CPU
        isgpu = MPI_ivector_read_scatter(arguments.input_gpu, mpi_size);

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
    
    double relative_residual;
    
    if (isgpu){ // GPU PROCESS
        //-------------------------------------------------------------------------------------------------------
        CHECK_CUDA(cudaSetDevice(0))
        
        cublasHandle_t  cublasHandle   = NULL;
        CHECK_CUBLAS( cublasCreate(&cublasHandle) )
        cusparseHandle_t cusparseHandle = NULL;
        CHECK_CUSPARSE( cusparseCreate(&cusparseHandle) )
        //-------------------------------------------------------------------------------------------------------
        
        const double mx_alpha = 1.00;
        const double mx_neg = -1.00;
        const double mx_beta = 0.00;
        
        // Doesn't need to be freed
        Device_SHARD_CSC dA = {
            .gind = A.gind,
            .recv = A.recv,
            .send = A.send
        };
        
        CHECK_CUSPARSE(device_csc_create(A.loc, &dA.loc))
        CHECK_CUSPARSE(device_csc_create(A.shr, &dA.shr))
        
        // // GET RID OF CPU matrices right away to avoid duplication. Edit: Keep them for hybrid methods (GPU local and CPU shared)
        // freeSparseMatrix(&A.loc);
        // freeSparseMatrix(&A.shr);
        
        Device_Vector dX;
        Device_Vector dX_shr;
        Device_Vector dB;
        CHECK_CUSPARSE(device_vector_init(X.nvals, &dX))
        CHECK_CUSPARSE(device_vector_init(dA.shr.data.n, &dX_shr))
        CHECK_CUSPARSE(device_vector_init(B.nvals, &dB))
        
        Device_Buffer_SpMV dA_loc_buf;
        Device_Buffer_SpMV dA_shr_buf;
        
        CHECK_CUSPARSE(device_buffer_spmv_create(cusparseHandle, dA.loc.desc, dX, dB, &mx_alpha, &mx_beta, &dA_loc_buf))
        CHECK_CUSPARSE(device_buffer_spmv_create(cusparseHandle, dA.shr.desc, dX_shr, dB, &mx_alpha, &mx_alpha, &dA_shr_buf))
        
        CHECK_CUSPARSE(device_vector_toGPU(X, dX))
        CHECK_CUSPARSE(device_vector_toGPU(B, dB))
        //-------------------------------------------------------------------------------------------------------
        
        // CPU Objects
        // [ ] IS FREED?
        Device_Vector Y; // = vector_init_const(A.loc.m, 0);
        CHECK_CUSPARSE(device_vector_init(dA.loc.data.m, &Y))
        CHECK_CUSPARSE(device_vector_zero(Y))
        
        // BICGStab variables
        double rho = 1.0;
        double alpha = 1.0;
        double omega = 1.0;
        // [ ] IS FREED?
        Device_Vector V; // = vector_init_const(Y.nvals,0); // Add pieces for this
        CHECK_CUSPARSE(device_vector_init(Y.nvals, &V))
        CHECK_CUSPARSE(device_vector_zero(V))
        // [ ] IS FREED?
        Device_Vector P; // = vector_init_const(Y.nvals, 0); // Add pieces for this
        CHECK_CUSPARSE(device_vector_init(Y.nvals, &P))
        CHECK_CUSPARSE(device_vector_zero(P))
        
        // [ ] IS FREED?
        Device_Vector R; // = vector_init_clone(B); // Add pieces for this
        CHECK_CUSPARSE(device_vector_init(B.nvals, &R))
        CHECK_CUDA(device_vector_GPUtoGPU(dB, R))
        
        // TODO: IMPLEMENT SpMxV
        // MPI_SHARD_CSC_mpi_spmxv_seq(A, X, Y, MPI_COMM_WORLD);
        CHECK_CUSPARSE(MPI_device_SHARD_CSC_mpi_spmxv(dA, X, dX, dX_shr, Y, MPI_COMM_WORLD, cusparseHandle, mx_alpha, mx_beta, dA_loc_buf, dA_shr_buf))
        CHECK_CUBLAS(device_vector_axpy(cublasHandle, Y, -1.0, R))
        CHECK_CUSPARSE(device_vector_zero(Y))
        // vector_sub_seq(R, Y, R);
        // vector_zero(Y);
        
        // [ ] IS FREED?
        Device_Vector R_0; // = vector_init_clone(R);
        CHECK_CUSPARSE(device_vector_init(R.nvals, &R_0))
        CHECK_CUDA(device_vector_GPUtoGPU(R, R_0))
        // [ ] IS FREED?
        Device_Vector S; // = vector_init(Y.nvals);
        CHECK_CUSPARSE(device_vector_init(Y.nvals, &S))
        // [ ] IS FREED?
        Device_Vector T; // = vector_init(Y.nvals);
        CHECK_CUSPARSE(device_vector_init(Y.nvals, &T))

        if (mpi_rank == 0)
            printf("LOG: Finished Creating Vectors\n");
        
        
        //-------------------------------------------------------------------------------------------------------

                // double times[4] = {};
        time_stamps.spmxv_begin = omp_get_wtime();

        for (size_t i = 0; i < niters; i++)
        {
            // calc rho_n+1
            double temp_rho; // = MPI_vector_dot(R, R_0, MPI_COMM_WORLD);
            CHECK_CUBLAS(MPI_device_vector_dot(cublasHandle, R, R_0, &temp_rho, MPI_COMM_WORLD))
            //==============
            // calc Beta
            double beta = (temp_rho / rho) * (alpha / omega);
            rho = temp_rho; // old rho is never used again after here
            //==============
            // calc P_n+1
            // vector_scale_seq(V, omega, V); // Changed V
            CHECK_CUBLAS(device_vector_scale(cublasHandle, omega, V))
            // vector_sub_seq(P, V, P);
            CHECK_CUBLAS(device_vector_axpy(cublasHandle, V, mx_neg, P))
            // vector_scale_seq(P, beta, P);
            CHECK_CUBLAS(device_vector_scale(cublasHandle, beta, P))
            // vector_add_seq(P, R, P);
            CHECK_CUBLAS(device_vector_axpy(cublasHandle, R, mx_alpha, P))
            //==============
            // calc V_n+1
            // CHECK_CUSPARSE(device_csr_spmv(cusparseHandle, dA, P, V, mx_alpha, mx_beta, dA_buf))
            CHECK_CUSPARSE(MPI_device_SHARD_CSC_mpi_spmxv(dA, X, P, dX_shr, V, MPI_COMM_WORLD, cusparseHandle, mx_alpha, mx_beta, dA_loc_buf, dA_shr_buf))
            
            //==============
            // calc alpha_n+1
            double temp_rv;
            // CHECK_CUBLAS(device_vector_dot(cublasHandle, R_0, V, &temp_rv))
            CHECK_CUBLAS(MPI_device_vector_dot(cublasHandle, R_0, V, &temp_rv, MPI_COMM_WORLD))
            alpha = rho/temp_rv; //vector_dot_seq(R_0, V);
            //==============
            // calc S
            // vector_scale_seq(V, alpha, S);
            CHECK_CUDA(device_vector_GPUtoGPU(R, S))
            // vector_sub_seq(R, S, S);
            CHECK_CUBLAS(device_vector_axpy(cublasHandle, V, -alpha, S))
            //==============
            // calc T
            // CSR_spmxv_seq(A, S, T);
            // CHECK_CUSPARSE(device_csr_spmv(cusparseHandle, dA, S, T, mx_alpha, mx_beta, dA_buf))
            CHECK_CUSPARSE(MPI_device_SHARD_CSC_mpi_spmxv(dA, X, S, dX_shr, T, MPI_COMM_WORLD, cusparseHandle, mx_alpha, mx_beta, dA_loc_buf, dA_shr_buf))
            //==============
            // calc omega_n+1
            double temp_ts;
            // CHECK_CUBLAS(device_vector_dot(cublasHandle, T, S, &temp_ts))
            CHECK_CUBLAS(MPI_device_vector_dot(cublasHandle, T, S, &temp_ts, MPI_COMM_WORLD))
            double temp_tt;
            // CHECK_CUBLAS(device_vector_dot(cublasHandle, T, T, &temp_tt))
            CHECK_CUBLAS(MPI_device_vector_dot(cublasHandle, T, T, &temp_tt, MPI_COMM_WORLD))
            // omega = vector_dot_seq(T, S)/vector_dot_seq(T, T);
            omega = temp_ts/temp_tt;
            
            //==============
            // calc X_n+1
            // vector_scale_seq(P, alpha, Y);
            // vector_add_seq(X, Y, X);
            // vector_scale_seq(S, omega, Y);
            // vector_add_seq(X, Y, X);
            CHECK_CUBLAS(device_vector_axpy(cublasHandle, P, alpha, dX))
            CHECK_CUBLAS(device_vector_axpy(cublasHandle, S, omega, dX))
            //==============
            // calc R_n+1
            // vector_scale_seq(T, omega, R);
            // vector_sub_seq(S, R, R);
            CHECK_CUDA(device_vector_GPUtoGPU(S, R))
            CHECK_CUBLAS(device_vector_axpy(cublasHandle, T, -omega, R))
            //==============
            // calc tol
            double tol;// = vector_dot_seq(S, S);
            // CHECK_CUBLAS(device_vector_dot(cublasHandle, S, S, &tol))
            CHECK_CUBLAS(MPI_device_vector_dot(cublasHandle, S, S, &tol, MPI_COMM_WORLD))
        }

        // // TODO: REMOVE THIS AFTER TESTING
        // MPI_SHARD_CSC_mpi_spmxv_seq(A, X, Y, MPI_COMM_WORLD);
        
        time_stamps.spmxv_end = omp_get_wtime();
        time_stamps.end = time_stamps.spmxv_end;

        CHECK_CUSPARSE(device_vector_toCPU(dX, X))

        //-------------------------------------------------------------------------------------------------------

        CHECK_CUSPARSE(device_csc_destroy(&dA.loc))
        CHECK_CUSPARSE(device_csc_destroy(&dA.shr))

        CHECK_CUBLAS(cublasDestroy(cublasHandle))
        CHECK_CUSPARSE(cusparseDestroy(cusparseHandle))

    }else{ // CPU PROCESS

        
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
        
        // TODO: IMPLEMENT SpMxV
        MPI_SHARD_CSC_mpi_spmxv_seq(A, X, Y, MPI_COMM_WORLD);
        vector_sub_seq(R, Y, R);
        vector_zero(Y);
        
        // [x] IS FREED?
        Vector R_0 = vector_init_clone(R);
        // [x] IS FREED?
        Vector S = vector_init(Y.nvals);
        // [x] IS FREED?
        Vector T = vector_init(Y.nvals);

        if (mpi_rank == 0)
            printf("LOG: Finished Creating Vectors\n");

    //-------------------------------------------------------------------------------------------------------
        // double times[4] = {};
        time_stamps.spmxv_begin = omp_get_wtime();

        for (size_t i = 0; i < niters; i++)
        {
            // calc rho_n+1
            double temp_rho = MPI_vector_dot(R, R_0, MPI_COMM_WORLD);
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
            MPI_SHARD_CSC_mpi_spmxv_seq(A, P, V, MPI_COMM_WORLD);  // result in V
            //==============
            // calc alpha_n+1
            alpha = rho/MPI_vector_dot(R_0, V, MPI_COMM_WORLD);
            //==============
            // calc S
            vector_scale_seq(V, alpha, S);
            vector_sub_seq(R, S, S);
            //==============
            // calc T
            MPI_SHARD_CSC_mpi_spmxv_seq(A, S, T, MPI_COMM_WORLD);
            //==============
            // calc omega_n+1
            omega = MPI_vector_dot(T, S,MPI_COMM_WORLD)/MPI_vector_dot(T, T,MPI_COMM_WORLD);
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
            double tol = MPI_vector_dot(S, S,MPI_COMM_WORLD);
            //==============
        }

        // // TODO: REMOVE THIS AFTER TESTING
        // MPI_SHARD_CSC_mpi_spmxv_seq(A, X, Y, MPI_COMM_WORLD);
        
        time_stamps.spmxv_end = omp_get_wtime();
        time_stamps.end = time_stamps.spmxv_end;
        
        //-------------------------------------------------------------------------------------------------------
        
        vector_destroy(&Y);
        vector_destroy(&V);
        vector_destroy(&P);
        vector_destroy(&R);
        vector_destroy(&R_0);
        vector_destroy(&S);
        vector_destroy(&T);
    }
    //-------------------------------------------------------------------------------------------------------

    // Calculate Relavtive residual (|| Ax - b || / || b ||)
    
    Vector tempVec = vector_init(B.nvals);

    MPI_SHARD_CSC_mpi_spmxv_seq(A, X, tempVec, MPI_COMM_WORLD);
    vector_sub_seq(B, tempVec, tempVec);
    double sy = MPI_vector_dot(tempVec, tempVec, MPI_COMM_WORLD);
    double sb = MPI_vector_dot(B, B, MPI_COMM_WORLD);
    relative_residual = sqrt(sy/sb);

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

    ivector_destroy(&partvec);
    vector_destroy(&X);
    vector_destroy(&B);
    SHARD_CSC_destroy(&A);


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

    MPI_Finalize();

    return EXIT_SUCCESS;
}