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
    Iter_Profile *iprof = calloc(niters, sizeof(Iter_Profile));

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
        int has_shr = (A.shr.n > 0);
        if (has_shr)
            CHECK_CUSPARSE(device_csc_create(A.shr, &dA.shr))

        // // GET RID OF CPU matrices right away to avoid duplication. Edit: Keep them for hybrid methods (GPU local and CPU shared)
        // freeSparseMatrix(&A.loc);
        // freeSparseMatrix(&A.shr);

        Device_Vector dX;
        Device_Vector dX_shr = {0};
        Device_Vector dB;
        CHECK_CUSPARSE(device_vector_init(X.nvals, &dX))
        if (has_shr)
            CHECK_CUSPARSE(device_vector_init(dA.shr.data.n, &dX_shr))
        CHECK_CUSPARSE(device_vector_init(B.nvals, &dB))

        Device_Buffer_SpMV dA_loc_buf;
        Device_Buffer_SpMV dA_shr_buf = NULL;

        CHECK_CUSPARSE(device_buffer_spmv_create(cusparseHandle, dA.loc.desc, dX, dB, &mx_alpha, &mx_beta, &dA_loc_buf))
        if (has_shr)
            CHECK_CUSPARSE(device_buffer_spmv_create(cusparseHandle, dA.shr.desc, dX_shr, dB, &mx_alpha, &mx_alpha, &dA_shr_buf))
        
        CHECK_CUSPARSE(device_vector_toGPU(X, dX))
        CHECK_CUSPARSE(device_vector_toGPU(B, dB))
        //-------------------------------------------------------------------------------------------------------
        
        // CPU Objects
        
        // GPU Objects for p-BiCGStab
        Device_Vector Y;
        CHECK_CUSPARSE(device_vector_init(B.nvals, &Y))
        Device_Vector R;
        CHECK_CUSPARSE(device_vector_init(B.nvals, &R))
        CHECK_CUDA(device_vector_GPUtoGPU(dB, R))
        
        CHECK_CUSPARSE(MPI_device_SHARD_CSC_mpi_spmxv(dA, X, dX, dX_shr, Y, MPI_COMM_WORLD, cusparseHandle, mx_alpha, mx_beta, dA_loc_buf, dA_shr_buf, NULL))
        CHECK_CUBLAS(device_vector_axpy(cublasHandle, Y, -1.0, R)) // r0 = b - Ax0
        CHECK_CUSPARSE(device_vector_zero(Y))

        Device_Vector R_0;
        CHECK_CUSPARSE(device_vector_init(R.nvals, &R_0))
        CHECK_CUDA(device_vector_GPUtoGPU(R, R_0))

        Device_Vector W;
        CHECK_CUSPARSE(device_vector_init(B.nvals, &W))
        CHECK_CUSPARSE(MPI_device_SHARD_CSC_mpi_spmxv(dA, X, R, dX_shr, W, MPI_COMM_WORLD, cusparseHandle, mx_alpha, mx_beta, dA_loc_buf, dA_shr_buf, NULL)) // w0 = A * r0
        
        Device_Vector T;
        CHECK_CUSPARSE(device_vector_init(B.nvals, &T))
        CHECK_CUSPARSE(MPI_device_SHARD_CSC_mpi_spmxv(dA, X, W, dX_shr, T, MPI_COMM_WORLD, cusparseHandle, mx_alpha, mx_beta, dA_loc_buf, dA_shr_buf, NULL)) // t0 = A * w0

        double r0_dot_r;
        CHECK_CUBLAS(MPI_device_vector_dot(cublasHandle, R_0, R, &r0_dot_r, MPI_COMM_WORLD))
        double r0_dot_w;
        CHECK_CUBLAS(MPI_device_vector_dot(cublasHandle, R_0, W, &r0_dot_w, MPI_COMM_WORLD))

        double alpha = r0_dot_r / r0_dot_w;
        double beta = 0.0;
        double omega = 0.0;
        
        Device_Vector P, P_prev, S, S_prev, Z, Z_prev, V, V_prev, Q;
        Device_Vector R_next, W_next, T_next;
        CHECK_CUSPARSE(device_vector_init(B.nvals, &P))
        CHECK_CUSPARSE(device_vector_zero(P))
        CHECK_CUSPARSE(device_vector_init(B.nvals, &P_prev))
        CHECK_CUSPARSE(device_vector_zero(P_prev))
        CHECK_CUSPARSE(device_vector_init(B.nvals, &S))
        CHECK_CUSPARSE(device_vector_zero(S))
        CHECK_CUSPARSE(device_vector_init(B.nvals, &S_prev))
        CHECK_CUSPARSE(device_vector_zero(S_prev))
        CHECK_CUSPARSE(device_vector_init(B.nvals, &Z))
        CHECK_CUSPARSE(device_vector_zero(Z))
        CHECK_CUSPARSE(device_vector_init(B.nvals, &Z_prev))
        CHECK_CUSPARSE(device_vector_zero(Z_prev))
        CHECK_CUSPARSE(device_vector_init(B.nvals, &V))
        CHECK_CUSPARSE(device_vector_zero(V))
        CHECK_CUSPARSE(device_vector_init(B.nvals, &V_prev))
        CHECK_CUSPARSE(device_vector_zero(V_prev))
        CHECK_CUSPARSE(device_vector_init(B.nvals, &Q))
        CHECK_CUSPARSE(device_vector_zero(Q))
        CHECK_CUSPARSE(device_vector_init(B.nvals, &R_next))
        CHECK_CUSPARSE(device_vector_init(B.nvals, &W_next))
        CHECK_CUSPARSE(device_vector_init(B.nvals, &T_next))

        if (mpi_rank == 0)
            printf("LOG: Finished Creating Vectors\n");

        //-------------------------------------------------------------------------------------------------------

        time_stamps.spmxv_begin = omp_get_wtime();

        for (size_t i = 0; i < niters; i++)
        {
            SpMV_Profile sp = {0};
            double t_vec0 = omp_get_wtime();

            // Residual replacement every 50 iterations
            if (i > 0 && i % 50 == 0) {
                CHECK_CUSPARSE(MPI_device_SHARD_CSC_mpi_spmxv(dA, X, dX, dX_shr, Y, MPI_COMM_WORLD, cusparseHandle, mx_alpha, mx_beta, dA_loc_buf, dA_shr_buf, NULL))
                CHECK_CUDA(device_vector_GPUtoGPU(dB, R))
                CHECK_CUBLAS(device_vector_axpy(cublasHandle, Y, -1.0, R)) // r = b - Ax
                CHECK_CUSPARSE(device_vector_zero(Y))
                CHECK_CUSPARSE(MPI_device_SHARD_CSC_mpi_spmxv(dA, X, R, dX_shr, W, MPI_COMM_WORLD, cusparseHandle, mx_alpha, mx_beta, dA_loc_buf, dA_shr_buf, NULL))
                CHECK_CUSPARSE(MPI_device_SHARD_CSC_mpi_spmxv(dA, X, W, dX_shr, T, MPI_COMM_WORLD, cusparseHandle, mx_alpha, mx_beta, dA_loc_buf, dA_shr_buf, NULL))
            }

            // p = r + beta * (p_prev - omega * s_prev)
            CHECK_CUDA(device_vector_GPUtoGPU(P_prev, P))
            CHECK_CUBLAS(device_vector_axpy(cublasHandle, S_prev, -omega, P))
            CHECK_CUBLAS(device_vector_scale(cublasHandle, beta, P))
            CHECK_CUBLAS(device_vector_axpy(cublasHandle, R, 1.0, P))
            
            // s = w + beta * (s_prev - omega * z_prev)
            CHECK_CUDA(device_vector_GPUtoGPU(S_prev, S))
            CHECK_CUBLAS(device_vector_axpy(cublasHandle, Z_prev, -omega, S))
            CHECK_CUBLAS(device_vector_scale(cublasHandle, beta, S))
            CHECK_CUBLAS(device_vector_axpy(cublasHandle, W, 1.0, S))

            // z = t + beta * (z_prev - omega * v_prev)
            CHECK_CUDA(device_vector_GPUtoGPU(Z_prev, Z))
            CHECK_CUBLAS(device_vector_axpy(cublasHandle, V_prev, -omega, Z))
            CHECK_CUBLAS(device_vector_scale(cublasHandle, beta, Z))
            CHECK_CUBLAS(device_vector_axpy(cublasHandle, T, 1.0, Z))

            // q = r - alpha * s
            CHECK_CUDA(device_vector_GPUtoGPU(R, Q))
            CHECK_CUBLAS(device_vector_axpy(cublasHandle, S, -alpha, Q))

            // y = w - alpha * z
            CHECK_CUDA(device_vector_GPUtoGPU(W, Y))
            CHECK_CUBLAS(device_vector_axpy(cublasHandle, Z, -alpha, Y))

            // Global reduction 1
            double local_dots1[2];
            CHECK_CUBLAS(device_vector_dot(cublasHandle, Q, Y, &local_dots1[0]))
            CHECK_CUBLAS(device_vector_dot(cublasHandle, Y, Y, &local_dots1[1]))
            double global_dots1[2];
            
            MPI_Request req1;
            MPI_Iallreduce(local_dots1, global_dots1, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &req1);

            // Overlap: v = A * z
            cudaDeviceSynchronize();
            double t_spmv0 = omp_get_wtime();
            CHECK_CUSPARSE(MPI_device_SHARD_CSC_mpi_spmxv(dA, X, Z, dX_shr, V, MPI_COMM_WORLD, cusparseHandle, mx_alpha, mx_beta, dA_loc_buf, dA_shr_buf, &sp))
            double t_spmv1 = omp_get_wtime();

            MPI_Wait(&req1, MPI_STATUS_IGNORE);
            omega = global_dots1[0] / global_dots1[1];

            // x = x + alpha * p + omega * q
            CHECK_CUBLAS(device_vector_axpy(cublasHandle, P, alpha, dX))
            CHECK_CUBLAS(device_vector_axpy(cublasHandle, Q, omega, dX))

            // r_next = q - omega * y
            CHECK_CUDA(device_vector_GPUtoGPU(Q, R_next))
            CHECK_CUBLAS(device_vector_axpy(cublasHandle, Y, -omega, R_next))

            // w_next = y - omega * (t - alpha * v)
            CHECK_CUDA(device_vector_GPUtoGPU(T, W_next))
            CHECK_CUBLAS(device_vector_axpy(cublasHandle, V, -alpha, W_next))
            CHECK_CUBLAS(device_vector_scale(cublasHandle, -omega, W_next))
            CHECK_CUBLAS(device_vector_axpy(cublasHandle, Y, 1.0, W_next))

            // Global reduction 2
            double local_dots2[4];
            CHECK_CUBLAS(device_vector_dot(cublasHandle, R_0, R_next, &local_dots2[0]))
            CHECK_CUBLAS(device_vector_dot(cublasHandle, R_0, W_next, &local_dots2[1]))
            CHECK_CUBLAS(device_vector_dot(cublasHandle, R_0, S, &local_dots2[2]))
            CHECK_CUBLAS(device_vector_dot(cublasHandle, R_0, Z, &local_dots2[3]))
            double global_dots2[4];
            
            MPI_Request req2;
            MPI_Iallreduce(local_dots2, global_dots2, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &req2);

            // Overlap: t_next = A * w_next
            cudaDeviceSynchronize();
            double t_spmv2 = omp_get_wtime();
            CHECK_CUSPARSE(MPI_device_SHARD_CSC_mpi_spmxv(dA, X, W_next, dX_shr, T_next, MPI_COMM_WORLD, cusparseHandle, mx_alpha, mx_beta, dA_loc_buf, dA_shr_buf, &sp))
            double t_spmv3 = omp_get_wtime();

            MPI_Wait(&req2, MPI_STATUS_IGNORE);

            double r0_dot_r_next = global_dots2[0];
            double r0_dot_w_next = global_dots2[1];
            double r0_dot_s = global_dots2[2];
            double r0_dot_z = global_dots2[3];
            
            beta = (alpha / omega) * (r0_dot_r_next / r0_dot_r);
            alpha = r0_dot_r_next / (r0_dot_w_next + beta * r0_dot_s - beta * omega * r0_dot_z);
            r0_dot_r = r0_dot_r_next;

            // Rotate
            CHECK_CUDA(device_vector_GPUtoGPU(P, P_prev))
            CHECK_CUDA(device_vector_GPUtoGPU(S, S_prev))
            CHECK_CUDA(device_vector_GPUtoGPU(Z, Z_prev))
            CHECK_CUDA(device_vector_GPUtoGPU(V, V_prev))
            
            CHECK_CUDA(device_vector_GPUtoGPU(R_next, R))
            CHECK_CUDA(device_vector_GPUtoGPU(W_next, W))
            CHECK_CUDA(device_vector_GPUtoGPU(T_next, T))

            double t_vec1 = omp_get_wtime();
            double spmv_total = (t_spmv1 - t_spmv0) + (t_spmv3 - t_spmv2);
            iprof[i].spmv = spmv_total;
            iprof[i].vector_ops = (t_vec1 - t_vec0) - spmv_total;
            iprof[i].spmv_detail = sp;
        }

        time_stamps.spmxv_end = omp_get_wtime();
        time_stamps.end = time_stamps.spmxv_end;

        CHECK_CUSPARSE(device_vector_toCPU(dX, X))

        //-------------------------------------------------------------------------------------------------------

        CHECK_CUSPARSE(device_vector_destroy(&P))
        CHECK_CUSPARSE(device_vector_destroy(&P_prev))
        CHECK_CUSPARSE(device_vector_destroy(&S))
        CHECK_CUSPARSE(device_vector_destroy(&S_prev))
        CHECK_CUSPARSE(device_vector_destroy(&Z))
        CHECK_CUSPARSE(device_vector_destroy(&Z_prev))
        CHECK_CUSPARSE(device_vector_destroy(&V))
        CHECK_CUSPARSE(device_vector_destroy(&V_prev))
        CHECK_CUSPARSE(device_vector_destroy(&Q))
        CHECK_CUSPARSE(device_vector_destroy(&R_next))
        CHECK_CUSPARSE(device_vector_destroy(&W_next))
        CHECK_CUSPARSE(device_vector_destroy(&T_next))
        CHECK_CUSPARSE(device_vector_destroy(&R))
        CHECK_CUSPARSE(device_vector_destroy(&R_0))
        CHECK_CUSPARSE(device_vector_destroy(&W))
        CHECK_CUSPARSE(device_vector_destroy(&T))

        CHECK_CUSPARSE(device_csc_destroy(&dA.loc))
        if (has_shr)
            CHECK_CUSPARSE(device_csc_destroy(&dA.shr))

        CHECK_CUBLAS(cublasDestroy(cublasHandle))
        CHECK_CUSPARSE(cusparseDestroy(cusparseHandle))

    }else{ // CPU PROCESS

        // CPU Objects for p-BiCGStab
        Vector R = vector_init_clone(B); 
        Vector Y = vector_init_const(A.loc.m, 0);
        
        MPI_SHARD_CSC_mpi_spmxv_seq(A, X, Y, MPI_COMM_WORLD, NULL);
        vector_sub_seq(R, Y, R); // r0 = b - A*x0
        vector_zero(Y);

        Vector R_0 = vector_init_clone(R);
        
        Vector W = vector_init(B.nvals);
        MPI_SHARD_CSC_mpi_spmxv_seq(A, R, W, MPI_COMM_WORLD, NULL); // w0 = A * r0
        
        Vector T = vector_init(B.nvals);
        MPI_SHARD_CSC_mpi_spmxv_seq(A, W, T, MPI_COMM_WORLD, NULL); // t0 = A * w0
        
        double r0_dot_r = MPI_vector_dot(R_0, R, MPI_COMM_WORLD);
        double r0_dot_w = MPI_vector_dot(R_0, W, MPI_COMM_WORLD);
        
        double alpha = r0_dot_r / r0_dot_w;
        double beta = 0.0;
        double omega = 0.0;
        
        Vector P = vector_init_const(B.nvals, 0);
        Vector P_prev = vector_init_const(B.nvals, 0);
        Vector S = vector_init_const(B.nvals, 0);
        Vector S_prev = vector_init_const(B.nvals, 0);
        Vector Z = vector_init_const(B.nvals, 0);
        Vector Z_prev = vector_init_const(B.nvals, 0);
        Vector V = vector_init_const(B.nvals, 0);
        Vector V_prev = vector_init_const(B.nvals, 0);
        Vector Q = vector_init_const(B.nvals, 0);
        
        Vector R_next = vector_init(B.nvals);
        Vector W_next = vector_init(B.nvals);
        Vector T_next = vector_init(B.nvals);

        if (mpi_rank == 0)
            printf("LOG: Finished Creating Vectors for p-BiCGStab (CPU)\n");

        time_stamps.spmxv_begin = omp_get_wtime();

        for (size_t i = 0; i < niters; i++)
        {
            SpMV_Profile sp = {0};
            double t_vec0 = omp_get_wtime();

            // Residual replacement every 50 iterations
            if (i > 0 && i % 50 == 0) {
                MPI_SHARD_CSC_mpi_spmxv_seq(A, X, Y, MPI_COMM_WORLD, NULL);
                vector_sub_seq(B, Y, R); // r = b - Ax
                MPI_SHARD_CSC_mpi_spmxv_seq(A, R, W, MPI_COMM_WORLD, NULL); // w = A*r
                MPI_SHARD_CSC_mpi_spmxv_seq(A, W, T, MPI_COMM_WORLD, NULL); // t = A*w
            }

            // FUSED LOOP 1: Calculate P, S, Z, Q, Y in one pass
            #pragma omp parallel for
            for(size_t k = 0; k < B.nvals; k++) {
                P.vals[k] = R.vals[k] + beta * (P_prev.vals[k] - omega * S_prev.vals[k]);
                S.vals[k] = W.vals[k] + beta * (S_prev.vals[k] - omega * Z_prev.vals[k]);
                Z.vals[k] = T.vals[k] + beta * (Z_prev.vals[k] - omega * V_prev.vals[k]);
                Q.vals[k] = R.vals[k] - alpha * S.vals[k];
                Y.vals[k] = W.vals[k] - alpha * Z.vals[k];
            }

            // Global reduction 1
            double local_dots1[2];
            local_dots1[0] = vector_dot_seq(Q, Y);
            local_dots1[1] = vector_dot_seq(Y, Y);
            double global_dots1[2];
            
            MPI_Request req1;
            MPI_Iallreduce(local_dots1, global_dots1, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &req1);
            
            // Overlap: v = A * z
            double t_spmv0 = omp_get_wtime();
            MPI_SHARD_CSC_mpi_spmxv_seq(A, Z, V, MPI_COMM_WORLD, &sp);
            double t_spmv1 = omp_get_wtime();
            
            MPI_Wait(&req1, MPI_STATUS_IGNORE);
            omega = global_dots1[0] / global_dots1[1];
            
            // FUSED LOOP 2: Calculate X, R_next, W_next in one pass
            #pragma omp parallel for
            for(size_t k = 0; k < B.nvals; k++) {
                X.vals[k] = X.vals[k] + alpha * P.vals[k] + omega * Q.vals[k];
                R_next.vals[k] = Q.vals[k] - omega * Y.vals[k];
                W_next.vals[k] = Y.vals[k] - omega * (T.vals[k] - alpha * V.vals[k]);
            }

            // Global reduction 2
            double local_dots2[4];
            local_dots2[0] = vector_dot_seq(R_0, R_next);
            local_dots2[1] = vector_dot_seq(R_0, W_next);
            local_dots2[2] = vector_dot_seq(R_0, S);
            local_dots2[3] = vector_dot_seq(R_0, Z);
            double global_dots2[4];
            
            MPI_Request req2;
            MPI_Iallreduce(local_dots2, global_dots2, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &req2);
            
            // Overlap: t_next = A * w_next
            double t_spmv2 = omp_get_wtime();
            MPI_SHARD_CSC_mpi_spmxv_seq(A, W_next, T_next, MPI_COMM_WORLD, &sp);
            double t_spmv3 = omp_get_wtime();
            
            MPI_Wait(&req2, MPI_STATUS_IGNORE);
            
            double r0_dot_r_next = global_dots2[0];
            double r0_dot_w_next = global_dots2[1];
            double r0_dot_s = global_dots2[2];
            double r0_dot_z = global_dots2[3];
            
            beta = (alpha / omega) * (r0_dot_r_next / r0_dot_r);
            alpha = r0_dot_r_next / (r0_dot_w_next + beta * r0_dot_s - beta * omega * r0_dot_z);
            r0_dot_r = r0_dot_r_next;
            
            // Rotate vectors using zero-copy pointer swaps
            double* tmp_ptr;
            tmp_ptr = P_prev.vals; P_prev.vals = P.vals; P.vals = tmp_ptr;
            tmp_ptr = S_prev.vals; S_prev.vals = S.vals; S.vals = tmp_ptr;
            tmp_ptr = Z_prev.vals; Z_prev.vals = Z.vals; Z.vals = tmp_ptr;
            tmp_ptr = V_prev.vals; V_prev.vals = V.vals; V.vals = tmp_ptr;
            
            tmp_ptr = R.vals; R.vals = R_next.vals; R_next.vals = tmp_ptr;
            tmp_ptr = W.vals; W.vals = W_next.vals; W_next.vals = tmp_ptr;
            tmp_ptr = T.vals; T.vals = T_next.vals; T_next.vals = tmp_ptr;

            double t_vec1 = omp_get_wtime();
            double spmv_total = (t_spmv1 - t_spmv0) + (t_spmv3 - t_spmv2);
            iprof[i].spmv = spmv_total;
            iprof[i].vector_ops = (t_vec1 - t_vec0) - spmv_total;
            iprof[i].spmv_detail = sp;
        }

        time_stamps.spmxv_end = omp_get_wtime();
        time_stamps.end = time_stamps.spmxv_end;
        
        vector_destroy(&Y);
        vector_destroy(&V);
        vector_destroy(&V_prev);
        vector_destroy(&P);
        vector_destroy(&P_prev);
        vector_destroy(&R);
        vector_destroy(&R_0);
        vector_destroy(&R_next);
        vector_destroy(&S);
        vector_destroy(&S_prev);
        vector_destroy(&T);
        vector_destroy(&T_next);
        vector_destroy(&W);
        vector_destroy(&W_next);
        vector_destroy(&Z);
        vector_destroy(&Z_prev);
        vector_destroy(&Q);
    }
    //-------------------------------------------------------------------------------------------------------

    // Calculate Relavtive residual (|| Ax - b || / || b ||)
    
    Vector tempVec = vector_init(B.nvals);

    MPI_SHARD_CSC_mpi_spmxv_seq(A, X, tempVec, MPI_COMM_WORLD, NULL);
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