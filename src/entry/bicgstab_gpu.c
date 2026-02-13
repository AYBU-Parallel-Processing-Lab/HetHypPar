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

#include "argp.h"
#include "unistd.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <stdio.h>  // fopen
#include <stdlib.h> // EXIT_FAILURE
#include <string.h> // strtok

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


int gpu_BiCGStab(cublasHandle_t       cublasHandle,
                 cusparseHandle_t     cusparseHandle,
                 int                  m,
                 cusparseSpMatDescr_t matA,
                 Device_Vector        d_B,
                 Device_Vector        d_X,
                 Device_Vector        d_R0,
                 Device_Vector        d_R,
                 Device_Vector        d_P,
                 Device_Vector        d_S,
                 Device_Vector        d_V,
                 Device_Vector        d_T,
                 Device_Vector        d_tmp,
                 void*                d_bufferMV,
                 int                  maxIterations,
                 double               tolerance) {
    const double zero      = 0.0;
    const double one       = 1.0;
    const double minus_one = -1.0;
    //--------------------------------------------------------------------------

    //--------------------------------------------------------------------------
    // ### 1 ### R0 = b - A * X0 (using initial guess in X)
    //    (a) copy b in R0
    CHECK_CUDA( cudaMemcpy(d_R0.vals, d_B.vals, m * sizeof(double),
                           cudaMemcpyDeviceToDevice) )
    //    (b) compute R = -A * X0 + R
    CHECK_CUSPARSE( cusparseSpMV(cusparseHandle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &minus_one, matA, d_X.desc, &one, d_R0.desc,
                                 CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                                 d_bufferMV) )
    //--------------------------------------------------------------------------
    double alpha, delta, delta_prev, omega;
    CHECK_CUBLAS( cublasDdot(cublasHandle, m, d_R0.vals, 1, d_R0.vals, 1,
                             &delta) )
    delta_prev = delta;
    // R = R0
    CHECK_CUDA( cudaMemcpy(d_R.vals, d_R0.vals, m * sizeof(double),
                           cudaMemcpyDeviceToDevice) )
    //--------------------------------------------------------------------------
    // nrm_R0 = ||R||
    double nrm_R;
    CHECK_CUBLAS( cublasDnrm2(cublasHandle, m, d_R0.vals, 1, &nrm_R) )
    double threshold = tolerance * nrm_R;
    printf("  Initial Residual: Norm %e' threshold %e\n", nrm_R, threshold);
    //--------------------------------------------------------------------------
    // ### 2 ### repeat until convergence based on max iterations and
    //           and relative residual
    for (int i = 1; i <= maxIterations; i++) {
        printf("  Iteration = %d; Error Norm = %e\n", i, nrm_R);
        //----------------------------------------------------------------------
        // ### 4, 7 ### P_i = R_i
        if (i == 1) {
            CHECK_CUDA(cudaMemcpy(d_P.vals, d_R.vals, m * sizeof(double),
                                  cudaMemcpyDeviceToDevice))
        }
        else {
            //------------------------------------------------------------------
            // ### 6 ### beta = (delta_i / delta_i-1) * (alpha / omega_i-1)
            //    (a) delta_i = (R'_0, R_i-1)
            CHECK_CUBLAS( cublasDdot(cublasHandle, m, d_R0.vals, 1, d_R.vals, 1,
                                     &delta) )
            //    (b) beta = (delta_i / delta_i-1) * (alpha / omega_i-1);
            double beta = (delta / delta_prev) * (alpha / omega);
            delta_prev  = delta;
            //------------------------------------------------------------------
            // ### 7 ### P = R + beta * (P - omega * V)
            //    (a) P = - omega * V + P
            double minus_omega = -omega;
            CHECK_CUBLAS( cublasDaxpy(cublasHandle, m, &minus_omega, d_V.vals, 1,
                                      d_P.vals, 1) )
            //    (b) P = beta * P
            CHECK_CUBLAS( cublasDscal(cublasHandle, m, &beta, d_P.vals, 1) )
            //    (c) P = R + P
            CHECK_CUBLAS( cublasDaxpy(cublasHandle, m, &one, d_R.vals, 1,
                                      d_P.vals, 1) )
        }
        //----------------------------------------------------------------------
        // // ### 9 ### P_aux = M_U^-1 M_L^-1 P_i
        // //    (a) M_L^-1 P_i => tmp    (triangular solver)
        // CHECK_CUDA( cudaMemset(d_tmp.vals,   0x0, m * sizeof(double)) )
        // CHECK_CUDA( cudaMemset(d_P_aux.vals, 0x0, m * sizeof(double)) )
        // CHECK_CUSPARSE( cusparseSpSV_solve(cusparseHandle,
        //                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
        //                                    &one, matM_lower, d_P.vec, d_tmp.vec,
        //                                    CUDA_R_64F,
        //                                    CUSPARSE_SPSV_ALG_DEFAULT,
        //                                    spsvDescrL) )
        // //    (b) M_U^-1 tmp => P_aux    (triangular solver)
        // CHECK_CUSPARSE( cusparseSpSV_solve(cusparseHandle,
        //                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
        //                                    &one, matM_upper, d_tmp.vec,
        //                                    d_P_aux.vec, CUDA_R_64F,
        //                                    CUSPARSE_SPSV_ALG_DEFAULT,
        //                                    spsvDescrU) )
        //----------------------------------------------------------------------
        // ### 10 ### alpha = (R'0, R_i-1) / (R'0, A * P_aux)
        //    (a) V = A * P_aux
        CHECK_CUSPARSE( cusparseSpMV(cusparseHandle,
                                     CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                     matA, d_P.desc, &zero, d_V.desc,
                                     CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                                     d_bufferMV) )
        //    (b) denominator = R'0 * V
        double denominator;
        CHECK_CUBLAS( cublasDdot(cublasHandle, m, d_R0.vals, 1, d_V.vals, 1,
                                 &denominator) )
        alpha = delta / denominator;
        PRINT_INFO(delta)
        PRINT_INFO(alpha)
        //----------------------------------------------------------------------
        // ### 11 ###  X_i = X_i-1 + alpha * P_aux
        CHECK_CUBLAS( cublasDaxpy(cublasHandle, m, &alpha, d_P.vals, 1,
                                  d_X.vals, 1) )
        //----------------------------------------------------------------------
        // ### 12 ###  S = R_i-1 - alpha * (A * P_aux)
        //    (a) S = R_i-1
        CHECK_CUDA( cudaMemcpy(d_S.vals, d_R.vals, m * sizeof(double),
                               cudaMemcpyDeviceToDevice) )
        //    (b) S = -alpha * V + R_i-1
        double minus_alpha = -alpha;
        CHECK_CUBLAS( cublasDaxpy(cublasHandle, m, &minus_alpha, d_V.vals, 1,
                                  d_S.vals, 1) )
        //----------------------------------------------------------------------
        // ### 13 ###  check ||S|| < threshold
        double nrm_S;
        CHECK_CUBLAS( cublasDnrm2(cublasHandle, m, d_S.vals, 1, &nrm_S) )
        PRINT_INFO(nrm_S)
        if (nrm_S < threshold)
            break;
        //----------------------------------------------------------------------
        // // ### 14 ### S_aux = M_U^-1 M_L^-1 S
        // //    (a) M_L^-1 S => tmp    (triangular solver)
        // cudaMemset(d_tmp.ptr, 0x0, m * sizeof(double));
        // cudaMemset(d_S_aux.ptr, 0x0, m * sizeof(double));
        // CHECK_CUSPARSE( cusparseSpSV_solve(cusparseHandle,
        //                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
        //                                    &one, matM_lower, d_S.vec, d_tmp.vec,
        //                                    CUDA_R_64F,
        //                                    CUSPARSE_SPSV_ALG_DEFAULT,
        //                                    spsvDescrL) )
        // //    (b) M_U^-1 tmp => S_aux    (triangular solver)
        // CHECK_CUSPARSE( cusparseSpSV_solve(cusparseHandle,
        //                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
        //                                    &one, matM_upper, d_tmp.vec,
        //                                    d_S_aux.vec, CUDA_R_64F,
        //                                    CUSPARSE_SPSV_ALG_DEFAULT,
        //                                    spsvDescrU))
        //----------------------------------------------------------------------
        // ### 15 ### omega = (A * S_aux, s) / (A * S_aux, A * S_aux)
        //    (a) T = A * S_aux
        CHECK_CUSPARSE( cusparseSpMV(cusparseHandle,
                                     CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                     matA, d_S.desc, &zero, d_T.desc,
                                     CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                                     d_bufferMV) )
        //    (b) omega_num = (A * S_aux, s)
        double omega_num, omega_den;
        CHECK_CUBLAS( cublasDdot(cublasHandle, m, d_T.vals, 1, d_S.vals, 1,
                                 &omega_num) )
        //    (c) omega_den = (A * S_aux, A * S_aux)
        CHECK_CUBLAS( cublasDdot(cublasHandle, m, d_T.vals, 1, d_T.vals, 1,
                                 &omega_den) )
        //    (d) omega = omega_num / omega_den
        omega = omega_num / omega_den;
        PRINT_INFO(omega)
        // ---------------------------------------------------------------------
        // ### 16 ### omega = X_i = X_i-1 + alpha * P_aux + omega * S_aux
        //    (a) X_i has been updated with h = X_i-1 + alpha * P_aux
        //        X_i = omega * S_aux + X_i
        CHECK_CUBLAS( cublasDaxpy(cublasHandle, m, &omega, d_S.vals, 1,
                                  d_X.vals, 1) )
        // ---------------------------------------------------------------------
        // ### 17 ###  R_i+1 = S - omega * (A * S_aux)
        //    (a) copy S in R
        CHECK_CUDA( cudaMemcpy(d_R.vals, d_S.vals, m * sizeof(double),
                               cudaMemcpyDeviceToDevice) )
        //    (a) R_i+1 = -omega * T + R
        double minus_omega = -omega;
        CHECK_CUBLAS( cublasDaxpy(cublasHandle, m, &minus_omega, d_T.vals, 1,
                                  d_R.vals, 1) )
       // ---------------------------------------------------------------------
        // ### 18 ###  check ||R_i|| < threshold
        CHECK_CUBLAS( cublasDnrm2(cublasHandle, m, d_R.vals, 1, &nrm_R) )
        PRINT_INFO(nrm_R)
        if (nrm_R < threshold)
            break;
    }
    //--------------------------------------------------------------------------
    printf("Check Solution\n"); // ||R = b - A * X||
    //    (a) copy b in R
    CHECK_CUDA( cudaMemcpy(d_R.vals, d_B.vals, m * sizeof(double),
                           cudaMemcpyDeviceToDevice) )
    // R = -A * X + R
    CHECK_CUSPARSE( cusparseSpMV(cusparseHandle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE, &minus_one,
                                 matA, d_X.desc, &one, d_R.desc, CUDA_R_64F,
                                 CUSPARSE_SPMV_ALG_DEFAULT, d_bufferMV) )
    // check ||R||
    CHECK_CUBLAS( cublasDnrm2(cublasHandle, m, d_R.vals, 1, &nrm_R) )
    printf("Final error norm = %e\n", nrm_R);
    //--------------------------------------------------------------------------
    return EXIT_SUCCESS;
}

//==============================================================================

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

    cublasHandle_t   cublasHandle   = NULL;
    cusparseHandle_t cusparseHandle = NULL;
    CHECK_CUBLAS( cublasCreate(&cublasHandle) )
    CHECK_CUSPARSE( cusparseCreate(&cusparseHandle) )

//-------------------------------------------------------------------------------------------------------
    // Read the matrix market file and convert it to CSR format
    time_stamps.file_read_begin = omp_get_wtime();

    // [ ] IS FREED?
    Device_CSR dA;
    {
        // [x] IS FREED?
        CSR cA = buReadSparseMatrix(arguments.input_matrix);
        CHECK_CUSPARSE(device_csr_create(cA, &dA));
        freeSparseMatrix(&cA);
    }

    printf("Matrix name : %s\n",arguments.input_matrix);
//-------------------------------------------------------------------------------------------------------
    // Read partition vectors mapping rows to GPU(0) CPU(1)

    // [ ] IS FREED?
    Device_Vector X;
    // [ ] IS FREED?
    Device_Vector B;
    
    {
        // [x] IS FREED?
        Vector Xt = vector_read(arguments.input_x, dA.data.n);
        // [x] IS FREED?
        Vector Bt = vector_read(arguments.input_y, dA.data.m);
        CHECK_CUSPARSE(device_vector_init(Xt.nvals, &X));
        CHECK_CUSPARSE(device_vector_init(Bt.nvals, &B));

        CHECK_CUSPARSE(device_vector_toGPU(Xt, X));
        CHECK_CUSPARSE(device_vector_toGPU(Bt, B));

        vector_destroy(&Xt);
        vector_destroy(&Bt);
    }

    time_stamps.file_read_end = omp_get_wtime();

//-------------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------------


    // [ ] IS FREED?
    Device_Vector Y;
    CHECK_CUSPARSE(device_vector_init(dA.data.m, &Y));
    CHECK_CUSPARSE(device_vector_zero(Y));
    
    // BICGStab variables
    double rho = 1.0;
    double alpha = 1.0;
    double omega = 1.0;
    
    // [ ] IS FREED?
    Device_Vector V; // Add pieces for this
    CHECK_CUSPARSE(device_vector_init(Y.nvals, &V));
    CHECK_CUSPARSE(device_vector_zero(V));
    // [ ] IS FREED?
    Device_Vector P; // Add pieces for this
    CHECK_CUSPARSE(device_vector_init(Y.nvals, &P));
    CHECK_CUSPARSE(device_vector_zero(P));
    
    // [ ] IS FREED?
    Device_Vector R;  // Add pieces for this
    CHECK_CUSPARSE(device_vector_init(B.nvals, &R));
    CHECK_CUDA(device_vector_GPUtoGPU(B, R));
    
    
    const double mx_alpha = 1.00;
    const double mx_neg = -1.00;
    const double mx_beta = 0.00;
    // [ ] IS FREED?
    Device_Buffer_SpMV dA_buf;
    CHECK_CUSPARSE(device_buffer_spmv_create(cusparseHandle, dA.desc, X, B, &mx_alpha, &mx_beta, &dA_buf))

    CHECK_CUSPARSE(device_csr_spmv(cusparseHandle, dA, X, Y, mx_alpha, mx_beta, dA_buf))
    CHECK_CUBLAS(device_vector_axpy(cublasHandle, Y, mx_neg, R))
    CHECK_CUSPARSE(device_vector_zero(Y))
    
    // [ ] IS FREED?
    Device_Vector R_0;
    CHECK_CUSPARSE(device_vector_init(R.nvals, &R_0))
    CHECK_CUDA(device_vector_GPUtoGPU(R, R_0))
    // [ ] IS FREED?
    Device_Vector S;
    CHECK_CUSPARSE(device_vector_init(Y.nvals, &S))
    // [ ] IS FREED?
    Device_Vector T;
    CHECK_CUSPARSE(device_vector_init(Y.nvals, &T))


//-------------------------------------------------------------------------------------------------------
    time_stamps.gpu_transfer_begin = omp_get_wtime();


    time_stamps.gpu_transfer_end = omp_get_wtime();
//-------------------------------------------------------------------------------------------------------
    // double times[4] = {};
    time_stamps.spmxv_begin = omp_get_wtime();
    // gpu_BiCGStab(cublasHandle, cusparseHandle, Ad.data.m, Ad.desc, B, X, R_0, R, P, S, V, T, Y, Ad_buf, niters, 0.0000000001);
    for (size_t i = 0; i < niters; i++)
    {
        // calc rho_n+1
        double temp_rho; // = vector_dot_seq(R, R_0);
        CHECK_CUBLAS(device_vector_dot(cublasHandle, R, R_0, &temp_rho))
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
        // CSR_spmxv_seq(A, P, V); // result in V
        CHECK_CUSPARSE(device_csr_spmv(cusparseHandle, dA, P, V, mx_alpha, mx_beta, dA_buf))
        //==============
        // calc alpha_n+1
        double temp_rv;
        CHECK_CUBLAS(device_vector_dot(cublasHandle, R_0, V, &temp_rv))
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
        CHECK_CUSPARSE(device_csr_spmv(cusparseHandle, dA, S, T, mx_alpha, mx_beta, dA_buf))
        //==============
        // calc omega_n+1
        double temp_ts;
        CHECK_CUBLAS(device_vector_dot(cublasHandle, T, S, &temp_ts))
        double temp_tt;
        CHECK_CUBLAS(device_vector_dot(cublasHandle, T, T, &temp_tt))
        // omega = vector_dot_seq(T, S)/vector_dot_seq(T, T);
        omega = temp_ts/temp_tt;
        
        //==============
        // calc X_n+1
        // vector_scale_seq(P, alpha, Y);
        // vector_add_seq(X, Y, X);
        // vector_scale_seq(S, omega, Y);
        // vector_add_seq(X, Y, X);
        CHECK_CUBLAS(device_vector_axpy(cublasHandle, P, alpha, X))
        CHECK_CUBLAS(device_vector_axpy(cublasHandle, S, omega, X))
        //==============
        // calc R_n+1
        // vector_scale_seq(T, omega, R);
        // vector_sub_seq(S, R, R);
        CHECK_CUDA(device_vector_GPUtoGPU(S, R))
        CHECK_CUBLAS(device_vector_axpy(cublasHandle, T, -omega, R))
        //==============
        // calc tol
        double tol;// = vector_dot_seq(S, S);
        CHECK_CUBLAS(device_vector_dot(cublasHandle, S, S, &tol))
        //==============
    }
    time_stamps.spmxv_end = omp_get_wtime();
    time_stamps.end = time_stamps.spmxv_end;
    
    //-------------------------------------------------------------------------------------------------------
    // Calculate Relavtive residual (|| Ax - b || / || b ||)
    
    // CSR_spmxv_seq(A, X, Y);
    CHECK_CUSPARSE(device_csr_spmv(cusparseHandle, dA, X, Y, mx_alpha, mx_beta, dA_buf))
    CHECK_CUBLAS(device_vector_axpy(cublasHandle, B, mx_neg, Y))
    double sy;
    CHECK_CUBLAS(device_vector_dot(cublasHandle, Y, Y, &sy))
    double sb;
    CHECK_CUBLAS(device_vector_dot(cublasHandle, B, B, &sb))
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
    Vector Xh = vector_init(X.nvals);
    CHECK_CUSPARSE(device_vector_toCPU(X, Xh))

    vector_write(arguments.output_x, Xh);

    vector_destroy(&Xh);

//-------------------------------------------------------------------------------------------------------

    return EXIT_SUCCESS;
}