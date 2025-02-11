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
    char *input_vector;
    char *output_vector;
};

// Option keys
#define OPT_INPUT_MATRIX 'm'
#define OPT_INPUT_VECTOR 'p'
#define OPT_OUTPUT 'o'

// Command line options
static struct argp_option options[] = {
    {"input-matrix", OPT_INPUT_MATRIX, "FILE", 0, "Input matrix market file path"},
    {"input-vector", OPT_INPUT_VECTOR, "FILE", 0, "Input partition vector file path"},
    {"output", OPT_OUTPUT, "FILE", 0, "Output vector file path"},
    {0}
};

// Parser function
static error_t parse_opt(int key, char *arg, struct argp_state *state) {
    struct arguments *arguments = state->input;

    switch (key) {
        case OPT_INPUT_MATRIX:
            arguments->input_matrix = arg;
            break;
        case OPT_INPUT_VECTOR:
            arguments->input_vector = arg;
            break;
        case OPT_OUTPUT:
            arguments->output_vector = arg;
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

int* readPartVect(char* fileName, int size) ;

// TODO: use GNU Octave to prepare some input output samples to test the functions
// TODO: Premake the vectors for CPU and re-implement the spmxv
// TODO: Simplify GPU spmxv
// TODO: Read an actual X vector and output Y

int main(int argc, char* argv[]) {  // matrix file ve part vector 
    int niters = 1e3;

    struct arguments arguments = {};
    // Default values
    arguments.input_matrix = NULL;
    arguments.input_vector = NULL;
    arguments.output_vector = NULL;

    // Parse arguments
    argp_parse(&argp, argc, argv, 0, 0, &arguments);

    // Check if all required arguments are provided
    if (!arguments.input_matrix || !arguments.input_vector || !arguments.output_vector) {
        fprintf(stderr, "Error: All input and output files must be specified.\n");
        return EXIT_FAILURE;
    }

    // Check file existence
    if (access(arguments.input_matrix, F_OK) == -1) {
        fprintf(stderr, "Error: Input matrix file '%s' does not exist\n", arguments.input_matrix);
        return EXIT_FAILURE;
    }

    if (access(arguments.input_vector, F_OK) == -1) {
        fprintf(stderr, "Error: Input vector file '%s' does not exist\n", arguments.input_vector);
        return EXIT_FAILURE;
    }

    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("GPU Memory - Total: %zu MB, Free: %zu MB\n", 
        total_mem / (1024*1024), free_mem / (1024*1024));


    time_stamps.begin = omp_get_wtime(); // The very Beginning timestamp
//-------------------------------------------------------------------------------------------------------
    // Read the matrix market file and convert it to CSR format
    time_stamps.file_read_begin = omp_get_wtime();
    CSR rmatrix = buReadSparseMatrix(arguments.input_matrix) ; 
    printf("Matrix name : %s\n",arguments.input_matrix);
//-------------------------------------------------------------------------------------------------------
    // Read partition vectors mapping rows to GPU(0) CPU(1)
    int* part_vec = readPartVect(arguments.input_vector, rmatrix.m) ;
    time_stamps.file_read_end = omp_get_wtime();
   
//-------------------------------------------------------------------------------------------------------
    // Print the sum of all non-zeros in the matrix
#ifndef NDEBUG
do{
    double sum = 0;
    for (size_t i = 0; i < rmatrix.nnz; i++)
    {
        sum += rmatrix.val[i];
    }
    DEBUGLOG("Sum of all values in matrix is %lf", sum);
}while(0);
#endif // !NDEBUG
//-------------------------------------------------------------------------------------------------------
    time_stamps.matrix_split_begin = omp_get_wtime();

    // SPLIT_CSR* splits =  cleanSplit( rmatrix, part_vec) ;
    SPLIT_CSR* splits =  sparseSplit( rmatrix, part_vec) ;

    SPLIT_CSR sm_gpu = splits[0] ; 
    SPLIT_CSR sm_cpu = splits[1] ; 
    
    printf("Vector sizes: loc.n=%d, shr.n=%d, loc.m=%d\n", 
        sm_gpu.loc.n, sm_gpu.shr.n, sm_gpu.loc.m);

    time_stamps.matrix_split_end = omp_get_wtime();
//-------------------------------------------------------------------------------------------------------

    // CPU Objects
    Vector X_loc = vector_init(sm_cpu.loc.n);
    Vector X_shr = vector_init(sm_cpu.shr.n);
    Vector Y = vector_init(sm_cpu.loc.m);

    // CPU corresponds of GPU objects
    Vector h_X_loc = vector_init(sm_gpu.loc.n);
    Vector h_X_shr = vector_init(sm_gpu.shr.n);
    Vector h_Y = vector_init(sm_gpu.loc.m);

    // Initialize vectors
    for (size_t i = 0; i < X_loc.nvals; i++)
        X_loc.vals[i] = sm_cpu.locp[i];
    for (size_t i = 0; i < X_shr.nvals; i++)
        X_shr.vals[i] = sm_cpu.shrp[i];
    for (size_t i = 0; i < Y.nvals; i++)
        Y.vals[i] = 0;

    for (size_t i = 0; i < h_X_loc.nvals; i++)
        h_X_loc.vals[i] = sm_gpu.locp[i];
    for (size_t i = 0; i < h_X_shr.nvals; i++)
        h_X_shr.vals[i] = sm_gpu.shrp[i];
    for (size_t i = 0; i < h_Y.nvals; i++){
        h_Y.vals[i] = 0;
    }


//-------------------------------------------------------------------------------------------------------
    time_stamps.gpu_transfer_begin = omp_get_wtime();


    // CUSPARSE APIs
    cusparseHandle_t    handle = NULL;
    CHECK_CUSPARSE( cusparseCreate(&handle) )

    // Y = α * (M * X) + β * Y
    const double alpha = 1.0f;
    const double beta = 1.0f;

    Device_CSR  d_csr_loc = {};
    Device_CSR  d_csr_shr = {};
    CHECK_CUSPARSE(device_csr_create(sm_gpu.loc, &d_csr_loc))
    CHECK_CUSPARSE(device_csr_create(sm_gpu.shr, &d_csr_shr))

    Device_Vector d_X_loc = {};
    Device_Vector d_X_shr = {};
    Device_Vector d_Y = {};
    CHECK_CUSPARSE(device_vector_init(h_X_loc.nvals, &d_X_loc))
    CHECK_CUSPARSE(device_vector_init(h_X_shr.nvals, &d_X_shr))
    CHECK_CUSPARSE(device_vector_init(h_Y.nvals, &d_Y))
    CHECK_CUSPARSE(device_vector_toGPU(h_X_loc, d_X_loc))
    CHECK_CUSPARSE(device_vector_toGPU(h_X_shr, d_X_shr))
    CHECK_CUSPARSE(device_vector_toGPU(h_Y, d_Y))


// Test to CPU:
    CHECK_CUSPARSE(device_vector_toCPU(d_X_loc, h_X_loc))

    Device_Buffer_SpMV  d_buf_loc = NULL;
    Device_Buffer_SpMV  d_buf_shr = NULL;
    CHECK_CUSPARSE(device_buffer_spmv_create(handle, d_csr_loc, d_X_loc, d_Y, &alpha, &beta, &d_buf_loc))
    CHECK_CUSPARSE(device_buffer_spmv_create(handle, d_csr_shr, d_X_shr, d_Y, &alpha, &beta, &d_buf_shr))

    time_stamps.gpu_transfer_end = omp_get_wtime();
//-------------------------------------------------------------------------------------------------------
    // double times[4] = {};
    time_stamps.spmxv_begin = omp_get_wtime();
    for (size_t i = 0; i < niters; i++)
    {

        time_stamps.gpu_begin  = time_stamps.gpu_spmxv_begin = omp_get_wtime(); // Start of GPU SpMV timestamp

        vector_zero(Y);
        CHECK_CUSPARSE(device_vector_zero(d_Y))

        CHECK_CUSPARSE(device_spmv(handle, d_csr_loc, d_X_loc, d_Y, &alpha, &beta, d_buf_loc))
        CHECK_CUSPARSE(device_spmv(handle, d_csr_shr, d_X_shr, d_Y, &alpha, &beta, d_buf_shr))



        time_stamps.gpu_spmxv_end = time_stamps.cpu_spmxv_begin = omp_get_wtime() ; // End of GPU SpMV timestamp

    //-------------------------------------------------------------------------------------------------------
    
        CSR_spmxv_seq(sm_cpu.loc, X_loc, Y);
        CSR_spmxv_seq_acc(sm_cpu.shr, X_shr, Y);

        time_stamps.cpu_spmxv_end = omp_get_wtime();

        CHECK_CUSPARSE(device_vector_toCPU(d_Y, h_Y))

        time_stamps.gpu_end = omp_get_wtime();

        times.cpu_spmxv += time_stamps.cpu_spmxv_end - time_stamps.cpu_spmxv_begin;
        times.gpu_spmxv += time_stamps.gpu_spmxv_end - time_stamps.gpu_spmxv_begin;
        //HACK: This will break if you move gpu times
        times.total_spmxv += time_stamps.gpu_end - time_stamps.gpu_begin;
    }
    time_stamps.spmxv_end = omp_get_wtime();
    //-------------------------------------------------------------------------------------------------------



    printf(
        "n_iters : %d \n"
        "spmv_gpu : %lf \n"
        "spmv_cpu : %lf \n"
        "spmv_total : %lf \n"
        "gpu_transfer : %lf \n"
        "file_read : %lf \n"
        ,
        niters, times.cpu_spmxv, times.gpu_spmxv, times.total_spmxv,
        time_stamps.gpu_transfer_end - time_stamps.gpu_transfer_begin,
        time_stamps.file_read_end - time_stamps.file_read_begin
        );

    printf("everything_total : %lf\n",time_stamps.end - time_stamps.begin) ;

    printf("\n----------------------------------------------------------------------\n");
   
//-------------------------------------------------------------------------------------------------------


    CHECK_CUSPARSE(device_csr_destroy(&d_csr_loc))
    CHECK_CUSPARSE(device_csr_destroy(&d_csr_shr))

    CHECK_CUSPARSE(device_vector_destroy(&d_X_loc))
    CHECK_CUSPARSE(device_vector_destroy(&d_X_shr))
    CHECK_CUSPARSE(device_vector_destroy(&d_Y))

    CHECK_CUSPARSE(device_buffer_spmv_destroy(&d_buf_loc))
    CHECK_CUSPARSE(device_buffer_spmv_destroy(&d_buf_shr))

    CHECK_CUSPARSE( cusparseDestroy(handle) )

//-------------------------------------------------------------------------------------------------------
    // freeSparseMatrix(&cmatrix) ;
    freeSparseMatrix(&rmatrix) ;
    freeSplit_CSR(&(splits[0])) ;
    freeSplit_CSR(&(splits[1])) ;
    free(part_vec) ; 

    // Free functions
    vector_destroy(&X_loc);
    vector_destroy(&X_shr);
    vector_destroy(&Y);
    vector_destroy(&h_X_loc);
    vector_destroy(&h_X_shr);
    vector_destroy(&h_Y);

    return 0 ;
}
int* readPartVect(char* fileName, int size) {

    int* part_vec;
    int pv_size = size;
    ALLOC_ARRAY(part_vec, pv_size);

    if (part_vec == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return NULL;
     }

     FILE* f = fopen(fileName,"r") ;
     if (f == NULL) {
        fprintf(stderr,"Part Vector File cannot be opened ! ") ;
        free(part_vec);
        return NULL ;
     }
    
    int i=0;
    while( i<size && fscanf(f, "%d",&part_vec[i++]) == 1 );
        
     
    if (i < size && !feof(f)) 
        fprintf(stderr, "Error reading file\n");
    

    fclose(f);

    return part_vec ;    
}
