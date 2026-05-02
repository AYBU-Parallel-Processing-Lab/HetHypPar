#pragma once

#ifdef __cplusplus
#define restrict __restrict__
#endif


#include "stdlib.h"
#include "hhp_util.h"
#include <cusparse.h>

#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    double * restrict vals;
    unsigned int nvals;
} Vector;

typedef struct {
    int * restrict vals;
    unsigned int nvals;
} iVector;

typedef struct{
    cusparseDnVecDescr_t desc;
    double * vals;
    unsigned int nvals;
} Device_Vector;

typedef void* Device_Buffer_SpMV;

// Per-SpMV-call timing breakdown. Accumulated across calls when reused.
typedef struct {
    double send_fill;    // packing X values into send buffer
    double local_spmv;   // local SpMV computation
    double comm_wait;    // blocked waiting for MPI receives
    double shared_spmv;  // SpMV on received shared data
    double send_wait;    // waiting for MPI sends to complete
} SpMV_Profile;

// Per-iteration timing breakdown for BiCGStab solver.
typedef struct {
    double spmv;           // total time in SpMV calls
    double vector_ops;     // dot products, scales, adds
    SpMV_Profile spmv_detail; // SpMV sub-timings accumulated over the iteration
} Iter_Profile;

// TODO: Verify restrict does what you think it does
// Column sorted Coordinate.
typedef struct {
    int* restrict I;                     // Row index
    int* restrict J;                     // Column index
    double* restrict val;                // value
    int nnz;                    // number of non-zeros
    int m;                      // number of rows
    int n;                      // number of columns
} COO;

// Compressed Sparse Row.
// .I contains the compressed index. has m+1 items
typedef COO CSR;

// Compressed Sparse Column.
// .I contains the compressed index. has n+1 items
typedef COO CSC;

typedef struct {
    cusparseSpMatDescr_t desc; // Matrix descriptor
    CSR data;
}Device_CSR;

typedef struct {
    cusparseSpMatDescr_t desc; // Matrix descriptor
    CSC data;
}Device_CSC;


/**
 * @struct COMM
 * @brief Structure for managing MPI communication in CSR-like format
 * 
 * The COMM structure manages the communication pattern between MPI processes
 * using a sparse representation similar to Compressed Sparse Row (CSR) format.
 * It stores which processes need to communicate with the current process and
 * which vector elements need to be exchanged.
 * 
 * The data structure optimizes communication by only tracking elements that 
 * require communication, excluding elements that can be processed locally.
 * 
 * @note The structure uses CSR-like indexing where:
 *       - I array works like the 'ia' array in CSR format (row pointers)
 *       - J array works like the 'ja' array in CSR format (column indices)
 *       - val array contains the actual values to communicate
 */
 typedef struct {
    /**
     * @brief Array of values to be communicated
     * 
     * Stores the actual data values that need to be sent/received.
     * Dimensioned as val[0..numCommItems-1]
     * 
     * @note This array is occasionally repurposed as a temporary buffer
     */
    double *val;
    
    /**
     * @brief Indices of items to communicate
     * 
     * Similar to the 'ja' array in CSR format, stores the indices of vector
     * elements that need to be communicated.
     * Dimensioned as J[0..numCommItems-1]
     */
    int *J;
    
    /**
     * @brief Process offsets in the communication arrays
     * 
     * Similar to the 'ia' array in CSR format, stores the starting positions
     * in the val and J arrays for each process.
     * Dimensioned as I[0..numProcs-1]
     * 
     * For process p, elements from I[p] to I[p+1]-1 in the J and val arrays
     * correspond to data exchanged with that process.
     */
    int *I;
    
    /**
     * @brief MPI ranks of processes that communicate with this process
     * 
     * Array of MPI ranks for processes that exchange data with the current process.
     * Dimensioned as ranks[0..num-1]
     */
    int *ranks;
    
    /**
     * @brief Number of processes in communication with this process
     * 
     * Indicates how many other MPI processes send/receive data to/from this process.
     */
    int num;
} COMM;

/**
 * @var numCommItems
 * @brief Global variable representing the total number of vector elements to communicate
 * 
 * This variable (used in comments for array dimensions) represents the total number
 * of vector elements that need to be communicated across all processes.
 * Elements that can be processed without communication are excluded.
 */

/**
 * @var numProcs
 * @brief Global variable representing the total number of MPI processes
 * 
 * This variable (used in comments for array dimensions) represents the total
 * number of MPI processes involved in the computation.
 */



typedef struct{
    CSR loc;                // Local Matrix
    CSR shr;                // Shared Matrix (needs communication to complete)
    iVector gind;           // Global indices of the columns. Has both loc and shr columns starting at gind.vals[0] and gind.vals[loc.n]. Size is loc.n + shr.n
    COMM send;              // Holds information on what items to send from input X of current process
    COMM recv;              // Holds information on what items to receive from input X of other processes for shr
} SHARD_CSR;



typedef struct{
    CSC loc;                // Local Matrix
    CSC shr;                // Shared Matrix (needs communication to complete)
    iVector gind;           // Global indices of the columns. Has both loc and shr columns starting at gind.vals[0] and gind.vals[loc.n]. Size is loc.n + shr.n
    COMM send;              // Holds information on what items to send from input X of current process
    COMM recv;              // Holds information on what items to receive from input X of other processes for shr
} SHARD_CSC;


typedef struct{
    Device_CSC loc;                // Local Matrix
    Device_CSC shr;                // Shared Matrix (needs communication to complete)
    iVector gind;           // Global indices of the columns. Has both loc and shr columns starting at gind.vals[0] and gind.vals[loc.n]. Size is loc.n + shr.n
    COMM send;              // Holds information on what items to send from input X of current process
    COMM recv;              // Holds information on what items to receive from input X of other processes for shr
} Device_SHARD_CSC;

// returns an array containing the rank of each row.
// NEEDS TO BE FREED.
static inline int* CalcWeights(CSR* in){
    int *out = (int*)malloc(in->m*sizeof(int));
    for(int i = 0; i < in->m; i++){
        out[i]= in->I[i+1] - in->I[i];
    }
    return out;
}

static inline COO SparseTranspose(COO in){
    int *temp = in.I;
    in.I = in.J;
    in.J = temp;
    int swap = in.m;
    in.m = in.n;
    in.n = swap;
    return in;
}

static inline void freeSparseMatrix(COO *in){
    if(in == NULL) return;

    FREE_AND_NULL_IF(in->I)
    FREE_AND_NULL_IF(in->J);
    FREE_AND_NULL_IF(in->val);
}


#ifdef __cplusplus
}
#endif