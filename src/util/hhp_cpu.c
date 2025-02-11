#include "omp.h"
#include <math.h>
#include <mpi.h>
#include <stdio.h>               // printf
#include <stdlib.h>              // EXIT_FAILURE
#include <string.h>
#include <time.h>

#include "mkl_spblas.h"
#include "hhp_common.h"
#include "hhp_util.h"
#include "hhp_cpu.h"

Vector vector_init(unsigned int nvals){
    double * vals = NULL;
    if (nvals <= 0)
        ABORT("Attempted to create a vector of size %d.", nvals)
    ALLOC_ARRAY(vals, nvals);
    return (Vector){
        .vals = vals,
        .nvals = nvals
    };
}

void vector_destroy(Vector* vec){
    if ((vec->nvals <= 0 ) || !vec->vals)
        ABORT("Attempted to free invalid Vector")
    FREE_AND_NULL(vec->vals);
    vec->nvals = 0;
}

void vector_zero(Vector vec){
    if ((vec.nvals <= 0 ) || !vec.vals)
        ABORT("Attempted to zero invalid Vector")
    void* res = memset(vec.vals, 0, vec.nvals*sizeof(*vec.vals));
    if (!res)
        ABORT("Failed to memset vector to 0")
}

// TODO: Test this
Vector vector_init_clone(Vector vecA){
    Vector res = {
        .vals = NULL,
        .nvals = vecA.nvals
    };
    if (!vecA.vals || (vecA.nvals == 0))
        ABORT("Attempted to clone invalid vector")
    ALLOC_ARRAY(res.vals, vecA.nvals);
    memcpy(res.vals, vecA.vals, sizeof(*vecA.vals)*vecA.nvals);

    return res;
}

// TODO: Test this
double vector_dot_seq(Vector vecA, Vector vecB){
    double res = 0.0;
    for (size_t i = 0; i < vecA.nvals; i++){
        res += vecA.vals[i]*vecB.vals[i];
    }
    return res;
}

// TODO: Test this
void vector_scale_seq_inplace(Vector vecA, double scalar){
    for (size_t i=0; i< vecA.nvals; i++){
        vecA.vals[i] = vecA.vals[i]*scalar;
    }
}

// TODO: Test this
void vector_add_seq_inplace(Vector vecA, Vector vecB){
    for (size_t i = 0; i < vecA.nvals; i++)
    {
        vecA.vals[i] += vecB.vals[i];
    }
}

// TODO: Test this
void vector_sub_seq_inplace(Vector vecA, Vector vecB){
    for (size_t i = 0; i < vecA.nvals; i++)
    {
        vecA.vals[i] -= vecB.vals[i];
    }
}

// TODO: Test this
void vector_mul_seq_inplace(Vector vecA, Vector vecB){
    for (size_t i = 0; i < vecA.nvals; i++)
    {
        vecA.vals[i] *= vecB.vals[i];
    }
}

// TODO: Test this
void vector_div_seq_inplace(Vector vecA, Vector vecB){
    for (size_t i = 0; i < vecA.nvals; i++)
    {
        vecA.vals[i] /= vecB.vals[i];
    }
}

// TODO: Test this
void CSR_spmxv_seq(CSR A, Vector X, Vector Y){

    // #pragma omp distribute parallel for simd
    for (size_t i = 0; i < A.m; i++) {
        double res = 0.0;
        for (size_t j=A.I[i]; j < A.I[i+1]; j++) {
            res += A.val[j]*X.vals[A.J[j]];
        }
        Y.vals[i] = res;
    }
}

// TODO: Test this
void CSR_spmxv_seq_acc(CSR A, Vector X, Vector Y){
    // #pragma omp distribute parallel for simd
    for (size_t i = 0; i < A.m; i++) {
        double res = 0.0;
        for (size_t j=A.I[i]; j < A.I[i+1]; j++) {
            res += A.val[j]*X.vals[A.J[j]];
        }
        Y.vals[i] += res;
    }
}


double spmxv_cpu_mkl(SPLIT_CSR in){
    double start, end ;

    double *X, *Y;
    ALLOC_ARRAY(X, in.loc.n + in.shr.n);
    CALLOC_ARRAY(Y, in.loc.m);

    for (size_t i=0; i<in.loc.n; i++) X[i] = in.locp[i] +1;
    for (size_t i=0; i<in.shr.n; i++) X[i + in.loc.n] = in.shrp[i] + 1;

    sparse_matrix_t loc, shr;
    sparse_status_t status;

    status = mkl_sparse_d_create_csr(&loc, SPARSE_INDEX_BASE_ZERO,
                                    in.loc.m, in.loc.n,
                                    in.loc.I, in.loc.I+1,
                                    in.loc.J, in.loc.val);

    if (status != SPARSE_STATUS_SUCCESS){
        return 0.0;
    // ABORT("MKL FAILED TO CREATE LOCAL MATRIX WITH ERROR %d", status)
    } 

    if(in.shr.n != 0){
        status = mkl_sparse_d_create_csr(&shr, SPARSE_INDEX_BASE_ZERO,
                                    in.shr.m, in.shr.n,
                                    in.shr.I, in.shr.I+1,
                                    in.shr.J, in.shr.val);

        if (status != SPARSE_STATUS_SUCCESS) ABORT("MKL FAILED TO CREATE SHARED MATRIX WITH ERROR %d", status)
    }


    struct matrix_descr descr = {
        SPARSE_MATRIX_TYPE_GENERAL,
        // SPARSE_FILL_MODE_FULL,
        // SPARSE_DIAG_NON_UNIT
    };
    start = omp_get_wtime();

    // start = omp_get_wtime();
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, loc, descr, X, 0.0, Y);
    if(in.shr.n != 0)
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, shr, descr, X+in.shr.n, 1.0, Y);

    end = omp_get_wtime();

    mkl_sparse_destroy(loc);
    if(in.shr.n != 0)
        mkl_sparse_destroy(shr);
    FREE_AND_NULL(X);
    FREE_AND_NULL(Y);

    return (end - start);
}