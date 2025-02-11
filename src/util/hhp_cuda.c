#include "hhp_cuda.h"
#include "hhp_common.h"
#include "hhp_util.h"
#include <cuda_runtime_api.h>    // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>            // cusparseSpMV
#include <driver_types.h>
#include <library_types.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>               // printf
#include <stdlib.h>              // EXIT_FAILURE
#include <time.h>
#include <sys/time.h>
#include <mpi.h>

// X and Y vectors has to be initialized
cusparseStatus_t device_csr_create(CSR in, Device_CSR* out){
    Device_CSR res = {
        .data = {
            .I = NULL,
            .J = NULL,
            .val = NULL,
            .nnz = in.nnz,
            .m = in.m,
            .n = in.n
        },
    };
    // Create Memory blocks for I, J, val
    CHECK_CUDA( cudaMalloc((void**) &(res.data.I), (in.m + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &(res.data.J),    in.nnz * sizeof(int)))
    CHECK_CUDA( cudaMalloc((void**) &(res.data.val),     in.nnz * sizeof(double)))
    
    // Transfer I, J, val to device
    CHECK_CUDA( cudaMemcpy(res.data.I, in.I,(in.m + 1) * sizeof(*in.I), cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(res.data.J, in.J,(in.nnz) * sizeof(*in.J), cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(res.data.val, in.val,(in.nnz) * sizeof(*in.val), cudaMemcpyHostToDevice) )

    CHECK_CUSPARSE(
        cusparseCreateCsr(&res.desc, in.m, in.n, in.nnz,
        res.data.I, res.data.J, res.data.val,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F
        )
    )

    *out = res;

    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t device_csr_destroy(Device_CSR* in){
    if (!in->data.I || !in->data.J || !in->data.val){
        ERRORLOG("Attempted to free NULLED  device CSR")
        return CUSPARSE_STATUS_INVALID_VALUE;
    }

    CHECK_CUSPARSE(cusparseDestroySpMat(in->desc))
    CHECK_CUDA(cudaFree(in->data.I))
    CHECK_CUDA(cudaFree(in->data.J))
    CHECK_CUDA(cudaFree(in->data.val))

    in->data = (CSR){};

    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t device_buffer_spmv_create(cusparseHandle_t handle, Device_CSR mat, Device_Vector X, Device_Vector Y, const double* alpha, const double* beta, Device_Buffer_SpMV* out){
    // Y = α * (matrix * X) + β * Y
    size_t buffersize = 0;
    CHECK_CUSPARSE(
        cusparseSpMV_bufferSize(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            alpha, mat.desc, X.desc, beta, Y.desc, CUDA_R_64F,
            CUSPARSE_MV_ALG_DEFAULT, &buffersize
        )
    )

    CHECK_CUDA( cudaMalloc(out, buffersize));
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t device_buffer_spmv_destroy(Device_Buffer_SpMV* in){
    if (*in == NULL){
        ERRORLOG("Attempted to free NULL buffer")
        return CUSPARSE_STATUS_SUCCESS;
    }
    CHECK_CUDA(cudaFree(in));
    *in = NULL;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t device_vector_init(size_t size, Device_Vector* out){
    out->nvals = size;

    CHECK_CUDA(cudaMalloc((void **)&(out->vals), sizeof(*out->vals)*out->nvals))

    CHECK_CUSPARSE(
        cusparseCreateDnVec(
            &(out->desc),
            out->nvals,
            out->vals,
            CUDA_R_64F
        )
    )
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t device_vector_destroy(Device_Vector* in){
    if (in->vals == NULL){
        ERRORLOG("Attempted to free NULL Device vector")
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    CHECK_CUDA(cudaFree(in->vals))
    CHECK_CUSPARSE(cusparseDestroyDnVec(in->desc))
    in->vals = NULL;
    in->nvals = 0;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t device_vector_zero(Device_Vector vec){
    CHECK_CUDA( cudaMemset(vec.vals, 0, vec.nvals*sizeof(*vec.vals)))
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t device_vector_toGPU(Vector src, Device_Vector dest){
    CHECK_CUDA(cudaMemcpy(dest.vals, src.vals, src.nvals*(sizeof(*src.vals)), cudaMemcpyHostToDevice))
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t device_vector_toCPU(Device_Vector src, Vector dest){
    CHECK_CUDA(cudaMemcpy(dest.vals, src.vals, src.nvals*(sizeof(*src.vals)), cudaMemcpyDeviceToHost))
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t device_spmv(cusparseHandle_t handle, Device_CSR mat, Device_Vector X, Device_Vector Y, const double* alpha, const double* beta, Device_Buffer_SpMV buf){
    CHECK_CUSPARSE(
        cusparseSpMV(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            alpha, mat.desc, X.desc, beta, Y.desc,
            CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, buf
        )
    )
    CHECK_CUDA(cudaDeviceSynchronize())
    return CUSPARSE_STATUS_SUCCESS;
}

