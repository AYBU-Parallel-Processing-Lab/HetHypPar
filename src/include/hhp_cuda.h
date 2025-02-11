#pragma once

#include "hhp_common.h"
#include <cusparse.h>
#include <driver_types.h>

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA ERROR [%s:%d]: %s (%d)\n",             \
               __FILE__, __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE ERROR  [%s:%d]: %s (%d)\n",         \
               __FILE__,__LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

cusparseStatus_t device_csr_create(CSR in, Device_CSR* out);
cusparseStatus_t device_csr_destroy(Device_CSR* in);
cusparseStatus_t device_buffer_spmv_create(cusparseHandle_t handle, Device_CSR mat, Device_Vector X, Device_Vector Y, const double* alpha, const double* beta, Device_Buffer_SpMV* out);
cusparseStatus_t device_buffer_spmv_destroy(Device_Buffer_SpMV* in);
cusparseStatus_t device_vector_init(size_t size, Device_Vector* out);
cusparseStatus_t device_vector_destroy(Device_Vector* in);
cusparseStatus_t device_vector_zero(Device_Vector vec);
cusparseStatus_t device_vector_toGPU(Vector src, Device_Vector dest);
cusparseStatus_t device_vector_toCPU(Device_Vector src, Vector dest);
cusparseStatus_t device_spmv(cusparseHandle_t handle, Device_CSR mat, Device_Vector X, Device_Vector Y, const double* alpha, const double* beta, Device_Buffer_SpMV buf);