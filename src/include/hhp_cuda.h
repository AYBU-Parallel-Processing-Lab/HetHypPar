#pragma once

#include "hhp_common.h"
#include <cusparse.h>
#include <cublas_v2.h>
#include <driver_types.h>
#include "mpi.h"

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("cuSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUBLAS(func)                                                     \
{                                                                              \
    cublasStatus_t status = (func);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
        printf("CUBLAS API failed at line %d with error: %d\n",                \
               __LINE__, status);                                              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#if defined(NDEBUG)
#   define PRINT_INFO(var)
#else
#   define PRINT_INFO(var) printf("  " #var ": %f\n", var);
#endif


cusparseStatus_t device_csr_create(CSR in, Device_CSR* out);
cusparseStatus_t device_csr_destroy(Device_CSR* in);
cusparseStatus_t device_csc_create(CSC in, Device_CSC* out);
cusparseStatus_t device_csc_destroy(Device_CSC* in);
cusparseStatus_t device_buffer_spmv_create(cusparseHandle_t handle, cusparseSpMatDescr_t mat, Device_Vector X, Device_Vector Y, const double* alpha, const double* beta, Device_Buffer_SpMV* out);
cusparseStatus_t device_buffer_spmv_destroy(Device_Buffer_SpMV* in);
cusparseStatus_t device_vector_init(size_t size, Device_Vector* out);
cusparseStatus_t device_vector_destroy(Device_Vector* in);
cusparseStatus_t device_vector_zero(Device_Vector vec);
cublasStatus_t device_vector_dot(cublasHandle_t cublasHande, Device_Vector v1, Device_Vector v2, double *out);
cublasStatus_t MPI_device_vector_dot(cublasHandle_t handle, Device_Vector vecA, Device_Vector vecB, double *out, MPI_Comm comm);
cublasStatus_t device_vector_axpy(cublasHandle_t cublasHandle, Device_Vector v1, const double a, Device_Vector out);
cublasStatus_t device_vector_scale(cublasHandle_t cublasHande, const double alpha, Device_Vector v);
cusparseStatus_t device_vector_toGPU(Vector src, Device_Vector dest);
cusparseStatus_t device_vector_toCPU(Device_Vector src, Vector dest);
cudaError_t device_vector_GPUtoGPU(Device_Vector src, Device_Vector dst);
cusparseStatus_t device_csr_spmv(cusparseHandle_t handle, Device_CSR mat, Device_Vector X, Device_Vector Y, const double alpha, const double beta, Device_Buffer_SpMV buf);
cusparseStatus_t device_csc_spmv(cusparseHandle_t handle, Device_CSC mat, Device_Vector X, Device_Vector Y, const double alpha, const double beta, Device_Buffer_SpMV buf);
cusparseStatus_t MPI_device_SHARD_CSC_mpi_spmxv(Device_SHARD_CSC A, Vector X, Device_Vector dX, Device_Vector dX_shr, Device_Vector Y, MPI_Comm comm, cusparseHandle_t handle, const double alpha, const double beta, Device_Buffer_SpMV locbuf, Device_Buffer_SpMV shrbuf);