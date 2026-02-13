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
#include "cublas_v2.h"

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

// X and Y vectors has to be initialized
cusparseStatus_t device_csc_create(CSC in, Device_CSC* out){
    Device_CSC res = {
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
    CHECK_CUDA( cudaMalloc((void**) &(res.data.I), (in.n + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &(res.data.J),    in.nnz * sizeof(int)))
    CHECK_CUDA( cudaMalloc((void**) &(res.data.val),     in.nnz * sizeof(double)))
    
    // Transfer I, J, val to device
    CHECK_CUDA( cudaMemcpy(res.data.I, in.I,(in.n + 1) * sizeof(*in.I), cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(res.data.J, in.J,(in.nnz) * sizeof(*in.J), cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(res.data.val, in.val,(in.nnz) * sizeof(*in.val), cudaMemcpyHostToDevice) )

    CHECK_CUSPARSE(
        cusparseCreateCsc
        (&res.desc, in.m, in.n, in.nnz,
        res.data.I, res.data.J, res.data.val,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F
        )
    )

    *out = res;

    return CUSPARSE_STATUS_SUCCESS;
}


cusparseStatus_t device_csc_destroy(Device_CSC* in){
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


cusparseStatus_t device_buffer_spmv_create(cusparseHandle_t handle, cusparseSpMatDescr_t mat, Device_Vector X, Device_Vector Y, const double* alpha, const double* beta, Device_Buffer_SpMV* out){
    // Y = α * (matrix * X) + β * Y
    size_t buffersize = 0;
    CHECK_CUSPARSE(
        cusparseSpMV_bufferSize(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            alpha, mat, X.desc, beta, Y.desc, CUDA_R_64F,
            CUSPARSE_SPMV_ALG_DEFAULT, &buffersize
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

cudaError_t device_vector_GPUtoGPU(Device_Vector src, Device_Vector dst){
    return cudaMemcpy(dst.vals, src.vals, src.nvals*sizeof(*(src.vals)), cudaMemcpyDeviceToDevice);
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

cublasStatus_t MPI_device_vector_dot(cublasHandle_t handle, Device_Vector vecA, Device_Vector vecB, double *out, MPI_Comm comm) {
    double ldot;
    CHECK_CUBLAS(device_vector_dot(handle, vecA, vecB, &ldot))
    double gdot = 0.0;
    MPI_CHECK(MPI_Allreduce(&ldot, &gdot, 1, MPI_DOUBLE, MPI_SUM, comm));
    *out = gdot;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t device_vector_dot(cublasHandle_t cublasHande, Device_Vector v1, Device_Vector v2, double *out){
    return cublasDdot(cublasHande, v1.nvals, v1.vals, 1, v2.vals, 1, out);
}
cublasStatus_t device_vector_scale(cublasHandle_t cublasHande, const double alpha, Device_Vector v){
    return cublasDscal(cublasHande, v.nvals, &alpha, v.vals, 1);
}

cublasStatus_t device_vector_axpy(cublasHandle_t cublasHandle, Device_Vector v1, const double a, Device_Vector out){
    return cublasDaxpy(cublasHandle, v1.nvals, &a, v1.vals, 1, out.vals, 1);
}

cusparseStatus_t device_vector_toGPU(Vector src, Device_Vector dest){
    CHECK_CUDA(cudaMemcpy(dest.vals, src.vals, src.nvals*(sizeof(*src.vals)), cudaMemcpyHostToDevice))
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t device_vector_toCPU(Device_Vector src, Vector dest){
    CHECK_CUDA(cudaMemcpy(dest.vals, src.vals, src.nvals*(sizeof(*src.vals)), cudaMemcpyDeviceToHost))
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t device_csc_spmv(cusparseHandle_t handle, Device_CSC mat, Device_Vector X, Device_Vector Y, const double alpha, const double beta, Device_Buffer_SpMV buf){
    CHECK_CUSPARSE(
        cusparseSpMV(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, mat.desc, X.desc, &beta, Y.desc,
            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buf
        )
    )
    CHECK_CUDA(cudaDeviceSynchronize())
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t device_csr_spmv(cusparseHandle_t handle, Device_CSR mat, Device_Vector X, Device_Vector Y, const double alpha, const double beta, Device_Buffer_SpMV buf){
    CHECK_CUSPARSE(
        cusparseSpMV(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, mat.desc, X.desc, &beta, Y.desc,
            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buf
        )
    )
    CHECK_CUDA(cudaDeviceSynchronize())
    return CUSPARSE_STATUS_SUCCESS;
}

/*
 * NOTES:
 *  - A is row partitioned,
 *  - X is of length A.loc.n
 *  - Y is of length A.loc.m
 *  - Y must be zeroed before this function
 */
cusparseStatus_t MPI_device_SHARD_CSC_mpi_spmxv(Device_SHARD_CSC A, Vector X, Device_Vector dX, Device_Vector dX_shr, Device_Vector Y, MPI_Comm comm, cusparseHandle_t handle, const double alpha, const double beta, Device_Buffer_SpMV locbuf, Device_Buffer_SpMV shrbuf) {

    int mpi_rank;
    int mpi_size;
    MPI_CHECK(MPI_Comm_rank(comm, &mpi_rank));
    MPI_CHECK(MPI_Comm_size(comm, &mpi_size));


    // // TODO: REMOVE AFTER DEBUG
    // char fpath[256] = {};
    // sprintf(fpath, "DEBUG_PRINT_Xloc_%d.txt",mpi_rank);
    // FILE* f = fopen(fpath, "w");
    // print_vector(f, X);
    // fclose(f);

    CHECK_CUSPARSE(device_vector_zero(Y))
    CHECK_CUSPARSE(device_vector_toCPU(dX, X))
    
    // [x] IS FREED?
    MPI_Request *recv_reqs;
    ALLOC_ARRAY(recv_reqs, A.recv.num);
    // [x] IS FREED?
    MPI_Request *send_reqs;
    ALLOC_ARRAY(send_reqs, A.send.num);

    {   // COMMUNICATE BETWEEN PROCESSES
        
        // fill send buffer
        int nsend = A.send.I[A.send.num];
        for (size_t i = 0; i < nsend; i++)
        {
            int idx = A.send.J[i];
            A.send.val[i] = X.vals[idx];
        }

        // ISSUE SENDS
        for (size_t i = 0; i < A.send.num; i++)
        {
            int js = A.send.I[i];
            int je = A.send.I[i+1];
            int recipient = A.send.ranks[i];
            int nsend = je - js;

            MPI_CHECK( MPI_Isend( A.send.val + js, nsend, MPI_DOUBLE, recipient, 0, comm, send_reqs + i) );
        }

        // ISSUE RECVS
        for (size_t i = 0; i < A.recv.num; i++)
        {
            int js = A.recv.I[i];
            int je = A.recv.I[i+1];
            int sender = A.recv.ranks[i];
            int nrecv = je - js;

            MPI_CHECK(MPI_Irecv(A.recv.val + js, nrecv, MPI_DOUBLE, sender, 0, comm, recv_reqs + i));
        }
    }


    {   // LOCAL SpMxV
        CHECK_CUSPARSE(device_csc_spmv(handle, A.loc, dX, Y, alpha, beta, locbuf))
    }


    // // TODO: REMOVE AFTER DEBUG
    // sprintf(fpath, "DEBUG_PRINT_Ylocloc_%d.txt",mpi_rank);
    // f = fopen(fpath, "w");
    // print_vector(f, Y);
    // fclose(f);

    // for (size_t received=0; received < A.recv.num; received++)
    // {   // SHARED SpMxV

    //     int recv_idx;
    //     MPI_CHECK(MPI_Waitany(A.recv.num, recv_reqs, &recv_idx, MPI_STATUS_IGNORE));

    //     // SpMxV on the Received columns
    //     int is = A.recv.I[recv_idx];
    //     int ie = A.recv.I[recv_idx+1];
    //     for (size_t i = is; i < ie; i++)
    //     {
    //         int ii = A.recv.J[i];
    //         double ival = A.recv.val[i];
    //         for (size_t j = A.shr.I[ii]; j < A.shr.I[ii + 1]; j++)
    //         {
    //             Y.vals[A.shr.J[j]] += A.shr.val[j]*ival;
    //         }
    //     }
    // }
    { // Shared SpMxV
        MPI_CHECK(MPI_Waitall(A.recv.num, recv_reqs, MPI_STATUSES_IGNORE));
        Vector temp = {
            .nvals = A.shr.data.n,
            .vals = A.recv.val
        };
        CHECK_CUSPARSE(device_vector_toGPU(temp, dX_shr))
        CHECK_CUSPARSE(device_csc_spmv(handle, A.shr, dX_shr, Y, alpha, alpha, shrbuf))
    }
    
    FREE_AND_NULL(recv_reqs);
    
    MPI_CHECK(MPI_Waitall(A.send.num, send_reqs, MPI_STATUSES_IGNORE));
    FREE_AND_NULL(send_reqs);

    // // TODO: REMOVE AFTER DEBUG
    // sprintf(fpath, "DEBUG_PRINT_Yloc_%d.txt",mpi_rank);
    // f = fopen(fpath, "w");
    // print_vector(f, Y);
    // fclose(f);

    return CUSPARSE_STATUS_SUCCESS;
}