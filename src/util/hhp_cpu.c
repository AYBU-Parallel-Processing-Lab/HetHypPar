#include "omp.h"
#include <math.h>
#include <mpi.h>
#include <stdio.h>               // printf
#include <stdlib.h>              // EXIT_FAILURE
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "mkl_spblas.h"
#include "hhp_common.h"
#include "hhp_util.h"
#include "hhp_cpu.h"

iVector ivector_init(unsigned int nvals){
    iVector res = {
        .nvals = nvals
    };
    if (nvals <= 0)
        ABORT("Attempted to create a vector of size %d.", nvals)
    ALLOC_ARRAY(res.vals, nvals);
    if (!res.vals)
        ABORT("Vector init failed, malloc returned NULL")
    return res;
}

Vector vector_init(unsigned int nvals){
    Vector res = {
        .nvals = nvals
    };
    if (nvals <= 0)
        ABORT("Attempted to create a vector of size %d.", nvals)
    ALLOC_ARRAY(res.vals, nvals);
    if (!res.vals)
        ABORT("Vector init failed, malloc returned NULL")
    return res;
}

Vector vector_init_const(unsigned int nvals, const double val){
    Vector res = vector_init(nvals);
    for (size_t i = 0; i < res.nvals; i++)
    {
        res.vals[i] = val;
    }
    return res;
}

// TODO: Test this
Vector vector_init_clone(Vector vecA){
    Vector res = vector_init(vecA.nvals);
    memcpy(res.vals, vecA.vals, sizeof(*vecA.vals)*vecA.nvals);

    return res;
}

void vector_destroy(Vector* vec){
    if ((vec->nvals <= 0 ) || !vec->vals){
        ERRORLOG("Attempted to free invalid Vector")
        return;
    }
    FREE_AND_NULL_IF(vec->vals);
    vec->nvals = 0;
}

void ivector_destroy(iVector* vec){
    if ((vec->nvals <= 0 ) || !vec->vals){
        ERRORLOG("Attempted to free invalid Vector")
        return;
    }
    FREE_AND_NULL_IF(vec->vals);
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
double vector_dot_seq(Vector vecA, Vector vecB){
    double res = 0.0;
    for (size_t i = 0; i < vecA.nvals; i++){
        res += vecA.vals[i]*vecB.vals[i];
    }
    return res;
}

// TODO: Test this
void vector_scale_seq(Vector vecA, double scalar, Vector out){
    for (size_t i=0; i< vecA.nvals; i++){
        out.vals[i] = vecA.vals[i]*scalar;
    }
}

// TODO: Test this
void vector_add_seq(Vector vecA, Vector vecB, Vector out){
    for (size_t i = 0; i < vecA.nvals; i++)
    {
        out.vals[i] = vecA.vals[i] + vecB.vals[i];
    }
}

// TODO: Test this
void vector_sub_seq(Vector vecA, Vector vecB, Vector out){
    for (size_t i = 0; i < vecA.nvals; i++)
    {
        out.vals[i] = vecA.vals[i] - vecB.vals[i];
    }
}

// TODO: Test this
void vector_mul_seq(Vector vecA, Vector vecB, Vector out){
    for (size_t i = 0; i < vecA.nvals; i++)
    {
        out.vals[i] = vecA.vals[i] * vecB.vals[i];
    }
}

// TODO: Test this
void vector_div_seq(Vector vecA, Vector vecB, Vector out){
    for (size_t i = 0; i < vecA.nvals; i++)
    {
        out.vals[i] = vecA.vals[i] / vecB.vals[i];
    }
}

int MPI_ivector_read_scatter(char* fileName, int size) {
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    int out;

    if (mpi_rank == 0) {
        
        iVector big = ivector_init(size);

        FILE* f = fopen(fileName, "r");
        if (f == NULL)
            MPI_ABORT("Could not open file: %s", fileName)

        int i=0;
        while(i<size && fscanf(f, "%d",&big.vals[i++]) == 1);

        fclose(f);
    
        if (i < size) 
            MPI_ABORT("File (%s) is smaller than expected (%d)",fileName, size)

        MPI_CHECK(MPI_Scatter(big.vals, 1, MPI_INT, &out, 1, MPI_INT, 0, MPI_COMM_WORLD));

        ivector_destroy(&big);
    }else{
        MPI_CHECK(MPI_Scatter(NULL, 1, MPI_INT, &out, 1, MPI_INT, 0, MPI_COMM_WORLD));
    }
    return out;
}


iVector MPI_ivector_read_bcast(char* fileName, int size) {
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    iVector vec = ivector_init(size);

    // Master node reads the file
    if (mpi_rank == 0) {
        FILE* f = fopen(fileName, "r");
        if (f == NULL)
            MPI_ABORT("Could not open file: %s", fileName)

        int i=0;
        while(i<size && fscanf(f, "%d",&vec.vals[i++]) == 1);

        fclose(f);
    
        if (i < size) 
            MPI_ABORT("File (%s) is smaller than expected (%d)",fileName, size)
    }

    // Broadcast vector from master to all other nodes
    MPI_Bcast(vec.vals, size, MPI_INT, 0, MPI_COMM_WORLD);

    return vec;
}

double MPI_vector_dot(Vector vecA, Vector vecB, MPI_Comm comm) {
    double ldot = vector_dot_seq(vecA, vecB);
    double gdot = 0.0;
    MPI_CHECK(MPI_Allreduce(&ldot, &gdot, 1, MPI_DOUBLE, MPI_SUM, comm));
    return gdot;
}

void MPI_vector_gather(Vector loc, Vector big, iVector gind, int root, MPI_Comm comm){
    int mpi_rank, mpi_size;
    MPI_CHECK(MPI_Comm_rank(comm, &mpi_rank));
    MPI_CHECK(MPI_Comm_size(comm, &mpi_size));

    iVector recvcounts = {};

    if (mpi_rank == root){
        recvcounts = ivector_init(mpi_size);
        for (size_t i = 0; i < recvcounts.nvals; i++)
            recvcounts.vals[i] = -999;
    }

    MPI_CHECK(MPI_Gather(&loc.nvals, 1, MPI_INT, recvcounts.vals, 1, MPI_INT, root, comm));

    if (mpi_rank == root)
    {
        // [x] IS FREED?
        // iVector recvcounts = ivector_init(mpi_size);
        // [x] IS FREED?
        iVector displs = ivector_init(mpi_size);

        // fill out displs
        displs.vals[0] = 0;
        for (size_t i = 1; i < mpi_size; i++)
            displs.vals[i] = displs.vals[i-1] + recvcounts.vals[i-1];

        // [x] IS FREED?
        Vector temp_big = vector_init(big.nvals);
        
        MPI_CHECK(MPI_Gatherv(loc.vals, loc.nvals, MPI_DOUBLE, temp_big.vals, recvcounts.vals, displs.vals, MPI_DOUBLE, root, comm));

        // [x] IS FREED?
        iVector gmap = ivector_init(big.nvals);

        MPI_CHECK(MPI_Gatherv(gind.vals, loc.nvals, MPI_INT, gmap.vals, recvcounts.vals, displs.vals, MPI_INT, root, comm));
        
        ivector_destroy(&recvcounts);
        ivector_destroy(&displs);
        
        for (size_t i = 0; i < big.nvals; i++)
            big.vals[gmap.vals[i]] = temp_big.vals[i];

        vector_destroy(&temp_big);
        ivector_destroy(&gmap);
    } else {
        // MPI_CHECK(MPI_Gather(&(loc.nvals), 1, MPI_INT, NULL, mpi_size, MPI_INT, root, comm));
        MPI_CHECK(MPI_Gatherv(loc.vals, loc.nvals, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, root, comm));
        MPI_CHECK(MPI_Gatherv(gind.vals, loc.nvals, MPI_INT, NULL, NULL, NULL, MPI_INT, root, comm));
    }
    return;
}

Vector MPI_vector_read_parted(char* fileName, int size, iVector partvec) {
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    Vector big_vec = {};
    Vector res = {};

    if (mpi_rank == 0) {
        // note that size is equal to partvec.nvals
        big_vec = vector_init(size);

        FILE* f = fopen(fileName, "r");
        if (f == NULL)
            MPI_ABORT("Could not open file: %s", fileName)
        
        int i=0;
        while(i < size && fscanf(f, "%lf", &big_vec.vals[i++]) == 1);
    
        fclose(f);
    
        if (i < size) 
            MPI_ABORT("File (%s) is smaller than expected (%d)", fileName, size)
    }

    // Calculate how many elements each rank will receive
    iVector sendcounts = ivector_init(mpi_size);  // Keep track of counts per rank
    for (size_t i = 0; i < mpi_size; i++)
        sendcounts.vals[i] = 0;
        
    for (size_t i = 0; i < partvec.nvals; i++)
        sendcounts.vals[partvec.vals[i]]++;

    // Create displacement array for MPI_Scatterv
    iVector displs = ivector_init(mpi_size);
    displs.vals[0] = 0;
    for (size_t i = 1; i < mpi_size; i++)
        displs.vals[i] = displs.vals[i-1] + sendcounts.vals[i-1];

    // Initialize the local vector for each rank
    int local_size = sendcounts.vals[mpi_rank];
    res = vector_init(local_size);
    
    // Use MPI_Scatterv to distribute data
    if (mpi_rank == 0) {
        // Create a temporary vector for reordering the data
        Vector temp_buffer = vector_init(size);
        
        // Reset sendcounts to use as current position trackers
        iVector current_pos = ivector_init(mpi_size);
        for (size_t i = 0; i < mpi_size; i++)
            current_pos.vals[i] = 0;
        
        for (size_t i = 0; i < partvec.nvals; i++)
        {
            int id = partvec.vals[i];
            int pos = displs.vals[id] + current_pos.vals[id];
            temp_buffer.vals[pos] = big_vec.vals[i];
            current_pos.vals[id]++;
        }
        
        // Use MPI_Scatterv to distribute the data
        MPI_Scatterv(temp_buffer.vals, sendcounts.vals, displs.vals, MPI_DOUBLE, 
                    res.vals, local_size, MPI_DOUBLE, 
                    0, MPI_COMM_WORLD);
        
        vector_destroy(&temp_buffer);
        ivector_destroy(&current_pos);
    } else {
        // Receive data
        MPI_Scatterv(NULL, NULL, NULL, MPI_DOUBLE, 
                    res.vals, local_size, MPI_DOUBLE, 
                    0, MPI_COMM_WORLD);
    }
    
    // Clean up
    if (mpi_rank == 0)
        vector_destroy(&big_vec);
    ivector_destroy(&sendcounts);
    ivector_destroy(&displs);
    
    return res;
}

// Vector MPI_vector_read_parted(char* fileName, int size, iVector partvec) {
//     int mpi_rank, mpi_size;
//     MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

//     Vector big_vec = {};
//     Vector res = {};

//     if (mpi_rank == 0) {
//         // note that size is equal to partvec.nvals
//         big_vec = vector_init(size);

//         FILE* f = fopen(fileName, "r");
//         if (f == NULL)
//             MPI_ABORT("Could not open file: %s", fileName)
        
//         int i=0;
//         while(i < size && fscanf(f, "%lf", &big_vec.vals[i++]) == 1);
    
//         fclose(f);
    
//         if (i < size) 
//             MPI_ABORT("File (%s) is smaller than expected (%d)", fileName, size)
//     }

//     // Calculate how many elements each rank will receive
//     iVector cum_size = ivector_init(mpi_size + 1);
//     cum_size.vals[0] = 0;
//     for (size_t i = 0; i < partvec.nvals; i++)
//         cum_size.vals[partvec.vals[i] + 1]++;

//     for (size_t i = 1; i < cum_size.nvals; i++)
//         cum_size.vals[i] += cum_size.vals[i-1];

//     // Initialize the local vector for each rank
//     int local_size = cum_size.vals[mpi_rank + 1] - cum_size.vals[mpi_rank];
//     res = vector_init(local_size);
    
//     // Create iVectors to help with scattering
    
//     // Use MPI_Scatterv to distribute data
//     if (mpi_rank == 0) {
//         iVector sendcounts = ivector_init(mpi_size);
//         // Create a temporary vector for reordering the data
//         Vector temp_buffer = vector_init(size);
        
//         for (size_t i = 0; i < partvec.nvals; i++)
//         {
//             int id = partvec.vals[i];
//             temp_buffer.vals[cum_size.vals[id] + sendcounts.vals[id]] = big_vec.vals[i];
//             sendcounts.vals[id]++;
//         }
        
//         // Use MPI_Scatterv to distribute the data
//         MPI_Scatterv(temp_buffer.vals, sendcounts.vals, cum_size.vals, MPI_DOUBLE, 
//                     res.vals, local_size, MPI_DOUBLE, 
//                     0, MPI_COMM_WORLD);
        
//         vector_destroy(&temp_buffer);
//         ivector_destroy(&sendcounts);
//     } else {
//         // Receive data
//         MPI_Scatterv(NULL, NULL, NULL, MPI_DOUBLE, 
//                     res.vals, local_size, MPI_DOUBLE, 
//                     0, MPI_COMM_WORLD);
//     }
    
//     // Clean up
//     if (mpi_rank == 0)
//         vector_destroy(&big_vec);
//     ivector_destroy(&cum_size);
    
//     return res;
// }

Vector vector_read(char* fileName, int size) {

    Vector res = vector_init(size);

    FILE* f = fopen(fileName,"r") ;
    if (f == NULL)
        ABORT("Could not open file: %s", fileName)
    
    int i=0;
    while( i<size && fscanf(f, "%lf",&res.vals[i++]) == 1 );

    fclose(f);

    if (i < size && !feof(f)) 
        fprintf(stderr, "Error reading file\n");

    return res;
}

void vector_write(char* fileName, Vector vec){
    FILE* f = fopen(fileName, "w");
    if (f == NULL)
        ABORT("Couldn't write to file %s", fileName)

    for (size_t i = 0; i < vec.nvals; i++)
    {
        int res = fprintf(f, "%lf\n",vec.vals[i]);
        if (res < 0)
            ABORT("failure to fprintf failed to write")
    }
    fclose(f);
}

// TODO: Test this
void CSR_spmxv_seq(CSR A, Vector X, Vector out){

    // #pragma omp distribute parallel for simd
    for (size_t i = 0; i < A.m; i++) {
        double res = 0.0;
        for (size_t j=A.I[i]; j < A.I[i+1]; j++) {
            res += A.val[j]*X.vals[A.J[j]];
        }
        out.vals[i] = res;
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

// TODO: Test this
void CSC_spmxv_seq_acc(CSC A, Vector X, Vector Y){
    
    // #pragma omp distribute parallel for simd
    for (size_t i = 0; i < A.n; i++) {
        for (size_t j=A.I[i]; j < A.I[i+1]; j++) {
            Y.vals[A.J[j]] += A.val[j]*X.vals[i];
        }
    }
}

void print_vector(FILE *f, Vector v){
  fprintf(f,"\n==========\nVECTOR PRINT\n========\n");
  
  fprintf(f,"nval = %d\n",v.nvals);
  fprintf(f,"val = ");
  for (size_t i = 0; i < v.nvals; i++)
  {
    fprintf(f," %lf",v.vals[i]);
  }
  fprintf(f,"\n");

  fprintf(f,"\n=========================\n");
}

/*
 * NOTES:
 *  - A is row partitioned,
 *  - X is of length A.loc.n
 *  - Y is of length A.loc.m
 *  - Y must be zeroed before this function
 */
void MPI_SHARD_CSC_mpi_spmxv_seq(SHARD_CSC A, Vector X, Vector Y, MPI_Comm comm) {

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

    vector_zero(Y);
    
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
        CSC_spmxv_seq_acc(A.loc, X, Y);
    }


    // // TODO: REMOVE AFTER DEBUG
    // sprintf(fpath, "DEBUG_PRINT_Ylocloc_%d.txt",mpi_rank);
    // f = fopen(fpath, "w");
    // print_vector(f, Y);
    // fclose(f);

    for (size_t received=0; received < A.recv.num; received++)
    {   // SHARED SpMxV

        int recv_idx;
        MPI_CHECK(MPI_Waitany(A.recv.num, recv_reqs, &recv_idx, MPI_STATUS_IGNORE));

        // SpMxV on the Received columns
        int is = A.recv.I[recv_idx];
        int ie = A.recv.I[recv_idx+1];
        for (size_t i = is; i < ie; i++)
        {
            int ii = A.recv.J[i];
            double ival = A.recv.val[i];
            for (size_t j = A.shr.I[ii]; j < A.shr.I[ii + 1]; j++)
            {
                Y.vals[A.shr.J[j]] += A.shr.val[j]*ival;
            }
        }
    }
    FREE_AND_NULL(recv_reqs);
    
    MPI_CHECK(MPI_Waitall(A.send.num, send_reqs, MPI_STATUSES_IGNORE));
    FREE_AND_NULL(send_reqs);

    // // TODO: REMOVE AFTER DEBUG
    // sprintf(fpath, "DEBUG_PRINT_Yloc_%d.txt",mpi_rank);
    // f = fopen(fpath, "w");
    // print_vector(f, Y);
    // fclose(f);

    return;
}

// double spmxv_cpu_mkl(SHARD_CSR in){
//     double start, end ;

//     double *X, *Y;
//     ALLOC_ARRAY(X, in.loc.n + in.shr.n);
//     CALLOC_ARRAY(Y, in.loc.m);

//     for (size_t i=0; i<in.loc.n; i++) X[i] = in.locp[i] +1;
//     for (size_t i=0; i<in.shr.n; i++) X[i + in.loc.n] = in.shrp[i] + 1;

//     sparse_matrix_t loc, shr;
//     sparse_status_t status;

//     status = mkl_sparse_d_create_csr(&loc, SPARSE_INDEX_BASE_ZERO,
//                                     in.loc.m, in.loc.n,
//                                     in.loc.I, in.loc.I+1,
//                                     in.loc.J, in.loc.val);

//     if (status != SPARSE_STATUS_SUCCESS){
//         return 0.0;
//     // ABORT("MKL FAILED TO CREATE LOCAL MATRIX WITH ERROR %d", status)
//     } 

//     if(in.shr.n != 0){
//         status = mkl_sparse_d_create_csr(&shr, SPARSE_INDEX_BASE_ZERO,
//                                     in.shr.m, in.shr.n,
//                                     in.shr.I, in.shr.I+1,
//                                     in.shr.J, in.shr.val);

//         if (status != SPARSE_STATUS_SUCCESS) ABORT("MKL FAILED TO CREATE SHARED MATRIX WITH ERROR %d", status)
//     }


//     struct matrix_descr descr = {
//         SPARSE_MATRIX_TYPE_GENERAL,
//         // SPARSE_FILL_MODE_FULL,
//         // SPARSE_DIAG_NON_UNIT
//     };
//     start = omp_get_wtime();

//     // start = omp_get_wtime();
//     mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, loc, descr, X, 0.0, Y);
//     if(in.shr.n != 0)
//         mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, shr, descr, X+in.shr.n, 1.0, Y);

//     end = omp_get_wtime();

//     mkl_sparse_destroy(loc);
//     if(in.shr.n != 0)
//         mkl_sparse_destroy(shr);
//     FREE_AND_NULL(X);
//     FREE_AND_NULL(Y);

//     return (end - start);
// }