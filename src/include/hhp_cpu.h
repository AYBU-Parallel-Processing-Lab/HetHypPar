#pragma once

#include "hhp_common.h"
#include "mpi.h"

#ifdef __cplusplus
extern "C" {
#endif
Vector vector_init(unsigned int nvals);
iVector ivector_init(unsigned int nvals);
Vector vector_init_const(unsigned int nvals, const double val);
Vector vector_init_clone(Vector vecA);
double vector_dot_seq(Vector vecA, Vector vecB);
void vector_zero(Vector vec);
void vector_scale_seq(Vector vecA, double scalar, Vector out);
void vector_add_seq(Vector vecA, Vector vecB, Vector out);
void vector_sub_seq(Vector vecA, Vector vecB, Vector out);
void vector_mul_seq(Vector vecA, Vector vecB, Vector out);
void vector_div_seq(Vector vecA, Vector vecB, Vector out);
void vector_destroy(Vector* vec);
void ivector_destroy(iVector* vec);

void vector_write(char* fileName, Vector vec);
Vector vector_read(char* fileName, int size);

iVector MPI_ivector_read_bcast(char* fileName, int size) ;
int MPI_ivector_read_scatter(char* fileName, int size);
Vector MPI_vector_read_parted(char* fileName, int size, iVector partvec);
double MPI_vector_dot(Vector vecA, Vector vecB, MPI_Comm comm);
void MPI_vector_gather(Vector loc, Vector big, iVector gind, int root, MPI_Comm comm);

void CSR_spmxv_seq(CSR A, Vector X, Vector out);
void CSR_spmxv_seq_acc(CSR A, Vector X, Vector Y);
void MPI_SHARD_CSC_mpi_spmxv_seq(SHARD_CSC A, Vector X, Vector Y, MPI_Comm comm, SpMV_Profile *prof);
#ifdef __cplusplus
}
#endif