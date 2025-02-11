#pragma once

#include "hhp_common.h"

Vector vector_init(unsigned int nvals);
Vector vector_init_clone(Vector vecA);
double vector_dot_seq(Vector vecA, Vector vecB);
void vector_zero(Vector vec);
void vector_scale_seq_inplace(Vector vecA, double scalar);
void vector_add_seq_inplace(Vector vecA, Vector vecB);
void vector_sub_seq_inplace(Vector vecA, Vector vecB);
void vector_mul_seq_inplace(Vector vecA, Vector vecB);
void vector_div_seq_inplace(Vector vecA, Vector vecB);
void vector_destroy(Vector* vec);

void CSR_spmxv_seq(CSR A, Vector X, Vector Y);
void CSR_spmxv_seq_acc(CSR A, Vector X, Vector Y);
double SPLIT_CSR_spmxv_cpu(SPLIT_CSR in, Vector Xl, Vector Xs, Vector Y);