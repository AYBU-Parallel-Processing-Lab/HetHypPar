#ifndef HHP_DP_HELPERS_H
#define HHP_DP_HELPERS_H

// Shared helpers for the device-pointer BiCGStab solvers (bicgstab-gpu-dp,
// bicgstab-hybrid-async-dp, bicgstab-hybrid-dist-dp). These were previously
// duplicated as static functions inside each entry file; consolidated here.

#include "hhp_common.h"   // CSR

// Read up to `max` whitespace-separated ints from `path` into `out`; returns
// the count read. Aborts if the file cannot be opened.
int read_ints(const char *path, int *out, int max);

// Permute rows of A by `perm` and renumber columns by `perm_inv`:
// out[i,j] = A[perm[i], perm[j]]. Allocates; free with freeSparseMatrix.
CSR csr_permute(CSR A, const int *perm, const int *perm_inv);

// Extract contiguous rows [row_start, row_end) as a CSR slice (global column
// indices preserved). Allocates; free with freeSparseMatrix.
CSR csr_row_slice(CSR A, int row_start, int row_end);

// Allocate one device double initialised from a host value. Aborts on failure.
double *dscalar(double init);

// Allocate n device doubles (rounds n<=0 up to 1). Aborts on failure.
double *dvec(int n);

// OpenMP host vector ops on raw arrays of length n (n<=0 is a no-op).
double host_dot(const double *a, const double *b, int n);   // returns sum a[i]*b[i]
void   host_axpy(double alpha, const double *x, double *y, int n); // y += alpha*x
void   host_scal(double alpha, double *x, int n);                  // x *= alpha
void   host_copy(const double *x, double *y, int n);              // y = x

#endif // HHP_DP_HELPERS_H
