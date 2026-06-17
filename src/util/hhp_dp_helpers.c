#include "hhp_dp_helpers.h"
#include "hhp_util.h"   // ABORT, ALLOC_ARRAY, CALLOC_ARRAY

#include <cuda_runtime_api.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int read_ints(const char *path, int *out, int max) {
    FILE *f = fopen(path, "r");
    if (!f) ABORT("Could not open file: %s", path)
    int i = 0, v;
    while (i < max && fscanf(f, "%d", &v) == 1) out[i++] = v;
    fclose(f);
    return i;
}

CSR csr_permute(CSR A, const int *perm, const int *perm_inv) {
    CSR out = {.m = A.m, .n = A.n, .nnz = A.nnz};
    CALLOC_ARRAY(out.I, A.m + 1);
    ALLOC_ARRAY(out.J, A.nnz);
    ALLOC_ARRAY(out.val, A.nnz);
    out.I[0] = 0;
    for (int i = 0; i < A.m; i++) {
        int old_row = perm[i];
        out.I[i + 1] = out.I[i] + (A.I[old_row + 1] - A.I[old_row]);
    }
    for (int i = 0; i < A.m; i++) {
        int old_row = perm[i], old_start = A.I[old_row];
        int cnt = A.I[old_row + 1] - old_start, new_start = out.I[i];
        for (int k = 0; k < cnt; k++) {
            out.J[new_start + k] = perm_inv[A.J[old_start + k]];
            out.val[new_start + k] = A.val[old_start + k];
        }
    }
    return out;
}

CSR csr_row_slice(CSR A, int row_start, int row_end) {
    int m_new = row_end - row_start, offset = A.I[row_start];
    int nnz_new = A.I[row_end] - offset;
    CSR out = {.m = m_new, .n = A.n, .nnz = nnz_new};
    CALLOC_ARRAY(out.I, m_new + 1);
    ALLOC_ARRAY(out.J, nnz_new > 0 ? nnz_new : 1);
    ALLOC_ARRAY(out.val, nnz_new > 0 ? nnz_new : 1);
    for (int i = 0; i <= m_new; i++) out.I[i] = A.I[row_start + i] - offset;
    for (int k = 0; k < nnz_new; k++) {
        out.J[k] = A.J[offset + k];
        out.val[k] = A.val[offset + k];
    }
    return out;
}

double *dscalar(double init) {
    double *p;
    if (cudaMalloc((void **)&p, sizeof(double)) != cudaSuccess) ABORT("cudaMalloc scalar")
    if (cudaMemcpy(p, &init, sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) ABORT("cudaMemcpy scalar")
    return p;
}

double *dvec(int n) {
    double *p;
    if (cudaMalloc((void **)&p, (n > 0 ? n : 1) * sizeof(double)) != cudaSuccess) ABORT("cudaMalloc vec")
    return p;
}

double host_dot(const double *a, const double *b, int n) {
    double s = 0.0;
    #pragma omp parallel for reduction(+:s)
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}
void host_axpy(double alpha, const double *x, double *y, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) y[i] += alpha * x[i];
}
void host_scal(double alpha, double *x, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) x[i] *= alpha;
}
void host_copy(const double *x, double *y, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) y[i] = x[i];
}
