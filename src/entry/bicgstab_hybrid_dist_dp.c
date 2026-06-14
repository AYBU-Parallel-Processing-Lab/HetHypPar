#include "stdio.h"
#include <math.h>
#include <omp.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "hhp_common.h"
#include "hhp_matrix.h"
#include "hhp_cuda.h"
#include "hhp_cpu.h"
#include "hhp_util.h"
#include "hhp_dp_kernels.h"

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

#include "argp.h"
#include "unistd.h"

// Single-process GPU+CPU hybrid BiCGStab -- NO MPI -- that DISTRIBUTES every
// operation (dots and vector ops, not just SpMV) between CPU and GPU, with
// cuBLAS device-pointer mode on the GPU side.
//
// Rows are permuted GPU-first; the GPU owns rows [0,ng) and the CPU owns
// [ng,n). Every BiCGStab vector is split: a device array (length ng) and a host
// array (length nc=n-ng). Element-wise ops (axpy/scal/copy) run independently on
// each side -- no communication. Dot products become GPU_partial + CPU_partial:
// the GPU partial is a dp-mode cublasDdot into a device scalar, the CPU partial
// an OpenMP reduction; the CPU partial is pushed to the device and the two are
// summed in a tiny kernel so the GPU critical path never leaves the device. The
// resulting scalars are copied back D->H only for the CPU's own vector ops.
// SpMV needs the full input vector, so P/S are gathered into a full device and a
// full host buffer each SpMV (D->H of the GPU slice, H->D of the CPU slice),
// then GPU and CPU each multiply their row slice concurrently.
//
// NOTE: distributing a reduction forces a CPU<->GPU scalar rendezvous every dot
// -- the very sync dp removes for a GPU-only dot. This solver exists to MEASURE
// that cost vs bicgstab-hybrid-async-dp (SpMV-only split). Expect it to be
// slower; the dp/overlap machinery only minimizes the unavoidable rendezvous.

struct arguments {
    char *input_matrix, *input_x, *input_y, *input_part, *input_gpu, *output_x;
    int n_iters;
};

#define OPT_INPUT_MATRIX 'm'
#define OPT_OUTPUT 'o'
#define OPT_INPUT_X 'x'
#define OPT_INPUT_Y 'y'
#define OPT_INPUT_PART 'p'
#define OPT_INPUT_GPU 'g'
#define OPT_N_ITERS 'n'

static struct argp_option options[] = {
    {"input-matrix", OPT_INPUT_MATRIX, "FILE", 0, "Input matrix market file"},
    {"input-x", OPT_INPUT_X, "FILE", 0, "Input X (initial guess) vector file"},
    {"input-y", OPT_INPUT_Y, "FILE", 0, "Target B vector file"},
    {"input-part", OPT_INPUT_PART, "FILE", 0, "2-rank partition vector file"},
    {"is-gpu", OPT_INPUT_GPU, "FILE", 0, "is_gpu file (2 lines, exactly one '1')"},
    {"output-x", OPT_OUTPUT, "FILE", 0, "Output X (solution) vector file"},
    {"n-iters", OPT_N_ITERS, "POSITIVE-INTEGER", 0, "BiCGStab iterations"},
    {0}
};

static error_t parse_opt(int key, char *arg, struct argp_state *state) {
    struct arguments *a = state->input;
    char *buf = NULL;
    switch (key) {
        case OPT_INPUT_MATRIX: a->input_matrix = arg; break;
        case OPT_INPUT_X:      a->input_x = arg; break;
        case OPT_INPUT_Y:      a->input_y = arg; break;
        case OPT_INPUT_PART:   a->input_part = arg; break;
        case OPT_INPUT_GPU:    a->input_gpu = arg; break;
        case OPT_OUTPUT:       a->output_x = arg; break;
        case OPT_N_ITERS:      a->n_iters = strtol(arg, &buf, 10); break;
        case ARGP_KEY_ARG:     return 0;
        default:               return ARGP_ERR_UNKNOWN;
    }
    return 0;
}

static char doc[] = "Fully-distributed GPU+CPU hybrid BiCGStab (no MPI), device-pointer dots";
static char args_doc[] = "";
static struct argp argp = {options, parse_opt, args_doc, doc};

// --- host helpers (OpenMP); nc==0 makes them no-ops ---
static double host_dot(const double *a, const double *b, int n) {
    double s = 0.0;
    #pragma omp parallel for reduction(+:s)
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}
static void host_axpy(double al, const double *x, double *y, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) y[i] += al * x[i];
}
static void host_scal(double al, double *x, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) x[i] *= al;
}
static void host_copy(const double *x, double *y, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) y[i] = x[i];
}

static int read_ints(const char *path, int *out, int max) {
    FILE *f = fopen(path, "r");
    if (!f) ABORT("Could not open file: %s", path)
    int i = 0, v;
    while (i < max && fscanf(f, "%d", &v) == 1) out[i++] = v;
    fclose(f);
    return i;
}

static double *dscalar(double init) {
    double *p;
    if (cudaMalloc((void **)&p, sizeof(double)) != cudaSuccess) ABORT("cudaMalloc scalar")
    if (cudaMemcpy(p, &init, sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) ABORT("cudaMemcpy scalar")
    return p;
}
static double *dvec(int n) {
    double *p;
    if (cudaMalloc((void **)&p, (n > 0 ? n : 1) * sizeof(double)) != cudaSuccess) ABORT("cudaMalloc vec")
    return p;
}

static CSR csr_permute(CSR A, const int *perm, const int *perm_inv) {
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

static CSR csr_row_slice(CSR A, int row_start, int row_end) {
    int m_new = row_end - row_start, offset = A.I[row_start];
    int nnz_new = A.I[row_end] - offset;
    CSR out = {.m = m_new, .n = A.n, .nnz = nnz_new};
    CALLOC_ARRAY(out.I, m_new + 1);
    ALLOC_ARRAY(out.J, nnz_new > 0 ? nnz_new : 1);
    ALLOC_ARRAY(out.val, nnz_new > 0 ? nnz_new : 1);
    for (int i = 0; i <= m_new; i++) out.I[i] = A.I[row_start + i] - offset;
    for (int k = 0; k < nnz_new; k++) { out.J[k] = A.J[offset + k]; out.val[k] = A.val[offset + k]; }
    return out;
}

// Globals bundling the split-vector / SpMV context (keeps the helper sigs sane).
typedef struct {
    cusparseHandle_t ch;
    cusparseSpMatDescr_t Agpu;          // ng x n on device
    cusparseDnVecDescr_t in_desc;       // size-n device input (d_full)
    double *d_full;                     // device full SpMV input (n)
    double *h_full;                     // pinned host full SpMV input (n)
    CSR Acpu;                           // nc x n host
    Device_Buffer_SpMV buf;
    int ng, nc, n;
    cudaStream_t compute_s, copy_s;
    cudaEvent_t in_ready, h2d_done;
} DistCtx;

// Distributed SpMV: out = A * in, where in/out are split (device head, host tail).
// out_desc must wrap d_out (size ng). Concurrent GPU/CPU multiply.
static void dist_spmv(DistCtx *c, cusparseDnVecDescr_t out_desc,
                      const double *d_in, const double *h_in,
                      double *d_out, double *h_out) {
    const double one = 1.0, zero = 0.0;
    int ng = c->ng, nc = c->nc, n = c->n;

    // copy_s waits for compute_s's pending writes to d_in.
    cudaEventRecord(c->in_ready, c->compute_s);
    cudaStreamWaitEvent(c->copy_s, c->in_ready, 0);

    // assemble full input. device: d_full[0,ng)=d_in (D2D), d_full[ng,n)=h_in (H2D).
    cudaMemcpyAsync(c->d_full, d_in, ng * sizeof(double), cudaMemcpyDeviceToDevice, c->compute_s);
    if (nc > 0)
        cudaMemcpyAsync(c->d_full + ng, h_in, nc * sizeof(double), cudaMemcpyHostToDevice, c->copy_s);
    // host: h_full[0,ng)=d_in (D2H), h_full[ng,n)=h_in (host copy).
    if (nc > 0) {
        cudaMemcpyAsync(c->h_full, d_in, ng * sizeof(double), cudaMemcpyDeviceToHost, c->copy_s);
        memcpy(c->h_full + ng, h_in, nc * sizeof(double));
    }
    // GPU SpMV must wait for the H2D of the CPU slice into d_full.
    if (nc > 0) {
        cudaEventRecord(c->h2d_done, c->copy_s);
        cudaStreamWaitEvent(c->compute_s, c->h2d_done, 0);
    }
    cusparseSpMV(c->ch, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, c->Agpu, c->in_desc,
                 &zero, out_desc, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, c->buf);
    // CPU SpMV: wait for the D2H of the GPU slice, then multiply its rows.
    if (nc > 0) {
        cudaStreamSynchronize(c->copy_s);
        Vector y = {.vals = h_out, .nvals = nc};
        Vector x = {.vals = c->h_full, .nvals = n};
        CSR_spmxv_omp(c->Acpu, x, y);
    }
}

// Distributed dot -> out_dev (device scalar). GPU partial (dp) + CPU partial,
// combined on device. h_src is a distinct pinned slot for this dot's CPU partial.
static void dist_dot(cublasHandle_t bh, DistCtx *c, double *d_g, double *d_c,
                     double *out_dev, double *h_src,
                     const double *d_a, const double *d_b,
                     const double *h_a, const double *h_b) {
    cublasDdot(bh, c->ng, d_a, 1, d_b, 1, d_g);     // GPU partial (dp), async on compute_s
    *h_src = host_dot(h_a, h_b, c->nc);             // CPU partial (omp), overlaps GPU dot
    cudaMemcpyAsync(d_c, h_src, sizeof(double), cudaMemcpyHostToDevice, c->compute_s);
    hhp_dp_add(out_dev, d_g, d_c, c->compute_s);    // combine on device
}

int main(int argc, char *argv[]) {
    struct arguments arguments = {};
    arguments.n_iters = 1;
    argp_parse(&argp, argc, argv, 0, 0, &arguments);

    if (!arguments.input_matrix || !arguments.input_x || !arguments.input_y
        || !arguments.input_part || !arguments.input_gpu || !arguments.output_x) {
        fprintf(stderr, "Error: -m, -x, -y, -p, -g, -o must all be specified.\n");
        return EXIT_FAILURE;
    }
    if (access(arguments.input_matrix, F_OK) == -1) ABORT("matrix missing")
    if (access(arguments.input_x, F_OK) == -1) ABORT("X missing")
    if (access(arguments.input_y, F_OK) == -1) ABORT("B missing")
    if (access(arguments.input_part, F_OK) == -1) ABORT("partition missing")
    if (access(arguments.input_gpu, F_OK) == -1) ABORT("is_gpu missing")

    int niters = arguments.n_iters; if (niters < 1) niters = 1;
    double t_begin = omp_get_wtime();

    double t_read0 = omp_get_wtime();
    CSR A = buReadSparseMatrix(arguments.input_matrix);
    int n = A.n;
    Vector Xin = vector_read(arguments.input_x, n);
    Vector Bin = vector_read(arguments.input_y, A.m);
    printf("Matrix name : %s\n", arguments.input_matrix);

    int isgpu[8];
    int nrank = read_ints(arguments.input_gpu, isgpu, 8);
    if (nrank != 2) ABORT("is_gpu must have 2 entries (got %d)", nrank)
    int gpu_rank = -1, gpu_count = 0;
    for (int r = 0; r < 2; r++) if (isgpu[r] == 1) { gpu_rank = r; gpu_count++; }
    if (gpu_count != 1) ABORT("is_gpu must mark exactly one rank as GPU")
    int cpu_rank = 1 - gpu_rank;

    int *part; ALLOC_ARRAY(part, A.m);
    int np = read_ints(arguments.input_part, part, A.m);
    if (np != A.m) ABORT("Partition has %d entries, expected %d", np, A.m)

    int *perm; ALLOC_ARRAY(perm, A.m);
    int *perm_inv; ALLOC_ARRAY(perm_inv, A.m);
    int ng = 0, nc = 0;
    for (int i = 0; i < A.m; i++) if (part[i] == gpu_rank) perm[ng++] = i;
    int gpu_end = ng;
    for (int i = 0; i < A.m; i++) if (part[i] == cpu_rank) perm[gpu_end + nc++] = i;
    if (ng + nc != A.m) ABORT("unmapped rows")
    for (int i = 0; i < A.m; i++) perm_inv[perm[i]] = i;
    printf("n_gpu=%d  n_cpu=%d  total=%d\n", ng, nc, A.m);

    CSR Aperm = csr_permute(A, perm, perm_inv);
    freeSparseMatrix(&A);
    Vector Xp = vector_init(n), Bp = vector_init(n);
    for (int i = 0; i < n; i++) Xp.vals[i] = Xin.vals[perm[i]];
    for (int i = 0; i < n; i++) Bp.vals[i] = Bin.vals[perm[i]];
    vector_destroy(&Xin); vector_destroy(&Bin);
    CSR Agpu_h = csr_row_slice(Aperm, 0, ng);
    CSR Acpu   = csr_row_slice(Aperm, ng, ng + nc);
    double t_read1 = omp_get_wtime();

    // --- CUDA setup ---
    CHECK_CUDA(cudaSetDevice(0))
    cublasHandle_t bh; CHECK_CUBLAS(cublasCreate(&bh))
    cusparseHandle_t ch; CHECK_CUSPARSE(cusparseCreate(&ch))
    cudaStream_t compute_s, copy_s;
    CHECK_CUDA(cudaStreamCreate(&compute_s))
    CHECK_CUDA(cudaStreamCreate(&copy_s))
    CHECK_CUBLAS(cublasSetStream(bh, compute_s))
    CHECK_CUSPARSE(cusparseSetStream(ch, compute_s))
    cudaEvent_t in_ready, h2d_done;
    CHECK_CUDA(cudaEventCreateWithFlags(&in_ready, cudaEventDisableTiming))
    CHECK_CUDA(cudaEventCreateWithFlags(&h2d_done, cudaEventDisableTiming))

    Device_CSR dAgpu; CHECK_CUSPARSE(device_csr_create(Agpu_h, &dAgpu))

    // split device vectors (size ng) and host vectors (size nc)
    double *dX = dvec(ng), *dB = dvec(ng), *dY = dvec(ng), *dV = dvec(ng),
           *dP = dvec(ng), *dR = dvec(ng), *dR0 = dvec(ng), *dS = dvec(ng), *dT = dvec(ng);
    double *hX = malloc((nc>0?nc:1)*sizeof(double)), *hB = malloc((nc>0?nc:1)*sizeof(double)),
           *hY = malloc((nc>0?nc:1)*sizeof(double)), *hV = malloc((nc>0?nc:1)*sizeof(double)),
           *hP = malloc((nc>0?nc:1)*sizeof(double)), *hR = malloc((nc>0?nc:1)*sizeof(double)),
           *hR0= malloc((nc>0?nc:1)*sizeof(double)), *hS = malloc((nc>0?nc:1)*sizeof(double)),
           *hT = malloc((nc>0?nc:1)*sizeof(double));
    // init X, B from permuted vectors
    CHECK_CUDA(cudaMemcpy(dX, Xp.vals, ng*sizeof(double), cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dB, Bp.vals, ng*sizeof(double), cudaMemcpyHostToDevice))
    for (int i = 0; i < nc; i++) hX[i] = Xp.vals[ng+i];
    for (int i = 0; i < nc; i++) hB[i] = Bp.vals[ng+i];
    CHECK_CUDA(cudaMemset(dV, 0, ng*sizeof(double)))
    CHECK_CUDA(cudaMemset(dP, 0, ng*sizeof(double)))
    for (int i = 0; i < nc; i++) hV[i] = 0.0;
    for (int i = 0; i < nc; i++) hP[i] = 0.0;

    // full SpMV buffers
    double *d_full = dvec(n), *h_full;
    CHECK_CUDA(cudaMallocHost((void**)&h_full, n*sizeof(double)))
    cusparseDnVecDescr_t in_desc, dY_desc, dV_desc, dT_desc;
    CHECK_CUSPARSE(cusparseCreateDnVec(&in_desc, n, d_full, CUDA_R_64F))
    CHECK_CUSPARSE(cusparseCreateDnVec(&dY_desc, ng>0?ng:1, dY, CUDA_R_64F))
    CHECK_CUSPARSE(cusparseCreateDnVec(&dV_desc, ng>0?ng:1, dV, CUDA_R_64F))
    CHECK_CUSPARSE(cusparseCreateDnVec(&dT_desc, ng>0?ng:1, dT, CUDA_R_64F))
    Device_Buffer_SpMV spmv_buf;
    { size_t bs = 0; const double a=1.0,b=0.0;
      CHECK_CUSPARSE(cusparseSpMV_bufferSize(ch, CUSPARSE_OPERATION_NON_TRANSPOSE, &a,
          dAgpu.desc, in_desc, &b, dY_desc, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bs))
      CHECK_CUDA(cudaMalloc(&spmv_buf, bs)) }

    DistCtx ctx = { .ch=ch, .Agpu=dAgpu.desc, .in_desc=in_desc, .d_full=d_full, .h_full=h_full,
                    .Acpu=Acpu, .buf=spmv_buf, .ng=ng, .nc=nc, .n=n,
                    .compute_s=compute_s, .copy_s=copy_s, .in_ready=in_ready, .h2d_done=h2d_done };

    // device scalars
    double *d_rho=dscalar(1.0), *d_alpha=dscalar(1.0), *d_omega=dscalar(1.0),
           *d_beta=dscalar(0.0), *d_neg_a=dscalar(0.0), *d_neg_w=dscalar(0.0),
           *d_one=dscalar(1.0), *d_neg_one=dscalar(-1.0),
           *d_g=dscalar(0.0), *d_c=dscalar(0.0),
           *d_rho_new=dscalar(0.0), *d_rv=dscalar(0.0),
           *d_ts=dscalar(0.0), *d_tt=dscalar(0.0), *d_tol=dscalar(0.0);
    // pinned host scalars: partials (distinct slots) + shadows for CPU vecops
    double *hp; CHECK_CUDA(cudaMallocHost((void**)&hp, 8*sizeof(double)))   // partial sources
    double *hs; CHECK_CUDA(cudaMallocHost((void**)&hs, 8*sizeof(double)))   // shadows
    // shadows: hs[0]=beta hs[1]=alpha hs[2]=neg_a hs[3]=omega hs[4]=neg_w
    hs[3] = 1.0;  // omega starts at 1
    hs[1] = 1.0;  // alpha starts at 1

    printf("LOG: Setup done\n");

    CHECK_CUBLAS(cublasSetPointerMode(bh, CUBLAS_POINTER_MODE_DEVICE))

    // --- Initial residual: R = B - A*X ; R0 = R ---
    dist_spmv(&ctx, dY_desc, dX, hX, dY, hY);                 // Y = A*X
    CHECK_CUBLAS(cublasDcopy(bh, ng, dB, 1, dR, 1))           // R = B
    CHECK_CUBLAS(cublasDaxpy(bh, ng, d_neg_one, dY, 1, dR, 1))// R -= Y
    host_copy(hB, hR, nc); host_axpy(-1.0, hY, hR, nc);
    CHECK_CUBLAS(cublasDcopy(bh, ng, dR, 1, dR0, 1))          // R0 = R
    host_copy(hR, hR0, nc);
    cudaStreamSynchronize(compute_s);

    double t_loop0 = omp_get_wtime();
    for (int it = 0; it < niters; it++) {
        // rho_new = R.R0 ; beta = (rho_new/rho)*(alpha/omega) ; rho = rho_new
        dist_dot(bh, &ctx, d_g, d_c, d_rho_new, &hp[0], dR, dR0, hR, hR0);
        hhp_dp_update_beta(d_beta, d_rho, d_rho_new, d_alpha, d_omega, compute_s);
        cudaMemcpyAsync(&hs[0], d_beta, sizeof(double), cudaMemcpyDeviceToHost, compute_s);
        cudaStreamSynchronize(compute_s);                    // CPU needs beta (and omega shadow hs[3])
        // P = (P - omega*V)*beta + R
        CHECK_CUBLAS(cublasDscal(bh, ng, d_omega, dV, 1))
        CHECK_CUBLAS(cublasDaxpy(bh, ng, d_neg_one, dV, 1, dP, 1))
        CHECK_CUBLAS(cublasDscal(bh, ng, d_beta, dP, 1))
        CHECK_CUBLAS(cublasDaxpy(bh, ng, d_one, dR, 1, dP, 1))
        host_scal(hs[3], hV, nc); host_axpy(-1.0, hV, hP, nc);
        host_scal(hs[0], hP, nc); host_axpy(1.0, hR, hP, nc);

        // V = A*P
        dist_spmv(&ctx, dV_desc, dP, hP, dV, hV);

        // rv = R0.V ; alpha = rho/rv ; neg_a = -alpha
        dist_dot(bh, &ctx, d_g, d_c, d_rv, &hp[1], dR0, dV, hR0, hV);
        hhp_dp_update_alpha(d_alpha, d_neg_a, d_rho, d_rv, compute_s);
        cudaMemcpyAsync(&hs[1], d_alpha, sizeof(double), cudaMemcpyDeviceToHost, compute_s);
        cudaMemcpyAsync(&hs[2], d_neg_a, sizeof(double), cudaMemcpyDeviceToHost, compute_s);
        cudaStreamSynchronize(compute_s);
        // S = R - alpha*V
        CHECK_CUBLAS(cublasDcopy(bh, ng, dR, 1, dS, 1))
        CHECK_CUBLAS(cublasDaxpy(bh, ng, d_neg_a, dV, 1, dS, 1))
        host_copy(hR, hS, nc); host_axpy(hs[2], hV, hS, nc);

        // T = A*S
        dist_spmv(&ctx, dT_desc, dS, hS, dT, hT);

        // ts = T.S ; tt = T.T ; omega = ts/tt ; neg_w = -omega
        dist_dot(bh, &ctx, d_g, d_c, d_ts, &hp[2], dT, dS, hT, hS);
        dist_dot(bh, &ctx, d_g, d_c, d_tt, &hp[3], dT, dT, hT, hT);
        hhp_dp_update_omega(d_omega, d_neg_w, d_ts, d_tt, compute_s);
        cudaMemcpyAsync(&hs[3], d_omega, sizeof(double), cudaMemcpyDeviceToHost, compute_s);
        cudaMemcpyAsync(&hs[4], d_neg_w, sizeof(double), cudaMemcpyDeviceToHost, compute_s);
        cudaStreamSynchronize(compute_s);
        // X += alpha*P + omega*S ; R = S - omega*T
        CHECK_CUBLAS(cublasDaxpy(bh, ng, d_alpha, dP, 1, dX, 1))
        CHECK_CUBLAS(cublasDaxpy(bh, ng, d_omega, dS, 1, dX, 1))
        CHECK_CUBLAS(cublasDcopy(bh, ng, dS, 1, dR, 1))
        CHECK_CUBLAS(cublasDaxpy(bh, ng, d_neg_w, dT, 1, dR, 1))
        host_axpy(hs[1], hP, hX, nc); host_axpy(hs[3], hS, hX, nc);
        host_copy(hS, hR, nc); host_axpy(hs[4], hT, hR, nc);

        // tol = S.S (parity; result unused, no rendezvous)
        dist_dot(bh, &ctx, d_g, d_c, d_tol, &hp[4], dS, dS, hS, hS);
    }
    cudaDeviceSynchronize();
    double t_loop1 = omp_get_wtime();

    // --- Final relative residual: ||A*X - B|| / ||B|| ---
    dist_spmv(&ctx, dY_desc, dX, hX, dY, hY);                 // Y = A*X
    CHECK_CUBLAS(cublasDaxpy(bh, ng, d_neg_one, dB, 1, dY, 1))// Y -= B
    host_axpy(-1.0, hB, hY, nc);
    // sy = Y.Y, sb = B.B  (combine partials on host)
    double sy_g, sb_g;
    cublasDdot(bh, ng, dY, 1, dY, 1, d_g); cudaMemcpyAsync(&sy_g, d_g, sizeof(double), cudaMemcpyDeviceToHost, compute_s);
    cudaStreamSynchronize(compute_s);
    cublasDdot(bh, ng, dB, 1, dB, 1, d_g); cudaMemcpy(&sb_g, d_g, sizeof(double), cudaMemcpyDeviceToHost);
    double sy = sy_g + host_dot(hY, hY, nc);
    double sb = sb_g + host_dot(hB, hB, nc);
    double relative_residual = sqrt(sy / sb);

    printf("n_iters : %d \n", niters);
    printf("spmv : %lf \n", t_loop1 - t_loop0);
    printf("file_read : %lf \n", t_read1 - t_read0);
    printf("relative_residual : %E\n", relative_residual);
    printf("everything_total : %lf\n", omp_get_wtime() - t_begin);
    printf("\n----------------------------------------------------------------------\n");

    // --- gather X back, unpermute, write ---
    Vector Xfull = vector_init(n);
    CHECK_CUDA(cudaMemcpy(Xfull.vals, dX, ng*sizeof(double), cudaMemcpyDeviceToHost))
    for (int i = 0; i < nc; i++) Xfull.vals[ng+i] = hX[i];
    Vector Xout = vector_init(n);
    for (int i = 0; i < n; i++) Xout.vals[perm[i]] = Xfull.vals[i];
    vector_write(arguments.output_x, Xout);

    vector_destroy(&Xfull); vector_destroy(&Xout);
    vector_destroy(&Xp); vector_destroy(&Bp);
    return EXIT_SUCCESS;
}
