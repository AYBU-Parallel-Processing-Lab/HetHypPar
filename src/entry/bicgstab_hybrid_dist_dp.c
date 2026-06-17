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
#include "hhp_dp_helpers.h"

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

// host_dot/axpy/scal/copy, read_ints, dscalar, dvec, csr_permute, csr_row_slice
// live in hhp_dp_helpers.h.

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
    unsigned long long *flagd;          // mapped-host flag (device ptr)
    volatile unsigned long long *flagv; // mapped-host flag (host ptr)
    unsigned long long *seq;            // shared monotonic sequence counter
    int *d_halo_idx;                    // GPU-owned columns the CPU rows touch (device)
    int *h_halo_idx;                    // same (host, for scatter)
    double *d_halo;                     // gathered halo values (device)
    double *h_halo;                     // gathered halo values (pinned host)
    int nh;                             // halo size
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
    // host input: CPU's own slice goes to h_full[ng,n); for h_full[0,ng) the CPU
    // only needs the halo (GPU columns its rows touch), so gather just those on
    // the GPU and D->H the compact buffer instead of the whole ng slice.
    if (nc > 0) {
        hhp_dp_gather(c->d_halo, d_in, c->d_halo_idx, c->nh, c->copy_s);
        cudaMemcpyAsync(c->h_halo, c->d_halo, c->nh * sizeof(double),
                        cudaMemcpyDeviceToHost, c->copy_s);
        memcpy(c->h_full + ng, h_in, nc * sizeof(double));
    }
    // GPU SpMV must wait for the H2D of the CPU slice into d_full.
    if (nc > 0) {
        cudaEventRecord(c->h2d_done, c->copy_s);
        cudaStreamWaitEvent(c->compute_s, c->h2d_done, 0);
    }
    cusparseSpMV(c->ch, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, c->Agpu, c->in_desc,
                 &zero, out_desc, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, c->buf);
    // CPU SpMV: wait for the D2H of the GPU slice (spin-wait on a flag the GPU
    // sets after the copy), then multiply its rows.
    if (nc > 0) {
        hhp_dp_set_flag(c->flagd, ++(*c->seq), c->copy_s);
        while (*c->flagv < *c->seq) {}
        // scatter halo values into their global positions in h_full
        #pragma omp parallel for
        for (int k = 0; k < c->nh; k++) c->h_full[c->h_halo_idx[k]] = c->h_halo[k];
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

// Like dist_dot but leaves the partials separate (d_g = GPU partial, d_c = CPU
// partial on device) for a fused combine+update kernel to consume.
static void dist_partials(cublasHandle_t bh, DistCtx *c, double *d_g, double *d_c,
                          double *h_src, const double *d_a, const double *d_b,
                          const double *h_a, const double *h_b) {
    cublasDdot(bh, c->ng, d_a, 1, d_b, 1, d_g);
    *h_src = host_dot(h_a, h_b, c->nc);
    cudaMemcpyAsync(d_c, h_src, sizeof(double), cudaMemcpyHostToDevice, c->compute_s);
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
    CHECK_CUDA(cudaSetDeviceFlags(cudaDeviceMapHost))   // allow GPU writes to mapped host mem
    CHECK_CUDA(cudaSetDevice(0))
    cublasHandle_t bh; CHECK_CUBLAS(cublasCreate(&bh))
    cusparseHandle_t ch; CHECK_CUSPARSE(cusparseCreate(&ch))
    cudaStream_t compute_s, copy_s;
    CHECK_CUDA(cudaStreamCreate(&compute_s))
    CHECK_CUDA(cudaStreamCreate(&copy_s))
    CHECK_CUBLAS(cublasSetStream(bh, compute_s))
    CHECK_CUSPARSE(cusparseSetStream(ch, compute_s))
    cudaEvent_t in_ready, h2d_done, e_ready, e_host;
    CHECK_CUDA(cudaEventCreateWithFlags(&in_ready, cudaEventDisableTiming))
    CHECK_CUDA(cudaEventCreateWithFlags(&h2d_done, cudaEventDisableTiming))
    CHECK_CUDA(cudaEventCreateWithFlags(&e_ready, cudaEventDisableTiming))
    CHECK_CUDA(cudaEventCreateWithFlags(&e_host, cudaEventDisableTiming))

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

    // --- Halo: the distinct GPU-owned columns (< ng) that the CPU rows touch.
    // The CPU only needs these of the GPU slice, not all ng, each SpMV.
    int nh = 0;
    int *halo_idx = NULL;
    {
        char *seen = calloc(ng > 0 ? ng : 1, 1);
        for (int k = 0; k < Acpu.nnz; k++) {
            int col = Acpu.J[k];
            if (col < ng && !seen[col]) { seen[col] = 1; nh++; }
        }
        halo_idx = malloc((nh > 0 ? nh : 1) * sizeof(int));
        int idx = 0;
        for (int col = 0; col < ng; col++) if (seen[col]) halo_idx[idx++] = col;
        free(seen);
    }
    printf("halo: nh=%d / ng=%d (%.1f%% of GPU slice)\n",
           nh, ng, ng > 0 ? 100.0 * nh / ng : 0.0);
    int *d_halo_idx; CHECK_CUDA(cudaMalloc((void**)&d_halo_idx, (nh > 0 ? nh : 1) * sizeof(int)))
    if (nh > 0) CHECK_CUDA(cudaMemcpy(d_halo_idx, halo_idx, nh * sizeof(int), cudaMemcpyHostToDevice))
    double *d_halo = dvec(nh);
    double *h_halo; CHECK_CUDA(cudaMallocHost((void**)&h_halo, (nh > 0 ? nh : 1) * sizeof(double)))

    DistCtx ctx = { .ch=ch, .Agpu=dAgpu.desc, .in_desc=in_desc, .d_full=d_full, .h_full=h_full,
                    .Acpu=Acpu, .buf=spmv_buf, .ng=ng, .nc=nc, .n=n,
                    .compute_s=compute_s, .copy_s=copy_s, .in_ready=in_ready, .h2d_done=h2d_done,
                    .d_halo_idx=d_halo_idx, .h_halo_idx=halo_idx, .d_halo=d_halo, .h_halo=h_halo, .nh=nh };

    // device scalars
    double *d_rho=dscalar(1.0), *d_alpha=dscalar(1.0), *d_omega=dscalar(1.0),
           *d_beta=dscalar(0.0), *d_neg_a=dscalar(0.0), *d_neg_w=dscalar(0.0),
           *d_one=dscalar(1.0), *d_neg_one=dscalar(-1.0),
           *d_g=dscalar(0.0), *d_c=dscalar(0.0),
           *d_g2=dscalar(0.0), *d_c2=dscalar(0.0),
           *d_rho_new=dscalar(0.0), *d_rv=dscalar(0.0),
           *d_ts=dscalar(0.0), *d_tt=dscalar(0.0), *d_tol=dscalar(0.0);
    // pinned host scalars: partials (distinct slots) + shadows for CPU vecops
    double *hp; CHECK_CUDA(cudaMallocHost((void**)&hp, 8*sizeof(double)))   // partial sources
    double *hs; CHECK_CUDA(cudaMallocHost((void**)&hs, 8*sizeof(double)))   // shadows
    // shadows: hs[0]=beta hs[1]=alpha hs[2]=neg_a hs[3]=omega hs[4]=neg_w
    hs[3] = 1.0;  // omega starts at 1
    hs[1] = 1.0;  // alpha starts at 1

    // Mapped host memory for low-latency GPU->host scalar publishing: the
    // combine-update kernel writes the scalar(s) into hm[] and bumps *flagh; the
    // host spin-waits on flagh instead of cudaStreamSynchronize (~1-2us vs ~15us).
    double *hm; unsigned long long *flagh;
    CHECK_CUDA(cudaHostAlloc((void**)&hm, 2*sizeof(double), cudaHostAllocMapped))
    CHECK_CUDA(cudaHostAlloc((void**)&flagh, sizeof(unsigned long long), cudaHostAllocMapped))
    *flagh = 0;
    double *dm; unsigned long long *flagd;
    CHECK_CUDA(cudaHostGetDevicePointer((void**)&dm, hm, 0))
    CHECK_CUDA(cudaHostGetDevicePointer((void**)&flagd, flagh, 0))
    unsigned long long seq = 0;
    volatile unsigned long long *flagv = flagh;
    ctx.flagd = flagd; ctx.flagv = flagv; ctx.seq = &seq;

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
        dist_partials(bh, &ctx, d_g, d_c, &hp[0], dR, dR0, hR, hR0);
        hhp_dp_cu_beta(d_beta, d_rho, d_g, d_c, d_alpha, d_omega, dm, flagd, ++seq, compute_s);
        hhp_dp_vecop_P(dP, dV, dR, d_omega, d_beta, ng, compute_s);  // GPU vecop runs while host spins
        while (*flagv < seq) {}                              // spin-wait for beta (~1-2us)
        hs[0] = hm[0];
        // fused host vecop: P = (P - omega*V)*beta + R  (one region, one pass)
        #pragma omp parallel for
        for (int i = 0; i < nc; i++)
            hP[i] = (hP[i] - hs[3]*hV[i]) * hs[0] + hR[i];

        // V = A*P
        dist_spmv(&ctx, dV_desc, dP, hP, dV, hV);

        // rv = R0.V ; alpha = rho/rv ; neg_a = -alpha
        dist_partials(bh, &ctx, d_g, d_c, &hp[1], dR0, dV, hR0, hV);
        hhp_dp_cu_alpha(d_alpha, d_neg_a, d_rho, d_g, d_c, dm, flagd, ++seq, compute_s);
        hhp_dp_vecop_S(dS, dR, dV, d_neg_a, ng, compute_s);
        while (*flagv < seq) {}                              // spin-wait for alpha,neg_a
        hs[1] = hm[0]; hs[2] = hm[1];
        // fused host vecop: S = R - alpha*V   (hs[2] = -alpha)
        #pragma omp parallel for
        for (int i = 0; i < nc; i++)
            hS[i] = hR[i] + hs[2]*hV[i];

        // T = A*S
        dist_spmv(&ctx, dT_desc, dS, hS, dT, hT);

        // ts = T.S ; tt = T.T ; omega = ts/tt ; neg_w = -omega
        // ts = T.S ; tt = T.T : two GPU partials + ONE fused CPU reduction
        cublasDdot(bh, ng, dT, 1, dS, 1, d_g);
        cublasDdot(bh, ng, dT, 1, dT, 1, d_g2);
        { double cts = 0.0, ctt = 0.0;
          #pragma omp parallel for reduction(+:cts,ctt)
          for (int i = 0; i < nc; i++) { double tv = hT[i]; cts += tv*hS[i]; ctt += tv*tv; }
          hp[2] = cts; hp[3] = ctt; }
        cudaMemcpyAsync(d_c,  &hp[2], sizeof(double), cudaMemcpyHostToDevice, compute_s);
        cudaMemcpyAsync(d_c2, &hp[3], sizeof(double), cudaMemcpyHostToDevice, compute_s);
        hhp_dp_cu_omega(d_omega, d_neg_w, d_g, d_c, d_g2, d_c2, dm, flagd, ++seq, compute_s);
        hhp_dp_vecop_XR(dX, dR, dP, dS, dT, d_alpha, d_omega, d_neg_w, ng, compute_s);
        while (*flagv < seq) {}                              // spin-wait for omega,neg_w
        hs[3] = hm[0]; hs[4] = hm[1];
        // fused host vecop: X += alpha*P + omega*S ; R = S - omega*T  (one region)
        #pragma omp parallel for
        for (int i = 0; i < nc; i++) {
            hX[i] += hs[1]*hP[i] + hs[3]*hS[i];
            hR[i]  = hS[i] + hs[4]*hT[i];
        }

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
