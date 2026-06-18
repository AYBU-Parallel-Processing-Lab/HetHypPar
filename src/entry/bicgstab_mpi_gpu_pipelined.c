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

#include <mpi.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

#include "argp.h"
#include "unistd.h"

// Distributed (MPI) all-GPU pipelined BiCGStab -- Algorithm 9 of Cools & Vanroose
// (2017), with the two dot-product reductions made non-blocking (MPI_Iallreduce)
// and overlapped with the SHARD SpMV. Every rank uses a GPU (homogeneous; the
// is_gpu file is read but ignored). The recurrence math is the one validated
// single-node in bicgstab-gpu-pipelined. Residual replacement (Sec 4.2) via env
// HHP_REPLACE=k. Correctness can be checked on 2 ranks here; the communication-
// hiding speedup only appears on a real multi-node interconnect.

struct arguments { char *input_matrix, *input_x, *input_y, *input_part, *input_gpu, *output_x; int n_iters; };
static struct argp_option options[] = {
    {"input-matrix",'m',"FILE",0,"matrix"},{"input-x",'x',"FILE",0,"X"},{"input-y",'y',"FILE",0,"B"},
    {"input-part",'p',"FILE",0,"partition"},{"is-gpu",'g',"FILE",0,"is_gpu"},{"output-x",'o',"FILE",0,"out X"},
    {"n-iters",'n',"POSITIVE-INTEGER",0,"iters"},{0}
};
static error_t parse_opt(int key, char *arg, struct argp_state *state) {
    struct arguments *a = state->input; char *buf = NULL;
    switch (key) {
        case 'm': a->input_matrix=arg; break; case 'x': a->input_x=arg; break; case 'y': a->input_y=arg; break;
        case 'p': a->input_part=arg; break; case 'g': a->input_gpu=arg; break; case 'o': a->output_x=arg; break;
        case 'n': a->n_iters=strtol(arg,&buf,10); break; case ARGP_KEY_ARG: return 0; default: return ARGP_ERR_UNKNOWN;
    }
    return 0;
}
static char doc[] = "MPI all-GPU pipelined BiCGStab (Alg. 9, Iallreduce-overlapped)";
static struct argp argp = {options, parse_opt, "", doc};

// Global dot of two local Device_Vectors (local dot + blocking Allreduce). Used
// outside the hot loop (init / final residual).
static double gdot(cublasHandle_t bh, Device_Vector a, Device_Vector b) {
    double loc, glob; CHECK_CUBLAS(device_vector_dot(bh, a, b, &loc))
    MPI_Allreduce(&loc, &glob, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); return glob;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &size);

    struct arguments arguments = {}; arguments.n_iters = 1;
    argp_parse(&argp, argc, argv, 0, 0, &arguments);
    int niters = arguments.n_iters; if (niters < 1) niters = 1;
    int replace_k = getenv("HHP_REPLACE") ? atoi(getenv("HHP_REPLACE")) : 0;

    double t_begin = omp_get_wtime();

    // --- read + distribute (matrix -> SHARD_CSC; X,B -> local slices) ---
    iVector partvec; SHARD_CSC A; Vector X, B; int ncols = 0;
    {
        CSR bigmat = {};
        if (rank == 0) { bigmat = buReadSparseMatrix(arguments.input_matrix); ncols = bigmat.n;
                         printf("Matrix name : %s\n", arguments.input_matrix); }
        MPI_Bcast(&ncols, 1, MPI_INT, 0, MPI_COMM_WORLD);
        (void)MPI_ivector_read_scatter(arguments.input_gpu, size);   // read but ignore (all-GPU)
        partvec = MPI_ivector_read_bcast(arguments.input_part, ncols);
        A = MPI_CSR_split_row(bigmat, partvec);
        X = MPI_vector_read_parted(arguments.input_x, ncols, partvec);
        B = MPI_vector_read_parted(arguments.input_y, ncols, partvec);
        if (rank == 0) freeSparseMatrix(&bigmat);
    }
    int nloc = X.nvals;   // local rows owned by this rank
    double t_read1 = omp_get_wtime();

    // --- all-GPU setup ---
    CHECK_CUDA(cudaSetDevice(0))
    cublasHandle_t bh; CHECK_CUBLAS(cublasCreate(&bh))
    cusparseHandle_t ch; CHECK_CUSPARSE(cusparseCreate(&ch))
    const double one = 1.0, zero = 0.0, m_one = -1.0;

    Device_SHARD_CSC dA = { .gind = A.gind, .recv = A.recv, .send = A.send };
    CHECK_CUSPARSE(device_csc_create(A.loc, &dA.loc))
    int has_shr = (A.shr.n > 0);
    if (has_shr) CHECK_CUSPARSE(device_csc_create(A.shr, &dA.shr))

    Device_Vector dX_shr = {0};
    if (has_shr) CHECK_CUSPARSE(device_vector_init(dA.shr.data.n, &dX_shr))
    Device_Buffer_SpMV locbuf, shrbuf = NULL;

    // work vectors (all local-sized)
    Device_Vector dX, dB, R, R0, W, T, P, S, Z, Q, Y, V, TMP;
    CHECK_CUSPARSE(device_vector_init(nloc, &dX)) CHECK_CUSPARSE(device_vector_init(nloc, &dB))
    CHECK_CUSPARSE(device_vector_init(nloc, &R))  CHECK_CUSPARSE(device_vector_init(nloc, &R0))
    CHECK_CUSPARSE(device_vector_init(nloc, &W))  CHECK_CUSPARSE(device_vector_init(nloc, &T))
    CHECK_CUSPARSE(device_vector_init(nloc, &P))  CHECK_CUSPARSE(device_vector_init(nloc, &S))
    CHECK_CUSPARSE(device_vector_init(nloc, &Z))  CHECK_CUSPARSE(device_vector_init(nloc, &Q))
    CHECK_CUSPARSE(device_vector_init(nloc, &Y))  CHECK_CUSPARSE(device_vector_init(nloc, &V))
    CHECK_CUSPARSE(device_vector_init(nloc, &TMP))
    CHECK_CUSPARSE(device_buffer_spmv_create(ch, dA.loc.desc, dX, dB, &one, &zero, &locbuf))
    if (has_shr) CHECK_CUSPARSE(device_buffer_spmv_create(ch, dA.shr.desc, dX_shr, dB, &one, &one, &shrbuf))
    CHECK_CUSPARSE(device_vector_toGPU(X, dX)) CHECK_CUSPARSE(device_vector_toGPU(B, dB))
    Vector Xs = vector_init(nloc);   // host scratch for the SHARD SpMV's internal D->H

    #define SPMV(IN, OUT) CHECK_CUSPARSE(MPI_device_SHARD_CSC_mpi_spmxv(dA, Xs, (IN), dX_shr, (OUT), \
                          MPI_COMM_WORLD, ch, one, zero, locbuf, shrbuf, NULL))

    // --- init: r = b - A x ; R0 = r ; w = A r ; t = A w ; p=s=z=v=0 ---
    SPMV(dX, TMP);                                              // TMP = A x
    CHECK_CUDA(device_vector_GPUtoGPU(dB, R)) CHECK_CUBLAS(device_vector_axpy(bh, TMP, m_one, R)) // R = b - A x
    CHECK_CUDA(device_vector_GPUtoGPU(R, R0))
    SPMV(R, W);                                                 // W = A r
    SPMV(W, T);                                                 // T = A w
    CHECK_CUSPARSE(device_vector_zero(P)) CHECK_CUSPARSE(device_vector_zero(S))
    CHECK_CUSPARSE(device_vector_zero(Z)) CHECK_CUSPARSE(device_vector_zero(V))

    double rho_old = gdot(bh, R0, R);
    double alpha = rho_old / gdot(bh, R0, W);
    double beta_prev = 0.0, omega_prev = 1.0, omega = 0.0;
    double bnorm = sqrt(gdot(bh, dB, dB));
    if (rank == 0) printf("LOG: setup done (nloc rank0=%d, shr=%d)\n", nloc, has_shr);

    double t_loop0 = omp_get_wtime();
    for (int i = 0; i < niters; i++) {
        double bw = beta_prev * omega_prev;
        // p = beta*p + r - beta*omega*s ; s = beta*s + w - beta*omega*z ; z = beta*z + t - beta*omega*v
        CHECK_CUBLAS(device_vector_scale(bh, beta_prev, P)) CHECK_CUBLAS(device_vector_axpy(bh, R, one, P)) CHECK_CUBLAS(device_vector_axpy(bh, S, -bw, P))
        CHECK_CUBLAS(device_vector_scale(bh, beta_prev, S)) CHECK_CUBLAS(device_vector_axpy(bh, W, one, S)) CHECK_CUBLAS(device_vector_axpy(bh, Z, -bw, S))
        CHECK_CUBLAS(device_vector_scale(bh, beta_prev, Z)) CHECK_CUBLAS(device_vector_axpy(bh, T, one, Z)) CHECK_CUBLAS(device_vector_axpy(bh, V, -bw, Z))
        // q = r - alpha s ; y = w - alpha z
        CHECK_CUDA(device_vector_GPUtoGPU(R, Q)) CHECK_CUBLAS(device_vector_axpy(bh, S, -alpha, Q))
        CHECK_CUDA(device_vector_GPUtoGPU(W, Y)) CHECK_CUBLAS(device_vector_axpy(bh, Z, -alpha, Y))

        // ---- reduction 1 (omega): local dots, Iallreduce, OVERLAP with v = A z ----
        double l1[2], g1[2];
        CHECK_CUBLAS(device_vector_dot(bh, Q, Y, &l1[0])) CHECK_CUBLAS(device_vector_dot(bh, Y, Y, &l1[1]))
        MPI_Request req1; MPI_Iallreduce(l1, g1, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &req1);
        SPMV(Z, V);                                            // overlap
        MPI_Wait(&req1, MPI_STATUS_IGNORE);
        omega = g1[0] / g1[1];

        // x += alpha p + omega q ; r = q - omega y ; w = y - omega(t - alpha v)
        CHECK_CUBLAS(device_vector_axpy(bh, P, alpha, dX)) CHECK_CUBLAS(device_vector_axpy(bh, Q, omega, dX))
        CHECK_CUDA(device_vector_GPUtoGPU(Q, R)) CHECK_CUBLAS(device_vector_axpy(bh, Y, -omega, R))
        CHECK_CUDA(device_vector_GPUtoGPU(Y, W)) CHECK_CUBLAS(device_vector_axpy(bh, T, -omega, W)) CHECK_CUBLAS(device_vector_axpy(bh, V, omega*alpha, W))

        int replace = (replace_k > 0 && (i + 1) % replace_k == 0);
        if (replace) {
            // residual replacement: recompute r,w,t,s,z,v from true definitions (6 SpMVs, no overlap)
            SPMV(dX, TMP); CHECK_CUDA(device_vector_GPUtoGPU(dB, R)) CHECK_CUBLAS(device_vector_axpy(bh, TMP, m_one, R))
            SPMV(R, W); SPMV(W, T); SPMV(P, S); SPMV(S, Z); SPMV(Z, V);
        } else {
            SPMV(W, T);                                       // t = A w (normal path overlaps reduction 2 below)
        }

        // ---- reduction 2 (beta, alpha): dots on (r,w,s,z) ----
        double l2[4], g2[4];
        CHECK_CUBLAS(device_vector_dot(bh, R0, R, &l2[0])) CHECK_CUBLAS(device_vector_dot(bh, R0, W, &l2[1]))
        CHECK_CUBLAS(device_vector_dot(bh, R0, S, &l2[2])) CHECK_CUBLAS(device_vector_dot(bh, R0, Z, &l2[3]))
        if (replace) {
            MPI_Allreduce(l2, g2, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);   // t already computed in RR
        } else {
            MPI_Request req2; MPI_Iallreduce(l2, g2, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &req2);
            SPMV(W, T);                                       // overlap: t = A w (for next iter's z recurrence)
            MPI_Wait(&req2, MPI_STATUS_IGNORE);
        }
        double rho_new = g2[0];
        double beta = (alpha / omega) * (rho_new / rho_old);
        double denom = g2[1] + beta * g2[2] - beta * omega * g2[3];
        rho_old = rho_new; beta_prev = beta; omega_prev = omega; alpha = rho_new / denom;
    }
    cudaDeviceSynchronize();
    double t_loop1 = omp_get_wtime();

    // --- final true residual ||A x - B|| / ||B|| ---
    SPMV(dX, TMP); CHECK_CUBLAS(device_vector_axpy(bh, dB, m_one, TMP))
    double rel = sqrt(gdot(bh, TMP, TMP)) / bnorm;

    if (rank == 0) {
        printf("n_iters : %d \n", niters);
        printf("spmv : %lf \n", t_loop1 - t_loop0);
        printf("file_read : %lf \n", t_read1 - t_begin);
        printf("relative_residual : %E\n", rel);
        printf("everything_total : %lf\n", omp_get_wtime() - t_begin);
        printf("\n----------------------------------------------------------------------\n");
    }
    MPI_Finalize();
    return EXIT_SUCCESS;
}
