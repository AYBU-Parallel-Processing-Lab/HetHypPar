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

// Single-process CPU+GPU hybrid PIPELINED BiCGStab -- NO MPI. Algorithm 9 of
// Cools & Vanroose (2017), work split across CPU and GPU like
// bicgstab-hybrid-dist-dp: GPU owns rows [0,ng), CPU owns [ng,n); every vector is
// split (device head + host tail). Each dot is GPU partial + CPU partial summed
// on the host (single-node analog of MPI_Allreduce). The pipelined structure
// does 2 such combines/iter (vs 5 dots in standard). SpMV is split with halo
// gather. Residual replacement (Sec 4.2) via env HHP_REPLACE=k.

struct arguments { char *input_matrix,*input_x,*input_y,*input_part,*input_gpu,*output_x; int n_iters; };
static struct argp_option options[] = {
    {"input-matrix",'m',"FILE",0,"matrix"},{"input-x",'x',"FILE",0,"X"},{"input-y",'y',"FILE",0,"B"},
    {"input-part",'p',"FILE",0,"partition"},{"is-gpu",'g',"FILE",0,"is_gpu"},{"output-x",'o',"FILE",0,"out"},
    {"n-iters",'n',"POSITIVE-INTEGER",0,"iters"},{0}};
static error_t parse_opt(int key, char *arg, struct argp_state *st){
    struct arguments *a=st->input; char*b=NULL;
    switch(key){case 'm':a->input_matrix=arg;break;case 'x':a->input_x=arg;break;case 'y':a->input_y=arg;break;
        case 'p':a->input_part=arg;break;case 'g':a->input_gpu=arg;break;case 'o':a->output_x=arg;break;
        case 'n':a->n_iters=strtol(arg,&b,10);break;case ARGP_KEY_ARG:return 0;default:return ARGP_ERR_UNKNOWN;}
    return 0;
}
static char doc[] = "Hybrid CPU+GPU pipelined BiCGStab (Alg. 9, no MPI)";
static struct argp argp = {options, parse_opt, "", doc};

typedef struct {
    cusparseHandle_t ch; cusparseSpMatDescr_t Agpu; cusparseDnVecDescr_t in_desc;
    double *d_full, *h_full; CSR Acpu; Device_Buffer_SpMV buf; int ng, nc, n;
    cudaStream_t cs, ps; cudaEvent_t in_ready, h2d;
    int *d_halo, *h_halo, nh; double *d_haloval, *h_haloval;
    unsigned long long *flagd; volatile unsigned long long *flagv, seq;  // mapped-host spin-wait
} Ctx;

// out = A * in (split). Halo-gathered: CPU only receives the GPU columns its rows touch.
static void spmv(Ctx *c, cusparseDnVecDescr_t out_desc, const double *d_in, const double *h_in,
                 double *d_out, double *h_out) {
    const double one=1.0, zero=0.0; int ng=c->ng, nc=c->nc;
    cudaEventRecord(c->in_ready, c->cs); cudaStreamWaitEvent(c->ps, c->in_ready, 0);
    cudaMemcpyAsync(c->d_full, d_in, ng*sizeof(double), cudaMemcpyDeviceToDevice, c->cs);
    if (nc > 0) {
        cudaMemcpyAsync(c->d_full+ng, h_in, nc*sizeof(double), cudaMemcpyHostToDevice, c->ps);
        hhp_dp_gather(c->d_haloval, d_in, c->d_halo, c->nh, c->ps);
        cudaMemcpyAsync(c->h_haloval, c->d_haloval, c->nh*sizeof(double), cudaMemcpyDeviceToHost, c->ps);
        memcpy(c->h_full+ng, h_in, nc*sizeof(double));
        cudaEventRecord(c->h2d, c->ps); cudaStreamWaitEvent(c->cs, c->h2d, 0);
    }
    cusparseSpMV(c->ch, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, c->Agpu, c->in_desc, &zero, out_desc,
                 CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, c->buf);
    if (nc > 0) {
        hhp_dp_set_flag(c->flagd, ++c->seq, c->ps);            // spin-wait for the halo D->H
        while (*c->flagv < c->seq) {}
        #pragma omp parallel for
        for (int k=0;k<c->nh;k++) c->h_full[c->h_halo[k]] = c->h_haloval[k];
        Vector y={.vals=h_out,.nvals=nc}, xx={.vals=c->h_full,.nvals=c->n};
        CSR_spmxv_omp(c->Acpu, xx, y);
    }
    // no GPU sync here: callers' subsequent device ops are CS-ordered after this SpMV.
}

// distributed vector ops: GPU slice (cuBLAS host-ptr, stream CS) + CPU slice (OMP)
static cublasHandle_t BH; static cudaStream_t CS; static int NG, NC;
static void v_scal(double a, double *dv, double *hv){ cublasDscal(BH, NG, &a, dv, 1); host_scal(a, hv, NC); }
static void v_axpy(double a, const double *dx, const double *hx, double *dy, double *hy){
    cublasDaxpy(BH, NG, &a, dx, 1, dy, 1); host_axpy(a, hx, hy, NC); }
static void v_copy(const double *dx, const double *hx, double *dy, double *hy){
    cudaMemcpyAsync(dy, dx, NG*sizeof(double), cudaMemcpyDeviceToDevice, CS); host_copy(hx, hy, NC); }
// global dot = GPU partial (host-ptr cublasDdot; blocks on CS) + CPU partial (OMP)
static double v_dot(const double *da, const double *db, const double *ha, const double *hb){
    double g=0; cublasDdot(BH, NG, da, 1, db, 1, &g); return g + host_dot(ha, hb, NC); }

// fused distributed recurrence ops: one GPU kernel (stream CS) + one fused OMP loop
static void p_axyz(double a, double c, double *od, double *oh, const double *Yd, const double *Yh, const double *Zd, const double *Zh){
    hhp_pipe_axyz(od, Yd, Zd, a, c, NG, CS);
    #pragma omp parallel for
    for (int i=0;i<NC;i++) oh[i] = a*oh[i] + Yh[i] + c*Zh[i]; }   // out = a*out + Y + c*Z
static void p_xcy(double c, double *od, double *oh, const double *Xd, const double *Xh, const double *Yd, const double *Yh){
    hhp_pipe_xcy(od, Xd, Yd, c, NG, CS);
    #pragma omp parallel for
    for (int i=0;i<NC;i++) oh[i] = Xh[i] + c*Yh[i]; }             // out = X + c*Y
static void p_acc(double a, double b, double *od, double *oh, const double *Xd, const double *Xh, const double *Yd, const double *Yh){
    hhp_pipe_acc(od, Xd, Yd, a, b, NG, CS);
    #pragma omp parallel for
    for (int i=0;i<NC;i++) oh[i] += a*Xh[i] + b*Yh[i]; }          // out += a*X + b*Y
static void p_xbycz(double b, double c, double *od, double *oh, const double *Xd, const double *Xh, const double *Yd, const double *Yh, const double *Zd, const double *Zh){
    hhp_pipe_xbycz(od, Xd, Yd, Zd, b, c, NG, CS);
    #pragma omp parallel for
    for (int i=0;i<NC;i++) oh[i] = Xh[i] + b*Yh[i] + c*Zh[i]; }   // out = X + b*Y + c*Z

int main(int argc, char *argv[]) {
    struct arguments A = {}; A.n_iters = 1;
    argp_parse(&argp, argc, argv, 0, 0, &A);
    if (!A.input_matrix||!A.input_x||!A.input_y||!A.input_part||!A.input_gpu||!A.output_x){
        fprintf(stderr,"need -m -x -y -p -g -o\n"); return EXIT_FAILURE; }
    int niters=A.n_iters; if(niters<1) niters=1;
    int replace_k = getenv("HHP_REPLACE") ? atoi(getenv("HHP_REPLACE")) : 0;
    double t_begin = omp_get_wtime();

    double t_read0 = omp_get_wtime();
    CSR M = buReadSparseMatrix(A.input_matrix); int n = M.n;
    Vector Xin = vector_read(A.input_x, n), Bin = vector_read(A.input_y, M.m);
    printf("Matrix name : %s\n", A.input_matrix);
    int isg[8]; int nr = read_ints(A.input_gpu, isg, 8); if(nr!=2) ABORT("is_gpu needs 2")
    int grank=-1,gc=0; for(int r=0;r<2;r++) if(isg[r]==1){grank=r;gc++;} if(gc!=1) ABORT("one GPU rank")
    int crank=1-grank;
    int *part; ALLOC_ARRAY(part,M.m); int np=read_ints(A.input_part,part,M.m); if(np!=M.m) ABORT("part size")
    int *perm; ALLOC_ARRAY(perm,M.m); int *pinv; ALLOC_ARRAY(pinv,M.m);
    int ng=0,nc=0; for(int i=0;i<M.m;i++) if(part[i]==grank) perm[ng++]=i;
    int ge=ng; for(int i=0;i<M.m;i++) if(part[i]==crank) perm[ge+nc++]=i;
    for(int i=0;i<M.m;i++) pinv[perm[i]]=i;
    printf("n_gpu=%d n_cpu=%d total=%d\n", ng, nc, M.m);
    CSR Ap = csr_permute(M, perm, pinv); freeSparseMatrix(&M);
    Vector Xp=vector_init(n), Bp=vector_init(n);
    for(int i=0;i<n;i++) Xp.vals[i]=Xin.vals[perm[i]];
    for(int i=0;i<n;i++) Bp.vals[i]=Bin.vals[perm[i]];
    vector_destroy(&Xin); vector_destroy(&Bin);
    CSR Agpu=csr_row_slice(Ap,0,ng), Acpu=csr_row_slice(Ap,ng,ng+nc);
    double t_read1 = omp_get_wtime();

    CHECK_CUDA(cudaSetDeviceFlags(cudaDeviceMapHost))
    CHECK_CUDA(cudaSetDevice(0))
    cublasHandle_t bh; CHECK_CUBLAS(cublasCreate(&bh)) cusparseHandle_t ch; CHECK_CUSPARSE(cusparseCreate(&ch))
    cudaStream_t cs, ps; CHECK_CUDA(cudaStreamCreate(&cs)) CHECK_CUDA(cudaStreamCreate(&ps))
    CHECK_CUBLAS(cublasSetStream(bh, cs)) CHECK_CUSPARSE(cusparseSetStream(ch, cs))
    cudaEvent_t e_in, e_h2d; CHECK_CUDA(cudaEventCreateWithFlags(&e_in,cudaEventDisableTiming))
    CHECK_CUDA(cudaEventCreateWithFlags(&e_h2d,cudaEventDisableTiming))
    Device_CSR dAgpu; CHECK_CUSPARSE(device_csr_create(Agpu,&dAgpu))
    BH=bh; CS=cs; NG=ng; NC=nc;

    #define DV() dvec(ng)
    #define HV() ((double*)malloc((nc>0?nc:1)*sizeof(double)))
    double *dX=DV(),*dB=DV(),*dR=DV(),*dR0=DV(),*dW=DV(),*dT=DV(),*dP=DV(),*dS=DV(),*dZ=DV(),*dQ=DV(),*dY=DV(),*dV=DV(),*dTMP=DV();
    double *hX=HV(),*hB=HV(),*hR=HV(),*hR0=HV(),*hW=HV(),*hT=HV(),*hP=HV(),*hS=HV(),*hZ=HV(),*hQ=HV(),*hY=HV(),*hV=HV(),*hTMP=HV();
    CHECK_CUDA(cudaMemcpy(dX,Xp.vals,ng*sizeof(double),cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dB,Bp.vals,ng*sizeof(double),cudaMemcpyHostToDevice))
    for(int i=0;i<nc;i++){hX[i]=Xp.vals[ng+i]; hB[i]=Bp.vals[ng+i];}

    double *d_full=dvec(n), *h_full; CHECK_CUDA(cudaMallocHost((void**)&h_full,n*sizeof(double)))
    cusparseDnVecDescr_t in_desc, Wd,Td,Vd,Sd,Zd,TMd;
    int g1 = ng>0?ng:1;
    CHECK_CUSPARSE(cusparseCreateDnVec(&in_desc,n,d_full,CUDA_R_64F))
    CHECK_CUSPARSE(cusparseCreateDnVec(&Wd,g1,dW,CUDA_R_64F))  CHECK_CUSPARSE(cusparseCreateDnVec(&Td,g1,dT,CUDA_R_64F))
    CHECK_CUSPARSE(cusparseCreateDnVec(&Vd,g1,dV,CUDA_R_64F))  CHECK_CUSPARSE(cusparseCreateDnVec(&Sd,g1,dS,CUDA_R_64F))
    CHECK_CUSPARSE(cusparseCreateDnVec(&Zd,g1,dZ,CUDA_R_64F))  CHECK_CUSPARSE(cusparseCreateDnVec(&TMd,g1,dTMP,CUDA_R_64F))
    Device_Buffer_SpMV sbuf; { size_t bs=0; const double a=1,b=0;
      CHECK_CUSPARSE(cusparseSpMV_bufferSize(ch,CUSPARSE_OPERATION_NON_TRANSPOSE,&a,dAgpu.desc,in_desc,&b,Wd,CUDA_R_64F,CUSPARSE_SPMV_ALG_DEFAULT,&bs))
      CHECK_CUDA(cudaMalloc(&sbuf,bs)) }

    int nh=0; int *halo=NULL;
    { char *seen=calloc(ng>0?ng:1,1);
      for(int k=0;k<Acpu.nnz;k++){int cc=Acpu.J[k]; if(cc<ng&&!seen[cc]){seen[cc]=1;nh++;}}
      halo=malloc((nh>0?nh:1)*sizeof(int)); int idx=0; for(int cc=0;cc<ng;cc++) if(seen[cc]) halo[idx++]=cc; free(seen); }
    int *d_halo; CHECK_CUDA(cudaMalloc((void**)&d_halo,(nh>0?nh:1)*sizeof(int)))
    if(nh>0) CHECK_CUDA(cudaMemcpy(d_halo,halo,nh*sizeof(int),cudaMemcpyHostToDevice))
    double *d_haloval=dvec(nh), *h_haloval; CHECK_CUDA(cudaMallocHost((void**)&h_haloval,(nh>0?nh:1)*sizeof(double)))
    printf("halo: %d / %d\n", nh, ng);

    Ctx ctx = {.ch=ch,.Agpu=dAgpu.desc,.in_desc=in_desc,.d_full=d_full,.h_full=h_full,.Acpu=Acpu,.buf=sbuf,
               .ng=ng,.nc=nc,.n=n,.cs=cs,.ps=ps,.in_ready=e_in,.h2d=e_h2d,
               .d_halo=d_halo,.h_halo=halo,.nh=nh,.d_haloval=d_haloval,.h_haloval=h_haloval};
    unsigned long long *flagh; CHECK_CUDA(cudaHostAlloc((void**)&flagh,sizeof(unsigned long long),cudaHostAllocMapped))
    *flagh=0; CHECK_CUDA(cudaHostGetDevicePointer((void**)&ctx.flagd,(void*)flagh,0)) ctx.flagv=flagh; ctx.seq=0;

    // --- init: r = b - A x ; R0 = r ; w = A r ; t = A w ; p=s=z=v=0 ---
    spmv(&ctx, TMd, dX, hX, dTMP, hTMP);                     // TMP = A x
    v_copy(dB,hB,dR,hR); v_axpy(-1.0,dTMP,hTMP,dR,hR);       // r = b - A x
    v_copy(dR,hR,dR0,hR0);                                   // R0 = r
    spmv(&ctx, Wd, dR, hR, dW, hW);                          // w = A r
    spmv(&ctx, Td, dW, hW, dT, hT);                          // t = A w
    CHECK_CUDA(cudaMemset(dP,0,ng*sizeof(double))) CHECK_CUDA(cudaMemset(dS,0,ng*sizeof(double)))
    CHECK_CUDA(cudaMemset(dZ,0,ng*sizeof(double))) CHECK_CUDA(cudaMemset(dV,0,ng*sizeof(double)))
    for(int i=0;i<nc;i++){hP[i]=0;hS[i]=0;hZ[i]=0;hV[i]=0;}

    double rho_old = v_dot(dR0,dR,hR0,hR);
    double alpha = rho_old / v_dot(dR0,dW,hR0,hW);
    double beta_prev=0.0, omega_prev=1.0, omega=0.0;
    double bnorm = sqrt(v_dot(dB,dB,hB,hB));
    printf("LOG: setup done\n");

    double t_loop0 = omp_get_wtime();
    for (int i=0;i<niters;i++){
        double bw = beta_prev*omega_prev;
        p_axyz(beta_prev,-bw, dP,hP, dR,hR, dS,hS);   // p = beta*p + r - bw*s
        p_axyz(beta_prev,-bw, dS,hS, dW,hW, dZ,hZ);   // s = beta*s + w - bw*z
        p_axyz(beta_prev,-bw, dZ,hZ, dT,hT, dV,hV);   // z = beta*z + t - bw*v
        p_xcy(-alpha, dQ,hQ, dR,hR, dS,hS);           // q = r - alpha*s
        p_xcy(-alpha, dY,hY, dW,hW, dZ,hZ);           // y = w - alpha*z
        // reduction 1 (omega) + SpMV v = A z
        double qy = v_dot(dQ,dY,hQ,hY), yy = v_dot(dY,dY,hY,hY);
        spmv(&ctx, Vd, dZ, hZ, dV, hV);
        omega = qy/yy;
        p_acc(alpha,omega, dX,hX, dP,hP, dQ,hQ);                  // x += alpha*p + omega*q
        p_xcy(-omega, dR,hR, dQ,hQ, dY,hY);                       // r = q - omega*y
        p_xbycz(-omega, omega*alpha, dW,hW, dY,hY, dT,hT, dV,hV); // w = y - omega*t + omega*alpha*v
        if (replace_k>0 && (i+1)%replace_k==0) {                                      // residual replacement
            spmv(&ctx,TMd,dX,hX,dTMP,hTMP); v_copy(dB,hB,dR,hR); v_axpy(-1.0,dTMP,hTMP,dR,hR);
            spmv(&ctx,Wd,dR,hR,dW,hW); spmv(&ctx,Td,dW,hW,dT,hT);
            spmv(&ctx,Sd,dP,hP,dS,hS); spmv(&ctx,Zd,dS,hS,dZ,hZ); spmv(&ctx,Vd,dZ,hZ,dV,hV);
        }
        spmv(&ctx, Td, dW, hW, dT, hT);                                              // t = A w (next iter)
        // reduction 2 (beta, alpha)
        double rr=v_dot(dR0,dR,hR0,hR), rw=v_dot(dR0,dW,hR0,hW), rs=v_dot(dR0,dS,hR0,hS), rz=v_dot(dR0,dZ,hR0,hZ);
        double beta = (alpha/omega)*(rr/rho_old);
        double denom = rw + beta*rs - beta*omega*rz;
        rho_old=rr; beta_prev=beta; omega_prev=omega; alpha = rr/denom;
    }
    cudaDeviceSynchronize();
    double t_loop1 = omp_get_wtime();

    spmv(&ctx, TMd, dX, hX, dTMP, hTMP); v_axpy(-1.0,dB,hB,dTMP,hTMP);
    double rel = sqrt(v_dot(dTMP,dTMP,hTMP,hTMP)) / bnorm;

    printf("n_iters : %d \n", niters);
    printf("spmv : %lf \n", t_loop1-t_loop0);
    printf("file_read : %lf \n", t_read1-t_read0);
    printf("relative_residual : %E\n", rel);
    printf("everything_total : %lf\n", omp_get_wtime()-t_begin);
    printf("\n----------------------------------------------------------------------\n");
    return EXIT_SUCCESS;
}
