#ifndef HHP_PROF_H
#define HHP_PROF_H

// Unified optional profiling for all BiCGStab solvers. Enabled by env HHP_PROF:
//   unset/0 -> off (zero overhead; every call early-returns)
//   1       -> accumulate per-section wall time, print a summary to stderr at end
//   2       -> also record per-iteration times and dump them to a TSV file
//
// Device-agnostic: GPU solvers pass a sync callback (e.g. cudaDeviceSynchronize)
// so async work is attributed to the right section; CPU/MPI solvers pass NULL.
//
// Usage per iteration:
//   prof_tick(&P);
//   spmv(...);   prof_lap(&P, PF_SPMV);
//   dots(...);   prof_lap(&P, PF_DOT);
//   vecops(...); prof_lap(&P, PF_VECOPS);
//   ...          prof_lap(&P, PF_COMM / PF_ALLREDUCE / PF_SYNC);
//   prof_iter_end(&P);
// Around the whole loop: prof_set_preprocess(&P, t) once; prof_report(...) at end.

#include <stddef.h>

typedef enum {
    PF_SPMV = 0,   // matrix-vector multiply (hybrid GPU+CPU slice)
    PF_DOT,        // dot products
    PF_VECOPS,     // axpy / scal / recurrence vector updates
    PF_COMM,       // CPU<->GPU transfers, halo gather, H2D/D2H
    PF_ALLREDUCE,  // MPI_Allreduce (MPI solvers); CPU<->GPU scalar rendezvous otherwise
    PF_SYNC,       // spin-wait / stream sync
    PF_OTHER,
    PF_NSEC
} ProfSec;

extern const char *PROF_NAMES[PF_NSEC];

typedef void (*prof_sync_fn)(void *);   // e.g. wraps cudaDeviceSynchronize

typedef struct {
    int level;                  // 0 off, 1 summary, 2 + per-iteration
    double t0;                  // last lap timestamp
    double preprocess;          // one-off setup time
    double sec[PF_NSEC];        // whole-run accumulators
    double itsec[PF_NSEC];      // current-iteration accumulators
    double *iter[PF_NSEC];      // per-iteration arrays (level 2)
    int niter, cap;
    prof_sync_fn sync; void *sync_arg;
} Prof;

// Reads HHP_PROF. max_iters sizes the per-iteration arrays (level 2). sync/arg
// may be NULL (no device sync).
void prof_init(Prof *p, int max_iters, prof_sync_fn sync, void *sync_arg);
void prof_set_preprocess(Prof *p, double seconds);

// Start timing the next section boundary (sync first if a callback is set).
static inline void prof_tick(Prof *p);
// Attribute elapsed-since-last-lap/tick to section s (sync first), then re-arm.
static inline void prof_lap(Prof *p, ProfSec s);
// Close the current iteration: push itsec[] into the per-iter arrays, zero itsec[].
void prof_iter_end(Prof *p);

// stderr summary; at level 2 also writes data/prof/<solver>__<matrix>.tsv.
void prof_report(Prof *p, const char *solver, const char *matrix_path);
void prof_free(Prof *p);

// --- inline hot path (defined here so the off-case is a cheap branch) ---
double hhp_wtime(void);   // omp_get_wtime wrapper (impl in hhp_prof.c)

static inline void prof_tick(Prof *p) {
    if (!p->level) return;
    if (p->sync) p->sync(p->sync_arg);
    p->t0 = hhp_wtime();
}
static inline void prof_lap(Prof *p, ProfSec s) {
    if (!p->level) return;
    if (p->sync) p->sync(p->sync_arg);
    double t = hhp_wtime();
    double d = t - p->t0;
    p->sec[s] += d;
    p->itsec[s] += d;
    p->t0 = t;
}

#endif // HHP_PROF_H
