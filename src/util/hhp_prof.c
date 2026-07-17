#include "hhp_prof.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

// Section names, indexed by ProfSec. Kept short for the stderr table / TSV header.
const char *PROF_NAMES[PF_NSEC] = {
    "spmv", "dot", "vecops", "comm", "allreduce", "sync", "other"
};

double hhp_wtime(void) { return omp_get_wtime(); }

void prof_init(Prof *p, int max_iters, prof_sync_fn sync, void *sync_arg) {
    memset(p, 0, sizeof(*p));
    const char *e = getenv("HHP_PROF");
    p->level = e ? atoi(e) : 0;
    p->sync = sync;
    p->sync_arg = sync_arg;
    if (p->level >= 2) {
        p->cap = max_iters > 0 ? max_iters : 1;
        for (int s = 0; s < PF_NSEC; s++)
            p->iter[s] = (double *)calloc((size_t)p->cap, sizeof(double));
    }
}

void prof_set_preprocess(Prof *p, double seconds) {
    if (!p->level) return;
    p->preprocess = seconds;
}

void prof_iter_end(Prof *p) {
    if (!p->level) return;
    if (p->level >= 2 && p->niter < p->cap) {
        for (int s = 0; s < PF_NSEC; s++)
            p->iter[s][p->niter] = p->itsec[s];
    }
    for (int s = 0; s < PF_NSEC; s++) p->itsec[s] = 0.0;
    p->niter++;
}

// Strip directory and extension from a path -> caller-provided buffer.
static void base_stem(const char *path, char *out, size_t cap) {
    if (!path) { snprintf(out, cap, "unknown"); return; }
    const char *slash = strrchr(path, '/');
    const char *b = slash ? slash + 1 : path;
    snprintf(out, cap, "%s", b);
    char *dot = strrchr(out, '.');
    if (dot && dot != out) *dot = '\0';
}

void prof_report(Prof *p, const char *solver, const char *matrix_path) {
    if (!p->level) return;

    double total = 0.0;
    for (int s = 0; s < PF_NSEC; s++) total += p->sec[s];
    int ni = p->niter > 0 ? p->niter : 1;

    fprintf(stderr, "\n[HHP_PROF] solver=%s  iters=%d  preprocess=%.4fs\n",
            solver ? solver : "?", p->niter, p->preprocess);
    fprintf(stderr, "  %-10s %12s %12s %8s\n", "section", "total_s", "us/iter", "%loop");
    for (int s = 0; s < PF_NSEC; s++) {
        if (p->sec[s] == 0.0) continue;
        fprintf(stderr, "  %-10s %12.6f %12.2f %7.1f%%\n",
                PROF_NAMES[s], p->sec[s], 1e6 * p->sec[s] / ni,
                total > 0 ? 100.0 * p->sec[s] / total : 0.0);
    }
    fprintf(stderr, "  %-10s %12.6f %12.2f\n", "loop_total", total, 1e6 * total / ni);

    if (p->level < 2) return;

    // Per-iteration dump: data/prof/<solver>__<matrix>.tsv
    char stem[256];
    base_stem(matrix_path, stem, sizeof(stem));
    mkdir("data", 0777);
    mkdir("data/prof", 0777);
    char fname[600];
    snprintf(fname, sizeof(fname), "data/prof/%s__%s.tsv",
             solver ? solver : "solver", stem);
    FILE *f = fopen(fname, "w");
    if (!f) {
        fprintf(stderr, "[HHP_PROF] could not open %s for writing\n", fname);
        return;
    }
    fprintf(f, "iter");
    for (int s = 0; s < PF_NSEC; s++) fprintf(f, "\t%s", PROF_NAMES[s]);
    fprintf(f, "\n");
    for (int i = 0; i < p->niter && i < p->cap; i++) {
        fprintf(f, "%d", i);
        for (int s = 0; s < PF_NSEC; s++) fprintf(f, "\t%.9g", p->iter[s][i]);
        fprintf(f, "\n");
    }
    fclose(f);
    fprintf(stderr, "[HHP_PROF] per-iteration dump -> %s\n", fname);
}

void prof_free(Prof *p) {
    for (int s = 0; s < PF_NSEC; s++) {
        free(p->iter[s]);
        p->iter[s] = NULL;
    }
}
