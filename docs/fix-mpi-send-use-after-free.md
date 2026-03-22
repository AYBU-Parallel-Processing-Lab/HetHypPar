# Fix: MPI Send Buffer Use-After-Free in `internal_setup_communication`

**Date:** 2026-03-22
**File:** `src/util/hhp_matrix.c` — `internal_setup_communication()`
**Impact:** 5,429 segfaults out of 32,242 benchmark runs (84% of all failures)

## Problem

The second round of MPI communication in `internal_setup_communication` exchanged global column indices between ranks using a fire-and-forget pattern:

```c
MPI_Isend(recv_gJ.vals + ..., size, MPI_INT, target, 0, MPI_COMM_WORLD, &temp_req);
MPI_Request_free(&temp_req);  // detach — no way to wait on this send
```

After posting all sends and receives, the code called `MPI_Waitall` only on the **receive** requests, then immediately freed the send buffer (`recv_gJ`).

`MPI_Request_free` detaches the request handle but does **not** guarantee the underlying send operation has completed. If MPI is still reading from `recv_gJ` when it gets freed, the receiving rank gets garbage data in `send.J`. The subsequent local-index lookup (`send.J[i] = locmap.vals[send.J[i]]`) then dereferences a garbage index, causing a segfault.

## Why It Was Non-Deterministic

The crash depended on the relative timing of two MPI processes. When one rank finished its local work (CSR splitting, communication setup) much faster than the other, it would free `recv_gJ` before MPI finished delivering the data to the slower rank.

## Why Specific Matrices Were Affected

Structurally asymmetric matrices create imbalanced communication patterns — one rank has many more shared columns than the other. Combined with lopsided partition weights (e.g., 90/10, 75/25), this increases the processing-time gap between ranks, making the race condition more likely to trigger.

Most affected matrices and their segfault rates:

| Matrix       | Segfault Rate |
|-------------|:------------:|
| g7jac080sc  | 97%          |
| lhr14       | 92%          |
| bayer02     | 90%          |
| fd18        | 88%          |
| g7jac080    | 97%          |
| g7jac100sc  | 90%          |

Symmetric matrices (e.g., `cage11`, `airfoil_2d`) were unaffected because both ranks had similar workloads and timing.

## Fix

Replace fire-and-forget sends with tracked requests and wait for send completion before freeing the buffer:

```c
// Track send requests
MPI_Request *send_reqs2 = NULL;
if (result->recv.num > 0)
    ALLOC_ARRAY(send_reqs2, result->recv.num);
for (size_t i = 0; i < result->recv.num; i++) {
    int target = result->recv.ranks[i];
    int size = result->recv.I[i+1] - result->recv.I[i];
    MPI_Isend(recv_gJ.vals + result->recv.I[i], size, MPI_INT,
              target, 0, MPI_COMM_WORLD, send_reqs2 + i);
}

// ... receive code unchanged ...

MPI_Waitall(send.num, reqs, stats);          // wait for receives
MPI_Waitall(result->recv.num, send_reqs2,    // wait for sends
            MPI_STATUSES_IGNORE);
FREE_AND_NULL(send_reqs2);

// Now safe to free recv_gJ
ivector_destroy(&recv_gJ);
```

## Verification

Tested with the four worst-offending matrices, 10 runs each, all using lopsided 2-rank partitions:

| Matrix      | Before Fix | After Fix |
|------------|:----------:|:---------:|
| lhr14      | ~1/10 pass | 10/10     |
| bayer02    | ~1/10 pass | 10/10     |
| g7jac080sc | ~0/10 pass | 10/10     |
| fd18       | ~1/10 pass | 10/10     |

## Lesson

Never use `MPI_Isend` + `MPI_Request_free` if the send buffer will be freed before a synchronization point that guarantees send completion. `MPI_Request_free` only detaches the handle — MPI may still be reading the buffer. Either track the request and `MPI_Waitall`, or use `MPI_Ssend` / `MPI_Barrier` to ensure completion.
