# Heterogeneous BiCGStab benchmark — findings

A narrative summary of the large sweep. The raw numbers live in
`data/big_benchmark/results.parquet` (and `summary.md` for the tables); this
file is the prose version.

## What was tested

Every solver variant we have built — ten of them, from the plain single-process
CPU and GPU solvers through the device-pointer, pipelined, and hybrid CPU+GPU
families — was run on **fourteen sparse matrices** spanning circuit, DNA-lattice,
CFD, FEM and optimization problems, ranging from a few million up to **111
million nonzeros**. For the hybrid solvers, which divide rows between the GPU and
the CPU, we swept **twelve GPU:CPU work ratios** and, for each, tried **two ways
of choosing the split**: PaToH hypergraph partitioning (which minimizes the
communication cut) and a naive contiguous split that simply balances nonzeros and
ignores the cut entirely. Everything ran on a single node (one RTX 3070, 8 GB)
in strict series so the timings are clean. Each number reported is the best of
two runs at the best OpenMP thread count.

## Which solver wins

The two **device-pointer hybrids dominate**. Measured against the original
`bicgstab-gpu`, `hybrid-async-dp` is usually the fastest, landing around
1.2–1.28×, and `hybrid-dist-dp` is close behind and remarkably consistent at
1.14–1.27× across every matrix. The pure-GPU device-pointer solver, `gpu-dp`,
also does well on its own (1.04–1.32×) without any CPU help at all. Everything
that distributes across MPI, or that pipelines the iteration, **loses on a single
node** — `mpi-gpu` is erratic and often below 1×, and both pipelined variants sit
firmly under 1×, exactly as the earlier analysis predicted (pipelining trades
extra local work for hidden communication that simply isn't expensive enough on
one machine to be worth hiding). The plain CPU and MPI-CPU solvers are an order
of magnitude slower, as expected for a memory-bandwidth-bound kernel with no GPU.

## The honest baseline: is the CPU split actually worth it?

Measuring against the *old* `bicgstab-gpu` flatters the hybrids, because `gpu-dp`
already beats that baseline by roughly 1.2× for free — pure GPU, no CPU
involvement. The fairer question is whether adding the CPU beats just running
`gpu-dp`. Against that baseline the picture is more sober: the heterogeneous
split nets a real **6–14% on about half the matrices** (the regular, FEM-like
ones), is a **wash** on a few, and actually **loses** on the irregular circuit
matrices (`Freescale2`, `circuit5M_dc`) and on `Hamrle3`, where bringing the CPU
in costs more in split-and-synchronize overhead than its extra memory bandwidth
contributes. So the CPU offload is worth it **conditionally** — it depends on the
matrix structure, helping on regular problems and hurting on irregular ones.

## async-dp versus dist-dp: peak versus robustness

The two leading hybrids differ in *how much* they put on the CPU.
`hybrid-async-dp` splits **only the matrix-vector multiply** and keeps all dot
products and vector updates on the GPU, so it synchronizes rarely; `hybrid-dist-dp`
splits **everything** — the vectors themselves are partitioned and each dot
becomes a GPU-partial plus a CPU-partial combined every iteration — so the CPU
carries a full slice of the work but pays far more coordination.

That difference shows up clearly against the `gpu-dp` baseline. `async-dp` has the
**higher peaks** but is **fragile**: when the SpMV split happens to be unbalanced
on an irregular matrix it has nothing else to fall back on, and it craters
(0.75× on `Hamrle3`, 0.85× on `circuit5M_dc`). `dist-dp` spreads the CPU's
contribution across the whole iteration, so it **never loses badly** — its worst
case is 0.92× — while matching `async-dp` on the easy matrices. If you had to
ship a single hybrid, **`dist-dp` is the safer choice**; `async-dp` is the one to
reach for only when you know the matrix partitions cleanly.

## Does hypergraph partitioning help? Mostly no.

This was the headline question, and the answer is a clear and slightly
surprising **no — not for the configuration that matters**. Comparing PaToH
against the naive nonzero-balanced split at the same work ratio, the benefit
tracks **how much a solver communicates, not how big the matrix is**. For the
SpMV-split solvers (`async` and `async-dp`, including the overall winner) PaToH
and naive are **statistically tied** — within about 1% on average, and that holds
just as firmly on the 100-million-nonzero matrices as on the small ones. PaToH
buys a modest ~2% for the fully-distributed solvers (which exchange dot products
too) and ~5% for `mpi-gpu` (which communicates the most). The intuition is
simple: with only two parts and a thin CPU slice, the communication cut is a tiny
fraction of the work either way, so minimizing it — the expensive thing PaToH
does — optimizes something that isn't the bottleneck. We had expected the big
matrices to be where cut-minimization finally paid off; the data refutes that.
**For two-way GPU+CPU SpMV splitting, a dumb contiguous nonzero-balance is as good
as hypergraph partitioning, at every scale.**

## Caveats and one casualty

A few things to keep in mind. All of this is **single-node**; the pipelined and
MPI results are expected to look very different across multiple physical nodes,
where communication is genuinely expensive — that regime was not measured here.
The reported residuals are mostly unusable (the cage and circuit matrices print a
placeholder residual that doesn't parse), so these are **timing** comparisons; the
pipelined solvers, which compute the residual differently, confirmed convergence
to ~1e-16. One matrix, `nlpkkt120`, is excluded entirely — it crashes even the
plain GPU solver, a matrix/vector issue unrelated to anything we tested.

Finally, a process note: PaToH originally failed on the five largest matrices.
The cause turned out to be a stack-allocated array sized by the row count in the
`patpart` wrapper, which overflows the 8 MB stack above ~2.1 million rows; raising
the child stack limit fixed it cleanly and let us complete the comparison on the
big matrices (which is what confirmed the "cut doesn't matter at scale" result).

## Bottom line

On a single GPU, the best thing you can do is use **device-pointer dots**, and
that alone (`gpu-dp`) gets you most of the way. Adding the **CPU via an SpMV split
helps another ~5–14% on regular matrices but can hurt on irregular ones**, so
`dist-dp` is the robust default and `async-dp` the high-ceiling option for
well-structured problems. And the **expensive hypergraph partitioner is
unnecessary** for this two-way split — naive nonzero-balancing is just as fast.
The communication-reduction techniques (pipelining, MPI) are multi-node tools and
shouldn't be expected to pay off on one machine.
