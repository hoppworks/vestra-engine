# Iteration 96 — strip-major hidden-panel MLP schedule (reverted)

## Hypothesis and isolated change

The accepted hidden-strip MLP route assigns one 55-token slab to each Rayon
worker and executes all 48 consecutive 64-neuron strips in that worker.  This
candidate inverted only that scheduling axis: all workers processed strip 0,
then all processed strip 1, and so on.  FC1, bias, GELU, FC2, and the FC2
accumulation order within every output element remained unchanged.  The
hypothesis was that a contemporaneously used FC1/FC2 panel would remain hotter
in cache across the CCDs than in the slab-major route.

The implementation was opt-in through `DA3_STRIP_MLP_STRIP_MAJOR=1`; the
accepted route remained available unchanged.  The Engine library test suite
passed locally before remote measurement (72 tests).  No correctness claim is
made from this result because the candidate failed the performance smoke gate
and was removed before the four-image parity gate.

## Locked smoke measurement

The idle Workhorse was verified to have no unrelated `vestra-engine`, Cargo,
or `da3-cli` process.  Both arms used Rust 1.93.0, the same DA3 BASE F32 GGUF,
`mountains.jpg`, 504x336 preprocessing, 16 Rayon/OpenMP threads, the accepted
BLIS/Head flags, and one warm-up plus ten timed iterations per invocation.
The two prebuilt binaries were run as five alternating control/candidate
pairs.  This quick screen records the command's p95 output because it was
already decisively worse; it is not a qualifying trial study.

| Pair | Accepted slab-major p95 (ms) | Strip-major p95 (ms) |
|---:|---:|---:|
| 1 | 171.928 | 186.891 |
| 2 | 166.743 | 187.474 |
| 3 | 170.423 | 204.782 |
| 4 | 173.859 | 184.799 |
| 5 | 186.170 | 216.172 |

The candidate was slower in all five pairs (roughly 10–20% by this screening
metric).  Allocating all live hidden slab buffers and repeatedly rescheduling
the same token slabs between hidden strips defeated the intended panel-locality
benefit.  It therefore cannot contribute to the `<= 131.959 ms` goal and was
cleanly removed without a full benchmark or parity run.

## Decision

Rejected.  The next MLP experiment must retain slab ownership and reduce FC2
output traffic *within* a slab (for example a four-strip output-panel
supertile), rather than exchanging slab locality for shared weight-panel
scheduling.
