# Hidden-strip 54-row scheduling — rejected

## Hypothesis

The accepted hidden-strip MLP schedules the locked 865 DA3 tokens as fifteen
55-row Rayon slabs plus one 40-row slab. A 54-row slab schedule creates
sixteen almost equal slabs plus a one-row tail, potentially reducing the last
worker imbalance while preserving FC1/FC2 arithmetic and panel order.

## Change and gates

Only `SLAB_ROWS` in `PackedMlpExecutor::run_hidden_strip_mlp` changed from 55
to 54. The helper's existing row-boundary regression test was extended to
exercise 54 rows. `cargo test --locked -p vestra-engine --lib` passed 72 tests
locally. The candidate was built from Engine `f00f8f4` plus this one local
change, with Kernels `d3a9b8b`.

## Paired smoke result

The control and candidate used the exact same BASE F32 model, `mountains.jpg`,
504×336 preprocessing, all accepted production flags, 16 threads, one warm-up
and ten timed iterations. Ten interleaved pairs on the Ryzen 9 9950X yielded:

| Pair | 55-row control (ms) | 54-row candidate (ms) |
| --- | ---: | ---: |
| 1 | 161.986 | 159.554 |
| 2 | 159.834 | 163.697 |
| 3 | 167.179 | 162.560 |
| 4 | 159.201 | 160.480 |
| 5 | 156.586 | 158.356 |
| 6 | 163.071 | 160.101 |
| 7 | 163.486 | 156.051 |
| 8 | 162.085 | 163.869 |
| 9 | 160.244 | 159.388 |
| 10 | 162.171 | 160.328 |
| Mean | 161.584 | 160.438 |

The 54-row path won 6/10 pairs and improved the mean by 1.146 ms. The
predeclared admission rule required at least 3 ms reduction and eight wins in
ten pairs before a full parity and randomized study. It therefore does not
qualify.

## Decision

Reverted. The modest improvement is not sufficient to add a shape-specific
scheduler exception, and no public benchmark value changes. Raw results are
retained at `/var/tmp/vestra-slab54-smoke-20260911.log` on the Workhorse.
