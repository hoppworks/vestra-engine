# Fat LTO release profile — rejected

## Hypothesis

Changing Cargo's release link-time optimization from `thin` to `fat` might
improve whole-program inlining and vectorization across Engine, Graph and
Kernels while preserving the identical CPU F32 workload.

## Method

The candidate was an unmodified Vestra Engine `097a317` built with
`CARGO_PROFILE_RELEASE_LTO=fat`; the control was the same revision built with
the repository's `lto = "thin"` release profile. Both binaries used the
locked BASE F32 GGUF, `mountains.jpg`, 504×336 preprocessing, all accepted
runtime flags, 16 threads, one warm-up and ten timed iterations. Five pairs
were interleaved on the Ryzen 9 9950X.

| Pair | Thin LTO control (ms) | Fat LTO candidate (ms) |
| --- | ---: | ---: |
| 1 | 166.674 | 163.664 |
| 2 | 161.183 | 165.991 |
| 3 | 158.408 | 161.521 |
| 4 | 160.754 | 162.048 |
| 5 | 164.371 | 158.462 |
| Mean | 162.278 | 162.337 |

Fat LTO won two of five pairs and is 0.059 ms slower on average. It does not
qualify for parity or a full randomized study.

## Decision

Keep Thin LTO with one codegen unit. Fat LTO increases build cost without a
reproducible inference benefit for the locked workload.
