# `target-cpu=native` code generation — rejected

## Hypothesis

LLVM's host-native CPU selection on the Workhorse might expose a useful code
generation distinction from the explicit `target-cpu=znver5` setting while
leaving the model, arithmetic and benchmark boundary unchanged.

## Method

Two Rust 1.93.0 release binaries from the same Engine/Kernels source were
compared: the accepted explicit `znver5` control and a `target-cpu=native`
candidate. Both used all accepted runtime settings, the locked BASE F32 GGUF,
`mountains.jpg`, 504×336 preprocessing, 16 threads, one warm-up and ten timed
iterations. Five interleaved pairs ran on the Ryzen 9 9950X.

| Pair | `znver5` control (ms) | `native` candidate (ms) |
| --- | ---: | ---: |
| 1 | 161.449 | 157.622 |
| 2 | 158.535 | 157.925 |
| 3 | 158.051 | 158.015 |
| 4 | 163.269 | 166.768 |
| 5 | 152.541 | 152.994 |
| Mean | 158.769 | 158.665 |

The candidate improves the mean by only 0.104 ms and wins three pairs. This
is indistinguishable from run noise and does not meet admission.

## Decision

Keep explicit `target-cpu=znver5`, which is reproducible and portable across
the intended Ryzen target class. No parity or full study is warranted.
