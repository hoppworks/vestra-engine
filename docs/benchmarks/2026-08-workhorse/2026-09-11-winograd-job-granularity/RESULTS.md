# Iteration 101 — Winograd job granularity (rejected)

## Hypothesis

The default F(2) convolution scheduler gives one Rayon work item at least
four tile blocks.  Reducing that lower bound to one could improve load balance
between the differently sized DPT lateral/refinement convolutions without
changing operations, tensors, or F32 arithmetic.

## Screening and controlled comparison

An initial same-binary screen exercised the available values `1`, `2`, `4`,
and `8`. Only `1` showed a possible signal, so the admission test alternated
the default and `DA3_WINOGRAD_MIN_BLOCKS_PER_JOB=1` under the locked Rust
configuration: Ryzen 9 9950X, 16 threads, DA3 BASE F32, `mountains.jpg`,
504x336, one warm-up and ten timed iterations per arm.

| Pair | Default median (ms) | One-block median (ms) |
|---:|---:|---:|
| 1 | 157.166 | 160.312 |
| 2 | 166.350 | 165.078 |
| 3 | 158.981 | 156.319 |
| 4 | 161.205 | 158.352 |
| 5 | 160.802 | 159.225 |

The one-block route wins four pairs, but the means are 160.901 ms and
159.857 ms: only 1.044 ms (0.65%).  It misses the pre-registered 2 ms
admission magnitude, so this screen is insufficient to alter the accepted
configuration or justify a larger qualification study.

## Decision

Rejected. Keep the four-block minimum as the stable production schedule.
