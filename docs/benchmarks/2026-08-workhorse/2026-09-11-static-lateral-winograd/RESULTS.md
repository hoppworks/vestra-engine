# Iteration 99 — static-width lateral Winograd kernels (reverted from Engine)

## Hypothesis and isolated change

The four DPT lateral 3x3 convolutions use exactly four F(2) tiles per kernel
call and fixed channel shapes `96/192/384/768 -> 128`.  The generic AVX-512
product accepts dynamic channel counts and tile count.  `vestra-kernels`
commit `5ba1d0e` added opt-in static-width variants, retaining the same
position -> output-panel -> input FMA order while removing dynamic bounds from
the hot loops.  The Engine candidate only updated the kernel dependency and
enabled `DA3_KERNELS_LATERAL_F2_STATIC=1` for the candidate arm.

## Correctness

- The Engine library suite passed: 72 tests.
- C++ F32 parity passed for all four locked images:

| Image | Pearson r | MAE |
|---|---:|---:|
| canyon | 0.9999936282 | 0.0018124970 |
| desk | 0.9999782565 | 0.0017728067 |
| mountains | 0.9999855792 | 0.0036749614 |
| street | 0.9999721254 | 0.0008209983 |

## Locked smoke measurement

Both arms used the Ryzen 9 9950X Workhorse, Rust 1.93.0, the same DA3 BASE
F32 GGUF, `mountains.jpg`, 504x336 preprocessing, 16 fixed threads, accepted
runtime flags, and one warm-up plus ten timed iterations per pair.

| Pair | Control median (ms) | Static lateral median (ms) |
|---:|---:|---:|
| 1 | 153.188 | 158.213 |
| 2 | 158.976 | 160.417 |
| 3 | 156.618 | 162.542 |
| 4 | 158.933 | 157.921 |
| 5 | 157.437 | 159.928 |

The candidate is slower in four of five pairs: 159.804 ms versus 157.030 ms
for control (+2.774 ms, +1.77%).  Constant-specializing the loops increased
instruction-cache pressure rather than improving the already-vectorized
generic kernel.

## Decision

Rejected.  The Engine dependency was restored to the accepted kernel revision.
The separate kernel commit remains available with its regression test, but it
is not part of Vestra Engine's benchmark configuration.
