# OpenMP passive wait policy — rejected

## Hypothesis

The accepted runtime already sets `KMP_BLOCKTIME=0` to release BLIS/OpenMP
workers between GEMMs and Rayon phases. Adding `OMP_WAIT_POLICY=PASSIVE` could
further reduce contention without changing the model or arithmetic.

## Method

The accepted Rust binary used the locked BASE F32 GGUF, `mountains.jpg`,
504×336 preprocessing, 16 threads, one warm-up and ten timed iterations.
All accepted environment variables were identical; only
`OMP_WAIT_POLICY=PASSIVE` differed. Five interleaved pairs ran on the Ryzen 9
9950X.

| Pair | Current policy (ms) | Passive candidate (ms) |
| --- | ---: | ---: |
| 1 | 159.345 | 158.046 |
| 2 | 151.026 | 156.893 |
| 3 | 154.217 | 161.365 |
| 4 | 157.481 | 159.143 |
| 5 | 153.039 | 164.657 |
| Mean | 155.022 | 160.021 |

The passive candidate won one of five pairs and is 4.999 ms slower on average.

## Decision

Reject `OMP_WAIT_POLICY=PASSIVE`. Retain `KMP_BLOCKTIME=0` with the runtime's
default wait policy. No C++ or parity run is warranted because the Rust-only
admission smoke is decisively negative.
