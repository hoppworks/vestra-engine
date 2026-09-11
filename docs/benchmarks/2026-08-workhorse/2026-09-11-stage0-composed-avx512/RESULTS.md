# Stage-0 composed transpose/lateral AVX-512 candidate

## Status

**Rejected after smoke admission.** The candidate preserved the locked C++ F32
parity envelope but did not meet the pre-registered performance admission rule.
It was not promoted to the default route and no full 10-trial qualification was
run.

## Hypothesis

DA3 BASE stage 0 applies a learned `96 -> 96`, 4×4/stride-4 transpose and
immediately applies a bias-free `96 -> 128`, 3×3/pad-1 lateral convolution.
Because there is no nonlinearity between them, composing them into a direct
`96 -> 128`, 6×6/stride-4 output-owned kernel should remove the 5.1 MiB
intermediate map and lower the pair's arithmetic work.

The implementation precomputes the composed filter plus nine edge/interior
bias classes. Its AVX-512 kernel accumulates sixteen output channels at once.
The route was opt-in only through `DA3_STAGE0_COMPOSED_TRANSPOSE_LATERAL=1`.

## Numerical gate

The exact DA3 BASE F32 GGUF, 504×336 input contract and four-image C++ PFM
corpus were used. Required: Pearson r >= 0.9999 and MAE <= 0.005 for every
image.

| Image | Pearson r | MAE | Result |
|---|---:|---:|---|
| canyon | 0.9999936282 | 0.001812496 | pass |
| desk | 0.9999782565 | 0.001772807 | pass |
| mountains | 0.9999855792 | 0.003674961 | pass |
| street | 0.9999721253 | 0.000820999 | pass |

Raw candidate PFMs and pose outputs are retained on the Workhorse at
`/var/tmp/vestra-compose-stage0-parity-20260911/`.

## Smoke timing

Hardware: AMD Ryzen 9 9950X, 16 fixed benchmark threads. The accepted Rust
configuration was held fixed; control and candidate were alternated by trial.
Each arm used one warm-up plus ten timed iterations of `mountains.jpg`.

| Trial | Control median (ms) | Candidate median (ms) | Candidate - control (ms) |
|---:|---:|---:|---:|
| 1 | 157.206 | 161.919 | +4.713 |
| 2 | 163.388 | 159.011 | -4.377 |
| 3 | 164.535 | 161.773 | -2.762 |
| 4 | 163.309 | 162.935 | -0.374 |
| 5 | 158.551 | 159.616 | +1.065 |
| mean | 161.398 | 161.051 | -0.347 |

The candidate won 3/5 pairs but reduced the mean by only 0.347 ms. Admission
required at least 3 ms and at least 8/10 wins before a full study. The result
does not qualify; no full benchmark or speedup claim is permitted.

Raw runner output is retained at
`/var/tmp/vestra-compose-stage0-smoke-20260911/runner.log`.

## Provenance

- Engine candidate commit: `b32143ada682e19134205902c9dd3db6a7fcb441`
- Kernel composition foundation: `1d7d0c2`
- AVX-512 kernel: `d5a761b`
- AVX-512 OC16 test: `77cc454`
- Warning cleanup: `bafee0c`

## Decision

Revert the Engine's opt-in dispatch and dependency pin. The measured result
does not justify retaining a selectable production path. The next attempt,
if any, must change the execution strategy (for example, a polyphase GEMM
schedule) rather than merely tuning this output-owned gather implementation.
