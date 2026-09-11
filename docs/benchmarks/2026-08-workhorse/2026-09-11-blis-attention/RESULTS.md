# BLIS materialized-attention candidate

## Status

**Rejected after parity and smoke admission.** The candidate holds F32 parity
but is materially slower than the accepted streaming Flash-attention route.
The Engine pin and kernel implementation were reverted after recording this
result.

## Hypothesis

For each DA3 64-wide attention head, compute `Q × K^T` and `softmax(scores) ×
V` as two large BLIS SGEMMs, with a parallel AVX-512 softmax between them.
This trades Flash's tile scheduling for BLIS's optimized large-matrix backend.

## Four-image C++ F32 gate

| Image | Pearson r | MAE | Result |
|---|---:|---:|---|
| canyon | 0.9999936281 | 0.001812528 | pass |
| desk | 0.9999782655 | 0.001772400 | pass |
| mountains | 0.9999855787 | 0.003675074 | pass |
| street | 0.9999721275 | 0.000820868 | pass |

The candidate's global-softmax reassociation remained within the locked F32
envelope. Raw outputs are retained at
`/var/tmp/vestra-blis-flash-parity-20260911/`.

## Smoke timing

AMD Ryzen 9 9950X, fixed 16-thread contract, DA3 BASE F32,
`mountains.jpg`, one warm-up plus ten timed iterations per arm. Control and
candidate alternated each trial.

| Trial | Flash control median (ms) | BLIS attention median (ms) | Candidate - control (ms) |
|---:|---:|---:|---:|
| 1 | 159.405 | 197.468 | +38.063 |
| 2 | 165.992 | 198.832 | +32.840 |
| 3 | 162.709 | 197.110 | +34.401 |
| 4 | 168.380 | 198.870 | +30.490 |
| 5 | 158.729 | 199.358 | +40.629 |
| mean | 163.043 | 198.328 | +35.285 |

The candidate lost 5/5 pairs. The score matrix (about 2.9 MiB per head),
serial head processing, and repeated large SGEMM setup outweigh any benefit
from BLIS. It fails the pre-registered admission criterion decisively; no
full qualification study is appropriate.

Raw runner output: `/var/tmp/vestra-blis-flash-smoke-20260911/runner.log`.

## Provenance

- Kernel candidate: `16350d1bdb445f82bffd011e8f1394d715d23c9e`
- Engine pin: `21863eb347524db36072b6045e2431f3edc19e50`
