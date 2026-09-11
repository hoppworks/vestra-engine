# OpenMP one-millisecond blocktime — rejected

## Hypothesis

The accepted `KMP_BLOCKTIME=0` policy prevents BLIS/OpenMP workers from
competing with Rayon after a GEMM. A one-millisecond active interval might
instead avoid thread wake-up latency before the next BLIS projection while
remaining short relative to inference.

## Method

The accepted Rust binary was measured with the locked BASE F32 GGUF,
`mountains.jpg`, 504×336 preprocessing, 16 threads, one warm-up and ten timed
iterations. All accepted flags were identical; only `KMP_BLOCKTIME=0` versus
`KMP_BLOCKTIME=1` differed. Ten interleaved pairs ran on the Ryzen 9 9950X.

| Pair | 0 ms control | 1 ms candidate |
| --- | ---: | ---: |
| 1 | 156.001 | 158.575 |
| 2 | 160.275 | 162.697 |
| 3 | 163.772 | 157.262 |
| 4 | 164.712 | 155.282 |
| 5 | 163.627 | 161.366 |
| 6 | 161.330 | 160.910 |
| 7 | 156.414 | 165.025 |
| 8 | 164.214 | 159.814 |
| 9 | 168.649 | 160.002 |
| 10 | 162.912 | 161.141 |
| Mean | 162.191 | 160.207 |

The candidate wins 7/10 pairs and reduces the mean by 1.983 ms. It fails the
predeclared admission requirement of at least 3 ms and eight wins in ten
pairs, so a parity/full-study promotion would be overfitting noise.

## Decision

Keep `KMP_BLOCKTIME=0`. The one-millisecond setting is retained only in this
record, not in the qualified benchmark environment.
