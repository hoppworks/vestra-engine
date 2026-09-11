# Full-matrix BLIS MLP path — rejected

## Hypothesis

With `DA3_KERNELS_BLIS_LINEAR=1`, the normal non-packed transformer route
executes complete FC1 and FC2 DA3 shapes through BLIS. The accepted
hidden-strip MLP uses panel-local AVX-512 primitives instead. Comparing these
two routes directly establishes whether a whole-matrix BLIS redesign could
provide a larger remaining MLP gain.

## Method

Both arms used the accepted binary, BASE F32 GGUF, `mountains.jpg`, 504×336
preprocessing, 16 Rayon/OpenMP threads, one warm-up and ten timed iterations.
The strip arm enabled `DA3_STRIP_MLP=1` and the qualified hoisted dispatch;
the BLIS arm enabled neither packed MLP mode, so only the normal BLIS linear
route differed. Five interleaved pairs ran on the Ryzen 9 9950X.

| Pair | Hidden strip (ms) | Full BLIS MLP (ms) |
| --- | ---: | ---: |
| 1 | 162.765 | 170.171 |
| 2 | 158.585 | 170.477 |
| 3 | 160.221 | 178.499 |
| 4 | 159.949 | 172.419 |
| 5 | 158.218 | 175.567 |
| Mean | 159.948 | 173.427 |

The full BLIS route loses every pair and is 13.479 ms slower.

## Decision

Keep hidden-strip MLP execution. A whole-matrix BLIS MLP is not a viable path
to the remaining target; it receives no parity or full-study promotion.
