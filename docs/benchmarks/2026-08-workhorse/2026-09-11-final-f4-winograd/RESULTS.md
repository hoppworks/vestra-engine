# Final F(4) Winograd output path — rejected

## Hypothesis

The final DA3 BASE head operation combines an align-corners resize with a
64→32 3×3 convolution. Its steady-state profile contribution is about 8 ms.
The existing exact-shape `DA3_WINO_F4_FINAL=1` implementation reduces the
Winograd-domain product count, so it might outweigh the extra materialized
resize buffer.

## Method

The tested binary was Vestra Engine `800a1a4`, Vestra Kernels `d3a9b8b`, and
the locked BASE F32 GGUF. Both arms used the identical `mountains.jpg` input,
504×336 preprocessing, 16 threads, one warm-up and ten timed inferences per
run. The only differing runtime flag was `DA3_WINO_F4_FINAL=1`. Six runs were
interleaved in randomized ABBA-style order on the Ryzen 9 9950X.

## Results

| Run | F(2) fused production path (ms) | F(4) materialized candidate (ms) |
| --- | ---: | ---: |
| 1 | 160.884 | 166.211 |
| 2 | 165.197 | 166.619 |
| 3 | 157.924 | 164.524 |
| Mean | 161.335 | 165.785 |

The candidate is 4.450 ms slower on average and loses every matched pair.
It does not qualify for the four-image PFM gate or full randomized study.

## Decision

Keep the default fused resize plus F(2) Winograd path. The existing F(4)
candidate remains opt-in only for reproducibility; it is not part of the
qualified product configuration.
