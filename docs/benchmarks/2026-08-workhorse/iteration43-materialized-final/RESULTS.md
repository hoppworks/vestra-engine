# Iteration 43 — materialised final resize control

The candidate disables the production fused final resize + F(2) Winograd
route with `DA3_DISABLE_FUSE_FINAL_RESIZE_WINO=1`.  It instead materialises
the `64×504×336` resize before the same prepared F(2) convolution.

| Arm | Mean of 10 trial medians | 95% CI |
| --- | ---: | --- |
| Rust F32, candidate | 185.855 ms | [181.940, 189.771] ms |
| C++/ggml F32 | 238.912 ms | [237.365, 240.458] ms |

The candidate remains faster than C++, but regresses materially from the
accepted BLIS-head route (181.138 ms).  The confidence intervals overlap and
the point estimate is 4.718 ms slower, so this is rejected despite the
favourable five-pair smoke result.

All four C++ F32 parity comparisons passed before the full study: Pearson
`r >= 0.9999721`, MAE `<= 0.0036752`.

Raw data remains at the Workhorse study path
`/tmp/da3-cpu-f32-materialized-final-20260815/raw-results.json`.
