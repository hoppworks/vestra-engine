# Iteration 46 — Flash 8×32 candidate

The candidate replaces the per-query-tile QK/PV 4×64 products with an
8-query × 32-output AVX-512 microkernel. It keeps the K-major FMA chain and
online-softmax order bit-identical while reducing live B-panel vectors.

| Arm | Mean of 10 trial medians | 95% CI |
| --- | ---: | --- |
| Rust F32, opt-in 8×32 | 170.903 ms | [165.207, 176.599] ms |
| C++/ggml F32 | 239.364 ms | [238.301, 240.427] ms |

This particular run is 40.06% faster at the point estimate, but the later
standard-path confirmation remains the conservative benchmark record because
host variance is materially wider on Rust.

Raw data: `/tmp/da3-cpu-f32-flash8x32-layerenv-20260817/raw-results.json`
on the Workhorse.
