# Iteration 47 — Flash 8×32 standard path

The verified 8×32 Flash microkernel is enabled by default for DA3-BASE QT8
tiles; `DA3_KERNELS_DISABLE_FLASH_GEMM_8X32=1` retains the historical 4×64
route for diagnosis.

| Arm | Mean of 10 trial medians | 95% CI |
| --- | ---: | --- |
| Rust F32, standard path | 175.116 ms | [171.285, 178.946] ms |
| C++/ggml F32 | 237.590 ms | [235.915, 239.264] ms |

The point estimate is 35.68% faster than C++. Four-image C++ F32 parity
passes: r >= 0.9999721 and MAE <= 0.0036752.

Raw data: `/tmp/da3-cpu-f32-final-20260818/raw-results.json` on the
Workhorse.
