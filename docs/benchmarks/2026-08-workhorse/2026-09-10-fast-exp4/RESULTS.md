# Reduced flash exponential — rejected

## Hypothesis

Flash attention evaluates a 64-wide Cephes-style exponential tile. On its
range-reduced interval, the two smallest polynomial terms are tiny. The
opt-in `DA3_KERNELS_FLASH_FAST_EXP4=1` route omitted those two vector FMAs
while retaining the same range reduction and all attention scheduling.

## Correctness

The AVX-512 oracle compared the full and reduced polynomials over a 64-value
score range and bounded relative error below `3e-5`. The four-image C++ F32
parity gate passed:

| Image | Pearson r | MAE |
|---|---:|---:|
| canyon | 0.9999936280 | 0.0018125188 |
| desk | 0.9999782584 | 0.0017727750 |
| mountains | 0.9999855774 | 0.0036752486 |
| street | 0.9999721246 | 0.0008210182 |

## Smoke admission measurement

AMD Ryzen 9 9950X, same binary/model/`desk` input/504x336/16 threads in both
arms. Each arm used one warm-up plus ten timed iterations. The common accepted
candidate flags were unchanged; only `DA3_KERNELS_FLASH_FAST_EXP4` differed.

| Pair | Control median ms | Reduced-exp median ms |
|---:|---:|---:|
| 1 | 153.390 | 154.708 |
| 2 | 160.553 | 159.775 |
| 3 | 161.048 | 158.560 |
| 4 | 163.305 | 162.768 |
| 5 | 160.282 | 163.253 |
| **Mean** | **159.716** | **159.813** |

The candidate was 0.097 ms slower and won only 2/5 pairs. It did not earn a
full ten-pair or randomized study, and is reverted. The full polynomial remains
the production route.

