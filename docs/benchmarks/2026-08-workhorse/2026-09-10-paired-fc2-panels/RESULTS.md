# Paired hidden-strip FC2 panels — 2026-09-10

## Hypothesis

The hidden-strip MLP reduces FC2 in 48 successive 64-wide hidden panels. The
established route loads and stores the same output panel after every input
panel. This candidate retained the accumulator across two consecutive panels,
while preserving the strictly ascending 128-FMA order for every output lane.

## Change and correctness evidence

`vestra-kernels` `ff189e8` added a two-panel AVX-512 primitive and a
bit-for-bit oracle against the two established one-panel calls. `vestra-engine`
`73d218c` exposed it only through `DA3_STRIP_MLP_PAIR_FC2=1` and only after
the existing validated-panel dispatch had been selected.

The local kernel suite passed 5/5 focused packed-GEMM tests and the Engine
suite passed 72/72. A clean Workhorse clone executed the AVX-512/FMA body of
the packed-GEMM suite: 5/5 passed. The four-image C++ F32 PFM gate also passed:

| Image | Pearson r | MAE |
|---|---:|---:|
| canyon | 0.9999936282 | 0.0018124970 |
| desk | 0.9999782565 | 0.0017728067 |
| mountains | 0.9999855792 | 0.0036749614 |
| street | 0.9999721254 | 0.0008209983 |

## Idle-host A/B result

Hardware was the Ryzen 9 9950X with the locked DA3-BASE F32 workload,
`desk.jpg` at 504 x 336, 16 Rayon/OpenMP threads, one warm-up and ten timed
iterations per trial. RN1 OC32 and hidden-strip dispatch were fixed in both
arms. Ten interleaved same-binary pairs varied only
`DA3_STRIP_MLP_PAIR_FC2`.

| Pair | Control median (ms) | Paired median (ms) | Delta (ms) |
|---:|---:|---:|---:|
| 1 | 161.280 | 162.608 | +1.328 |
| 2 | 166.876 | 151.528 | -15.348 |
| 3 | 162.618 | 158.986 | -3.632 |
| 4 | 160.859 | 161.688 | +0.829 |
| 5 | 164.925 | 166.121 | +1.196 |
| 6 | 156.034 | 155.721 | -0.313 |
| 7 | 163.007 | 164.364 | +1.357 |
| 8 | 162.511 | 157.095 | -5.416 |
| 9 | 155.721 | 155.493 | -0.228 |
| 10 | 146.507 | 158.611 | +12.104 |
| **Mean** | **160.034** | **159.222** | **-0.812** |

The candidate is faster in only 5/10 pairs and misses the predeclared
2-ms/8-of-10 admission gate. **Decision: reject and revert this change.**
No public performance claim includes this route.
