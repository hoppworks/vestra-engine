# Hidden-strip packed-panel dispatch candidate — 2026-09-10

## Hypothesis

The 55-row hidden-strip MLP executor invokes validated packed-panel helpers
inside its tight per-strip loops. The helpers still need no dynamic policy
checks after the executor has already established the compatible shape and
backend. `DA3_STRIP_MLP_HOIST_DISPATCH=1` snapshots that selection once and
uses the serial-panel route directly. It changes no arithmetic or tensor
layout.

## Idle-host admission smoke

The test used `vestra-engine` `f3a3ff7`, `vestra-kernels` `d3a9b8b`, the
locked DA3-BASE F32 model and `desk.jpg` at 504 x 336 on the Ryzen 9 9950X.
The RN1 OC32 candidate was enabled in both arms; all other candidate flags and
the 16-thread configuration were held fixed. Five interleaved trials used one
warm-up plus ten timed iterations and varied only
`DA3_STRIP_MLP_HOIST_DISPATCH`.

| Pair | Baseline median (ms) | Hoisted median (ms) | Delta (ms) |
|---:|---:|---:|---:|
| 1 | 163.033 | 156.844 | -6.189 |
| 2 | 162.258 | 161.750 | -0.508 |
| 3 | 158.346 | 157.262 | -1.084 |
| 4 | 162.911 | 156.480 | -6.431 |
| 5 | 154.208 | 160.996 | +6.788 |
| **Mean** | **160.151** | **158.666** | **-1.485** |

The candidate is faster in four of five pairs and clears its predeclared
one-millisecond smoke admission. It is retained for the next composition
step; it has not yet earned a final randomized claim.

## C++ F32 parity on the target

The composed RN1-plus-dispatch path yielded the same four-image PFM comparison
as RN1 alone, because the dispatch route preserves all arithmetic:

| Image | Pearson r | MAE |
|---|---:|---:|
| canyon | 0.9999936282 | 0.0018124970 |
| desk | 0.9999782565 | 0.0017728067 |
| mountains | 0.9999855792 | 0.0036749614 |
| street | 0.9999721254 | 0.0008209983 |

Every image passes `r >= 0.9999` and `MAE <= 0.005`.
