# Iteration 97 — four-panel FC2 supertile (reverted from Engine)

## Hypothesis and isolated change

The accepted MLP route writes the FC2 partial output once for each 64-hidden
channel strip.  This candidate added an AVX-512 kernel in `vestra-kernels`
commit `0bf9578` that consumes four consecutive strips while retaining the
same 6x64 output tile in registers.  It preserves the FMA order for every
output lane (strip 0→3, then channel 0→63) while reducing FC2 partial-output
load/store rounds from 48 to 12 per slab.

The Engine candidate, gated by `DA3_STRIP_MLP_GROUP_FC2=1`, materialized four
small hidden strips per slab, then called that kernel.  The ordinary
single-strip route stayed intact for direct alternating control measurement.

## Correctness

- `vestra-kernels` tests passed: 46 unit, 7 integration tests.  A new oracle
  confirms bitwise equality of the grouped primitive against four existing
  FC2 panel calls for signed inputs and nonzero accumulated output.
- Engine library tests passed: 72 tests.
- The remote four-image C++ F32 gate passed with the established values:

| Image | Pearson r | MAE |
|---|---:|---:|
| canyon | 0.9999936282 | 0.0018124970 |
| desk | 0.9999782565 | 0.0017728067 |
| mountains | 0.9999855792 | 0.0036749614 |
| street | 0.9999721254 | 0.0008209983 |

Each remains inside `r >= 0.9999` and `MAE <= 0.005`.

## Locked smoke measurement

On the idle Ryzen 9 9950X Workhorse, both arms used Rust 1.93.0, the same
DA3 BASE F32 GGUF, `mountains.jpg`, 504x336 preprocessing, 16 Rayon/OpenMP
threads, and the accepted BLIS/Head flags.  Each invocation used one warm-up
and ten timed iterations; the five control/candidate pairs alternated.

| Pair | Accepted MLP median (ms) | Four-panel median (ms) |
|---:|---:|---:|
| 1 | 163.504 | 153.459 |
| 2 | 163.845 | 164.010 |
| 3 | 159.519 | 160.311 |
| 4 | 164.942 | 162.754 |
| 5 | 157.459 | 161.928 |

The arithmetic mean is 161.854 ms for control and 160.492 ms for the
candidate: only 1.362 ms (0.84%) lower, and the candidate wins just two of
five alternating pairs.  This misses the predeclared admission threshold of
at least 2 ms with at least 8/10 wins, so it does not justify a full
qualification study.

## Decision

The Engine integration and dependency update were reverted.  The separately
versioned kernel remains available with its bitwise regression coverage, but
is not imported by the accepted Engine configuration.  The result shows that
FC2 partial-output traffic is not the dominant remaining MLP cost; the next
candidate must attack larger backbone or head regions rather than widening
this isolated FC2 residency window.
