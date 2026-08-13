# Scientific benchmark results

Generated: `2026-08-13T06:41:57.643876+00:00`

The primary estimator is the arithmetic mean of 10 independent trial medians;
the interval is a two-sided 95% Student-t confidence interval across trials.

| Runtime | Device | Precision | Mean median (ms) | 95% CI (ms) | SD (ms) | p95 pooled (ms) | RSS (MiB) |
|---|---|---|---:|---:|---:|---:|---:|
| rust | cpu | f32 | 171.542 | [166.078, 177.006] | 7.639 | 186.065 | 804.0 |
| cpp | cpu | f32 | 238.136 | [236.287, 239.986] | 2.586 | 248.357 | 618.3 |

## Interpretation rules

- Direct implementation comparisons use only F32 arms on the same device.
- Q8_0 and Q4_K are compression/accuracy trade-offs, not same-precision speed claims.
- CPU and GPU results are separate populations and must not be presented as language-only effects.
- Timing excludes model load and image decode; it includes preprocessing, backbone, depth/confidence head and host postprocessing.
- Raw commands, trial order, per-iteration samples, hashes and hardware metadata are preserved in `raw-results.json`.
