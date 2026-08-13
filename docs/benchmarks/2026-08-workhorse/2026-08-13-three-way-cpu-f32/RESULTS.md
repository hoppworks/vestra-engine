# Scientific benchmark results

Generated: `2026-08-13T07:30:31.803642+00:00`

The primary estimator is the arithmetic mean of 10 independent trial medians;
the interval is a two-sided 95% Student-t confidence interval across trials.

| Runtime | Device | Precision | Mean median (ms) | 95% CI (ms) | SD (ms) | p95 pooled (ms) | RSS (MiB) |
|---|---|---|---:|---:|---:|---:|---:|
| rust | cpu | f32 | 171.087 | [164.408, 177.767] | 9.338 | 187.590 | 803.8 |
| cpp | cpu | f32 | 239.412 | [237.173, 241.652] | 3.130 | 252.267 | 618.8 |
| pytorch | cpu | f32 | 256.835 | [254.640, 259.030] | 3.069 | 270.987 | n/a |

## Interpretation rules

- Direct implementation comparisons use only F32 arms on the same device.
- Q8_0 and Q4_K are compression/accuracy trade-offs, not same-precision speed claims.
- CPU and GPU results are separate populations and must not be presented as language-only effects.
- Timing excludes model load and image decode; it includes preprocessing, backbone, depth/confidence head and host postprocessing.
- Raw commands, trial order, per-iteration samples, hashes and hardware metadata are preserved in `raw-results.json`.
