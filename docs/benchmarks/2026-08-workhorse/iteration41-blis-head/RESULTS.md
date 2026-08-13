# Scientific benchmark results

Generated: `2026-08-13T05:08:00.398889+00:00`

The primary estimator is the arithmetic mean of 10 independent trial medians;
the interval is a two-sided 95% Student-t confidence interval across trials.

| Runtime | Device | Precision | Mean median (ms) | 95% CI (ms) | SD (ms) | p95 pooled (ms) | RSS (MiB) |
|---|---|---|---:|---:|---:|---:|---:|
| rust | cpu | f32 | 181.138 | [175.623, 186.653] | 7.711 | 197.384 | 804.1 |
| cpp | cpu | f32 | 238.513 | [236.701, 240.325] | 2.533 | 251.468 | 618.2 |

## Interpretation rules

- Direct implementation comparisons use only F32 arms on the same device.
- Q8_0 and Q4_K are compression/accuracy trade-offs, not same-precision speed claims.
- CPU and GPU results are separate populations and must not be presented as language-only effects.
- Timing excludes model load and image decode; it includes preprocessing, backbone, depth/confidence head and host postprocessing.
- Raw commands, trial order, per-iteration samples, hashes and hardware metadata are preserved in `raw-results.json`.
