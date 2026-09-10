# Iteration 58 — landscape final resize + Winograd dispatch

## Hypothesis and isolated change

The existing exact fused final resize + F(2) Winograd route was admitted only
for the portrait tuple `(64, 288, 192, 504, 336)`. The locked benchmark uses a
landscape image: preprocessing produces `(height, width) = (336, 504)` and the
final head map is `(64, 192, 288)`. The fused kernel itself accepts generic
dimensions, so the landscape path was unnecessarily taking the materialized
resize route.

Commit `59d2f94` changes only the dispatch predicate. It now accepts the two
exact DA3-BASE orientations and adds a unit test that rejects transposed or
otherwise altered shapes. Model arithmetic, model weights, image policy,
thread budget, and the timed boundary are unchanged.

## Correctness

- `cargo test --locked -p vestra-engine --lib`: 70 tests passed.
- Four-image Rust versus current-upstream C++ F32 parity on the Workhorse:

| Image | Pearson r | MAE |
|---|---:|---:|
| canyon | 0.9999936280 | 0.0018125201 |
| desk | 0.9999782568 | 0.0017729246 |
| mountains | 0.9999855775 | 0.0036752027 |
| street | 0.9999721240 | 0.0008210033 |

Every result meets the locked gate: Pearson r >= 0.9999 and MAE <= 0.005.

## Smoke evidence — not a final benchmark

The Workhorse was concurrently occupied by unrelated browser, video, and
frontend jobs, so no latency value from this iteration is publishable. An
alternating same-process head-profile smoke nevertheless compared the two
routes at the same landscape shape:

| Route | Final output section (two samples) |
|---|---:|
| fused resize + F(2) | 114.037 ms, 115.844 ms |
| materialized resize + F(2) | 117.732 ms, 118.718 ms |

The fused route was directionally faster in this noisy diagnostic. The timer
includes both resize and `out2a` convolution, by design; it must not be read
as resize-only time. The route remains enabled, but no total-inference gain is
claimed until the dedicated 20-trial randomized study can run on an idle
Workhorse.

## Next hypothesis

The remaining high-value DPT opportunity is a channel-blocked feature-map
layout that permits AVX-512 input/output transforms, ReLU, residual addition,
and resize to stay vectorized across a complete residual unit. This is a
separate architectural experiment, not a follow-up micro-tune of this
dispatch correction.
