# Iteration 65 — commuted DPT fusion resize and 1x1 projection

## Hypothesis

Each DPT feature-fusion branch evaluates a bilinear align-corners resize,
followed by a spatially shared 128-to-128 1x1 affine projection. In real
arithmetic these operations commute:

```text
resize(W * x + b) = W * resize(x) + b
```

Executing the projection on the source map should reduce its GEMM work before
the 2x upsample. The candidate is opt-in through
`DA3_COMMUTE_FUSION_RESIZE_1X1=1`; its normal route remains unchanged.
No model-loading, image-decoding, or input-dependent work moved outside the
locked timed boundary.

## Change and regression test

`crates/da-engine/src/dpt_head.rs` computes the source-resolution 1x1 output,
then resizes that result only when the opt-in is set. The existing route still
resizes first and projects at target resolution.

`resize_and_1x1_projection_commute_within_f32_envelope` exercises a
two-channel non-square, align-corners resize with non-diagonal OIHW weights
and bias. It verifies the two orders within `1e-6` F32 error.

`cargo test --locked -p vestra-engine --lib` passed: 71 tests.

## Four-image F32 parity

| Image | Pearson r | MAE |
|---|---:|---:|
| canyon | 0.9999936280 | 0.0018125200 |
| desk | 0.9999782569 | 0.0017729214 |
| mountains | 0.9999855776 | 0.0036751982 |
| street | 0.9999721240 | 0.0008210028 |

Every result meets the locked `r >= 0.9999`, `MAE <= 0.005` gate.

## Smoke evidence — not accepted yet

The Workhorse was not idle. All commands used the identical F32 model, image,
resolution, 16-thread budget, and timed boundary, but are explicitly only
short candidate checks, not the final randomized study.

| Order | Standard route median | Commuted candidate median |
|---|---:|---:|
| 1 warmup + 5 timed iterations | 226.998 ms | 208.347 ms |
| Alternating pair, 1 warmup + 3 timed iterations | 207.451 ms | 216.989 ms |
| Alternating pair, 1 warmup + 3 timed iterations | 234.551 ms | 230.168 ms |

Head profiles did show a directional fusion-only reduction in one adjacent
pair (31.628 ms to 22.126 ms), but the whole-image values are too variable to
promote this candidate or claim an improvement. It stays opt-in pending an
idle-workhorse study with the locked 20-trial protocol.
