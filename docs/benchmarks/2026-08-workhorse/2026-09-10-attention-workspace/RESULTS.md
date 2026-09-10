# Iteration 56 — reusable attention activations (candidate)

## Hypothesis

DA3-BASE executes twelve CPU transformer blocks at the fixed 504x336 token
shape. Reusing the Q/K/V, attention-layout, RoPE-position, and fallback-QKV
buffers across those blocks should reduce allocator and zero-initialisation
traffic without changing arithmetic, kernels, precision, thread count, or the
timed work.

## Change

Commit `1abb23e` adds a typed `VitWorkspace` and uses one instance for the
single-view backbone loop. The multiview compatibility route retains a fresh
workspace per call. A unit test poisons retained buffers with NaNs and checks
bitwise equality against the fresh route after reuse.

## Verification

- `cargo test --locked -p vestra-engine --lib`: 69 tests passed.
- Four-image C++ F32 parity against the retained current-public reference:
  all images pass the locked gate (Pearson r >= 0.9999, MAE <= 0.005).
  Measured `(r, MAE)`: canyon `(0.9999936281, 0.0018124714)`, desk
  `(0.9999782659, 0.0017725341)`, mountains `(0.9999855791,
  0.0036750568)`, street `(0.9999721281, 0.0008208277)`.

## Smoke measurement — not a publishable result

On the Ryzen 9 9950X Workhorse, 16 threads, F32 model, 504x336 input, one
warm-up, three randomized process trials and five samples per trial:

| Arm | Mean of trial medians |
|---|---:|
| Rust before workspace | 230.528 ms |
| Rust after workspace | 224.756 ms |

The observed direction is favorable, but this smoke has only three trials and
its confidence interval is wide. It is not a 50% claim and does not qualify
for the final comparison. The candidate remains enabled because it preserves
parity and is a low-risk reduction in repeated allocation; future serious
candidates must be measured with the locked 20-trial protocol.

## Next hypothesis

The warm profile attributes the stable work primarily to the transformer MLP
and DPT head, not allocation. The next experiment should compare physical-core
and BLIS/Faer scheduling only when applied identically within the Rust arm and
then retain the fair C++ comparison unchanged. It must not alter the fixed
16-thread benchmark contract.
