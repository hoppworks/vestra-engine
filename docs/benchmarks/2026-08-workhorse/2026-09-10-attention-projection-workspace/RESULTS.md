# Iteration 60 — reusable attention projection activation

## Hypothesis and change

After the MLP workspace change, the attention output projection still created a
fresh `[865, 768]` vector in every transformer block. Its lifetime ends at the
attention residual addition and FC2 needs the identical shape immediately
afterwards. Commit `353dd13` therefore takes the shared branch buffer for the
projection, returns it after the residual addition, and reuses it for FC2.

Only output-buffer ownership changes. GEMM dispatch, biases, LayerScale,
arithmetic order, precision, inputs, and the benchmark boundary are unchanged.

## Verification

- `cargo test --locked -p vestra-engine --lib`: 70 tests passed.
- The existing fresh/reused/NaN-poisoned transformer-block test covers the
  projection buffer and asserts bitwise-equal output. It also verifies stable
  fixed-geometry activation capacities.

## Measurement status

No latency number is attributed to this change yet: the Workhorse remains
under unrelated load, and allocation reuse is expected to be materially
smaller than the remaining FC1/FC2 arithmetic cost. It must be included in the
next idle-machine alternating smoke and full randomized study, together with
the four-image C++ F32 parity gate.
