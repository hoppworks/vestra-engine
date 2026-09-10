# Packed direct QKV candidate — 2026-09-10

## Hypothesis

The established direct QKV kernel already has the desired execution shape: 36
Rayon output-panel jobs, six-token microtiles, K-ascending AVX-512 FMA,
separate bias addition, and direct HND (`[head, token, dimension]`) output.
It nevertheless loads each 64-column weight panel through a 9,216-byte stride
between consecutive K values (`weight[k * 2304 + column]`).

For each panel that spans roughly 6.75 MiB and about 768 4-KiB pages. Packing
immutable `768 × 2304` QKV weights at model load as `[panel][K][64]` makes the
same 192-KiB panel contiguous (about 48 pages). No arithmetic, activation
layout, scheduler, bias position, or attention operation changes.

The retained packed QKV copies add approximately 81 MiB for the twelve DA3-BASE
blocks. Model preparation is explicitly outside the locked inference boundary;
the added RSS must be recorded during qualification.

## Change

- `vestra-kernels` `865a9d4`: `PreparedLinearF32::run_qkv_da3_base`, which
  retains the production direct-QKV six-row/64-column schedule and HND stores,
  consuming only model-owned packed weights.
- `vestra-engine`: opt-in `PackedQkvExecutor`, constructed once at model load
  only when `DA3_PACKED_QKV=1`. It is passed into the backbone without changing
  the default direct-QKV route or the existing MLP candidate selection.

## Local verification

`cargo test -p vestra-kernels --lib`: 46/46 passed.

The dedicated regression test initializes Q/K/V with NaNs, compares every HND
element bit-for-bit against the established column-split direct QKV route for
all 865 tokens and 12 heads, then repeats the comparison with a distinct input
on the same packed weights to detect stale output state.

`cargo test -p vestra-engine --lib`: 72/72 passed.

The local development host cannot execute the AVX-512 test body; the target
Workhorse must execute it before this candidate can be promoted.

## Predeclared admission

The Workhorse was busy with unrelated Chrome, Flutter, VM and video workloads
when this candidate was prepared. No timing was collected. On an idle host,
hold every existing candidate flag and thread setting constant, change only
`DA3_PACKED_QKV`, and collect ten alternating same-binary pairs using the
per-layer `DA_PHASE_PROFILE` QKV timers plus uninstrumented end-to-end timing.

Retain this candidate only if all conditions hold:

- aggregate QKV time improves by at least 15%;
- uninstrumented end-to-end median improves by at least 2 ms in at least eight
  of ten pairs; and
- the four-image C++ F32 parity gate remains at Pearson r >= 0.9999 and MAE
  <= 0.005 for every image.

Only then may it enter the locked randomized independent-trial study. This is
a locality hypothesis, not a claim that it closes the remaining target gap.
