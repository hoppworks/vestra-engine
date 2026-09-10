# rn1 F(2) OC32 product candidate — 2026-09-10

## Hypothesis

rn1 contains four ReLU-input `128 -> 128` residual convolutions at `96 × 144`
or its portrait orientation. Each uses F(2) Winograd product blocks with four
tiles. The established generic AVX-512 product kernel computes one OC16 panel
at a time and broadcasts each transformed activation once for that panel.

The candidate pairs two adjacent OC16 panels into one OC32 register block.
Each of the four transformed activation broadcasts is shared between both
panels. Every individual output lane retains its existing ascending-input FMA
order; no reduction is split or reassociated.

## Static cost model

The candidate leaves 3,623,878,656 scalar FMA operations (7.248 GFLOP) across
rn1 unchanged. It changes the four-tile block as follows:

| Property | Generic OC16 | Candidate OC32 |
|---|---:|---:|
| AVX-512 FMA operations | 65,536 | 65,536 |
| Activation broadcasts | 65,536 | 32,768 |
| Independent accumulators | 4 | 8 |
| Product scratch | 64 KiB | 64 KiB |

This is a register/dataflow hypothesis, not a claim of reduced arithmetic or
DRAM traffic. Related broad tiles4 and larger-tile experiments were previously
rejected; this candidate changes only rn1's fixed channel blocking.

## Change

`vestra-kernels` `d3a9b8b` adds
`winograd_f2_blocked_rn1_128x128_tiles4_f32`, selected only when all of these
are true:

- `DA3_RN1_F2_OC32=1`;
- ReLU-input F(2) path;
- `128 -> 128` channels;
- four active tiles; and
- `96 × 144` or `144 × 96` input geometry.

All other shapes, partial tile blocks, disabled flags and unsupported ISAs use
the established generic route.

## Verification

`cargo test -p vestra-kernels --lib`: 46/46 passed locally.

The target-only regression allocates signed, irregular exact rn1-shaped
buffers, initializes output with NaNs, compares every product value bitwise
against the generic AVX-512 F(2) route, and requires the special route to
report that it executed. The local non-AVX host intentionally does not count
that test as executed; it must run on the Workhorse before promotion.

### Target ISA execution

On the AMD Ryzen 9 9950X Workhorse, a clean temporary clone at kernel commit
`d3a9b8b` ran `RAYON_NUM_THREADS=1 cargo test -p vestra-kernels rn1_oc32
--lib`. The target-only AVX-512/FMA test executed (not skipped) and passed
`1/1`, proving bitwise equality with the generic F(2) product on the target
ISA. The Workhorse still had unrelated interactive load, so this is strictly a
correctness result and contains no timing claim.

## Predeclared admission

The Workhorse is occupied by unrelated workloads, so no timing is recorded.
When idle, compare otherwise identical binaries and candidate flags with only
`DA3_RN1_F2_OC32` changed:

- ten alternating pairs, each with one warm-up and ten timed iterations;
- at least 15% lower summed rn1 `rc1 + rc2` time;
- at least 2.0 ms lower end-to-end latency in at least eight pairs; and
- unchanged four-image C++ F32 parity: Pearson r >= 0.9999 and MAE <= 0.005
  for every image.

Only then may this candidate join the randomized full study.
