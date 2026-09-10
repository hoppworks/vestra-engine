# Iteration 62 — panel-packed MLP candidate

## Hypothesis and change

The direct AVX projection route repeatedly walks narrow 64-column weight
panels with a full-row stride. This candidate packs immutable DA3-BASE F32
projection weights at model load into contiguous `[panel][K][64]` storage and
executes K-ascending AVX-512 FMAs from that layout. Model loading is outside
the locked timed boundary; all input-dependent LayerNorm, GEMM, bias, GELU,
LayerScale, and residual work remains timed.

`vestra-kernels` commit `c650933` owns the packed layout and kernel. Engine
commit `363c752` builds packed MLP layers only when `DA3_PACKED_MLP=1` is set;
the qualified default remains unchanged.

## Correctness

- Kernel pack/unpack tests preserve sampled source weight bits and reject
  non-DA3 shapes.
- `cargo test --locked` in `vestra-kernels`: 40 unit tests plus 12 integration
  tests passed.
- `cargo test --locked -p vestra-engine --lib`: 70 tests passed.
- Four-image Rust versus C++ F32 parity:

| Image | Pearson r | MAE |
|---|---:|---:|
| canyon | 0.9999936282 | 0.0018124947 |
| desk | 0.9999782566 | 0.0017728056 |
| mountains | 0.9999855792 | 0.0036749605 |
| street | 0.9999721254 | 0.0008209984 |

Every image remains within the locked `r >= 0.9999`, `MAE <= 0.005` gate.

## Smoke evidence — inconclusive

The Workhorse remains under unrelated CPU load. Two alternating profile pairs
summed the twelve `mlp_with_ln2` timers as follows:

| Pair | Normal BLIS route | Packed candidate |
|---|---:|---:|
| 1 | 88.783 ms | 76.386 ms |
| 2 | 72.718 ms | 77.477 ms |

The change is neither accepted nor promoted: its explicit admission target is
`<= 43 ms` aggregate MLP time, including all runtime work. The first-stage
panel layout is retained behind its opt-in switch solely so the next
cache-blocked/superkernel iteration has a real-model parity and timing route.

## Iteration 63 — cache-blocked 12x32 variant (reverted)

Kernel commit `6a53f8b` changed the candidate to 12x32 register tiles and a
two-dimensional 60-row by 256-column work scheduler. It retained the same
packed model weights and ascending-K FMA order, and passed the same four-image
parity gate:

| Image | Pearson r | MAE |
|---|---:|---:|
| canyon | 0.9999936282 | 0.0018124947 |
| desk | 0.9999782566 | 0.0017728056 |
| mountains | 0.9999855792 | 0.0036749605 |
| street | 0.9999721254 | 0.0008209984 |

It failed the performance admission decisively in alternating same-binary
profiles while the Workhorse was busy: normal MLP totals were 73.269 ms and
65.467 ms; the 12x32 candidate was 136.275 ms and 149.854 ms. This is large
enough to reject even before an idle-machine study. Engine commit `e669576`
was cleanly superseded by a dependency rollback to `c650933`; the failed
kernel revision remains only in the independently versioned kernel history
and is not imported by Vestra Engine.
