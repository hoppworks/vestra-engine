# Iteration 66 — OC16 DPT reassemble transposed convolution

## Hypothesis and change

The two non-overlapping DPT reassemble transposed convolutions execute
254,803,968 MACs at the locked 504×336 geometry:

| Stage | Shape | MACs |
|---|---|---:|
| 0 | 96→96, k4/s4 | 127,401,984 |
| 1 | 192→192, k2/s2 | 127,401,984 |

The former AVX-512 route reduced sixteen input-channel lanes separately for
each output channel, requiring horizontal reductions and rereading the same
HWC pixel for many output channels. Kernel commit `2fff281` adds a separate
model-owned `[oc16][ky][kx][ic][lane]` layout. One AVX-512 accumulator holds
sixteen independent output channels; it uses ascending input-channel FMA and
scatters once into the established CHW output layout.

Engine commits `a2e248f` and `641ec7c` cache both transpose filters with the
loaded model and add per-stage timing. The OC16 route is now the default on a
supported CPU. `DA3_DISABLE_TRANSPOSE_OC16=1` restores the preceding prepared
filter route. Unsupported ISA continues to use the generic implementation.

## Correctness

- `oc16_transpose_matches_generic_nonoverlap_oracle` compares a signed,
  random 16→16 non-overlap convolution against the generic oracle.
- `cargo test --locked` in `vestra-kernels`: 42 unit tests plus 12 integration
  tests passed.
- `cargo test --locked -p vestra-engine --lib`: 71 tests passed.

Full C++ F32 parity on Workhorse:

| Image | Pearson r | MAE |
|---|---:|---:|
| canyon | 0.9999936280 | 0.0018125213 |
| desk | 0.9999782568 | 0.0017729259 |
| mountains | 0.9999855775 | 0.0036752036 |
| street | 0.9999721240 | 0.0008210033 |

Every image passes the locked `r >= 0.9999`, `MAE <= 0.005` gate.

## Short isolated timing

The Workhorse remained busy, so this is not a final benchmark claim. With
identical loaded model, image, resize, 16-thread budget, and timed boundary,
the post-warmup per-stage profile was:

| Route | Stage 0 | Stage 1 | Sum |
|---|---:|---:|---:|
| Prior prepared filter | 1.027 ms | 1.189 ms | 2.216 ms |
| OC16 candidate | 0.719 ms | 0.752 ms | 1.471 ms |

The measured local saving is 0.745 ms. This is sufficient to retain the
strictly paritied micro-optimization, but it is not counted in the qualified
whole-inference headline until the idle-machine randomized study is rerun.
