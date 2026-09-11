# Current CPU F32 baseline — Ryzen 9 9950X

## Result

This is a fresh, randomized direct CPU F32 comparison of the accepted Vestra
route and the C++/ggml reference. It is a baseline requalification, not a
claim that the 1.5x objective has been reached.

| Runtime | Mean of 10 trial medians (ms) | 95% CI (ms) | Trial-median SD (ms) |
|---|---:|---:|---:|
| Vestra Engine (Rust) | 158.713 | [156.077, 161.349] | 3.685 |
| C++/ggml | 197.210 | [196.163, 198.257] | 1.464 |

Rust is 24.26% lower latency / 1.243x throughput in this study. The confidence
intervals do not overlap, but the locked 1.5x goal remains open: the Rust arm
must reach at most 131.959 ms against the original 197.939 ms reference.

## Locked workload

- Hardware: AMD Ryzen 9 9950X; 16 benchmark threads.
- Model: `depth-anything-base-f32.gguf`.
- Image: `mountains.jpg`; 504x336 inference resize.
- Timed work: preprocessing, backbone, depth/confidence head and
  postprocessing; model load and image decoding excluded by both CLIs.
- Each independent process: one untimed warm-up followed by ten timed
  iterations. Ten paired trials alternated starting arm.
- Common accepted Rust switches: `DA3_STRIP_MLP`, `DA3_RN1_F2_OC32`,
  `DA3_STRIP_MLP_HOIST_DISPATCH`, `DA3_KERNELS_ENABLE_OUT1_F2_128X64`,
  `DA3_FUSED_FINAL_RESIZE_ROW_RING`, `DA3_KERNELS_BLIS_LINEAR`, and
  `DA3_HEAD_BLIS_GEMM`.

## Raw trial medians

| Trial | Rust ms | C++ ms |
|---:|---:|---:|
| 1 | 161.384 | 194.600 |
| 2 | 158.575 | 197.700 |
| 3 | 158.826 | 200.000 |
| 4 | 160.040 | 196.600 |
| 5 | 150.596 | 196.300 |
| 6 | 163.721 | 196.700 |
| 7 | 162.051 | 198.800 |
| 8 | 158.396 | 197.300 |
| 9 | 158.106 | 197.500 |
| 10 | 155.436 | 196.600 |

The primary estimator is the arithmetic mean of the ten independent trial
medians. The 95% intervals use the two-sided Student-t multiplier 2.262 with
n=10. The full per-iteration command log is retained on Workhorse at
`/var/tmp/vestra-current-formal-manual-20260911/runner.log`.

## Commands

The Rust arm used:

```text
vestra-engine bench --model depth-anything-base-f32.gguf --image mountains.jpg --warmup 1 --repeat 10
```

The C++ arm used:

```text
da3-cli depth --model depth-anything-base-f32.gguf --input mountains.jpg --threads 16 --repeat 10
```

The runner alternated `Rust → C++` for odd trials and `C++ → Rust` for even
trials, with a three-second cooldown between arms. Source state: Vestra Engine
`e4b37b3`; C++ checkout `/var/roothome/da3-current-upstream-739992d`.

