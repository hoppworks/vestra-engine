# Repository-split CPU-F32 validation

This is the required post-split validation run. It uses the split Vestra
Engine and Vestra Kernels trees, not the historical in-tree kernel crate.

| Runtime | Mean of 10 trial medians | 95% CI |
|---|---:|---:|
| Vestra Engine (split) | 172.260 ms | [166.546, 177.974] ms |
| C++/ggml control | 240.540 ms | [238.263, 242.817] ms |

The split Vestra Engine is 39.6% higher-throughput than the same-window C++
control, equivalently 28.4% lower latency. Its result is consistent with the
pre-split 20-trial Rust revalidation of 171.141 ms; this validates no relevant
runtime regression from the repository boundary change.

## Contract

- Host: AMD Ryzen 9 9950X, 16 benchmark threads, quiet at launch.
- Workload: identical DA3-BASE F32 GGUF and `mountains.jpg`, resized to
  504×336; loading and JPEG decode excluded.
- Per trial: one warm-up and ten timed iterations; ten fresh processes per
  arm, with a three-second cooldown.
- Split build: `-C target-cpu=znver5 --cfg da3_blis`, with the recorded BLIS
  runtime libraries and the two qualified BLIS environment switches.
- Kernel revision: `b35a91742ee03cb06d5686ef1ec74ff895151427`.

The trial order was not interleaved across arms, so this is a post-split
regression validation, not a replacement for the existing randomized study.
The raw records are retained unchanged: `rust-raw.txt` SHA-256
`eb0f1f14c10ca181f86f013d852bc7a633b945c0c0e312154eebaefeb9ab2c51` and
`cpp-raw.txt` SHA-256
`0ab32e5c20c6f70fd5e72bbcc2727c8952354935bc259c322c489c2b0c993a47`.

## Numerical parity

The four-image C++ F32 parity gate passed unchanged after the split:

| Image | Pearson r | MAE |
|---|---:|---:|
| canyon | 0.999993628 | 0.001812542 |
| desk | 0.999978257 | 0.001772926 |
| mountains | 0.999985578 | 0.003675167 |
| street | 0.999972124 | 0.000821043 |

Every image exceeds r ≥ 0.9999 and stays below MAE ≤ 0.005.
