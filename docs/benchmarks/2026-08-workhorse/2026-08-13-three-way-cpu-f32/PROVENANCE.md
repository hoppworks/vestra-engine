# Same-machine DA3-BASE CPU F32 study — three runtimes

This study answers the direct same-machine question: how do the optimized Rust
runtime, C++/ggml reference, and official PyTorch DA3 implementation perform
on the Workhorse when executing the same single-image DA3-BASE F32 workload?

## N=10 result

| Runtime | Mean of 10 trial medians | 95% CI | Trial-median SD | Pooled p95 |
|---|---:|---:|---:|---:|
| Rust | 171.087 ms | [164.408, 177.767] ms | 9.338 ms | 187.590 ms |
| C++/ggml | 239.413 ms | [237.173, 241.652] ms | 3.130 ms | 252.267 ms |
| Official PyTorch | 256.835 ms | [254.640, 259.030] ms | 3.069 ms | 270.987 ms |

At N=10, Rust is 39.9% faster in throughput terms than C++/ggml and 50.1%
faster than official PyTorch. C++/ggml is 7.3% faster than official PyTorch.
These statements use the reciprocal latency convention (`reference / candidate
- 1`), not a percentage of the candidate latency.

The official PyTorch and C++ intervals are stable. Rust's N=10 trial-median
SD is 9.338 ms, so the existing N=20 Rust-vs-C++ revalidation remains the
stronger direct two-arm estimate. This three-arm study is retained as the
correct same-machine PyTorch comparison; extend it to N=20 if a single
higher-confidence three-arm table is required.

## Locked conditions

- Hardware: AMD Ryzen 9 9950X Workhorse.
- Model: DA3-BASE F32, one view, `mountains.jpg` resized to 504x336.
- CPU budget: 16 threads in every arm.
- Per fresh process: model loaded once; image decoded once; one untimed
  warm-up; then 10 timed iterations.
- Study: 10 randomized trials per arm (300 timed samples total), seed
  `20260812`, three-second cooldown between process trials.
- Timed work: preprocessing, DA3 backbone, depth/confidence head, and host
  postprocessing. Model load and JPEG decode are excluded.

## Official PyTorch arm

- Official Depth Anything 3 source:
  `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`.
- PyTorch: `2.7.1+cu128`, running CPU F32 only; CUDA is not used in this
  study.
- The official source was run in an isolated, persistent container. The timer
  reads the official runner's emitted internal iteration samples, so Podman's
  per-process startup is outside the timed samples.
- The high-level API's automatic mixed-precision wrapper was deliberately
  bypassed. The runner uses the official preprocessing, `DepthAnything3Net`
  model, and official output conversion in F32 so that the arm matches the
  F32 comparison contract.
- Peak RSS for PyTorch is not reported: `/usr/bin/time` can observe only the
  Podman client, not the persistent Python container process.

## Numerical fidelity

`official-pytorch-parity.json` contains a fresh four-image comparison with
C++/ggml F32. Every image passes the declared gate (Pearson r >= 0.9999 and
MAE <= 0.005): canyon r=0.99999136 / MAE=0.00180643; desk
r=0.99997486 / MAE=0.00290071; mountains r=0.99998025 / MAE=0.00366968;
street r=0.99996016 / MAE=0.00182240.

## Integrity

- `raw-results.json` SHA-256:
  `94214ff8f5a44e14551e5ae5f4ced208d9117baa00d4adb59ced2a755764ae55`
- `RESULTS.md` SHA-256:
  `bf0938fbf5f4dbcc0458e326bcea723f3b5796c8d039fd12ec45be16c6472284`
- `official-pytorch-parity.json` SHA-256:
  `bc73d70cd60b52b118464dccb491a05fdd1fe4c60f341afa84a643085a1121eb`

GPU and Depth Anything V2/DA2K comparisons are separate studies. They must
not be combined with this CPU DA3-BASE implementation comparison.
