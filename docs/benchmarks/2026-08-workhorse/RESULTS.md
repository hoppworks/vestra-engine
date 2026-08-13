# Scientific benchmark results

Generated: `2026-08-12T06:39:15.419820+00:00`

The primary estimator is the arithmetic mean of 10 independent trial medians;
the interval is a two-sided 95% Student-t confidence interval across trials.

| Runtime | Device | Precision | Mean median (ms) | 95% CI (ms) | SD (ms) | p95 pooled (ms) | RSS (MiB) |
|---|---|---|---:|---:|---:|---:|---:|
| cpp | cpu | q8_0 | 240.972 | [238.424, 243.521] | 3.563 | 260.203 | 367.0 |
| cpp | cpu | f32 | 246.683 | [245.629, 247.737] | 1.474 | 260.296 | 618.5 |
| pytorch | cpu | f32 | 251.894 | [250.377, 253.411] | 2.121 | 260.083 | 1599.8 |
| cpp | cpu | q4_k | 282.049 | [279.894, 284.205] | 3.014 | 292.554 | 324.2 |
| rust | cpu | f32 | 2954.256 | [2951.567, 2956.945] | 3.759 | 2967.247 | 1047.9 |
| pytorch | gpu | f32 | 15.944 | [15.924, 15.965] | 0.028 | 16.114 | 1600.0 |
| cpp | gpu | q8_0 | 21.239 | [21.214, 21.265] | 0.035 | 22.777 | 690.3 |
| cpp | gpu | q4_k | 21.287 | [21.252, 21.323] | 0.049 | 22.891 | 648.7 |
| cpp | gpu | f32 | 22.495 | [22.430, 22.560] | 0.091 | 24.041 | 933.0 |

## Interpretation rules

- Direct implementation comparisons use only F32 arms on the same device.
- Q8_0 and Q4_K are compression/accuracy trade-offs, not same-precision speed claims.
- CPU and GPU results are separate populations and must not be presented as language-only effects.
- Timing excludes model load and image decode; it includes preprocessing, backbone, depth/confidence head and host postprocessing.
- Raw commands, trial order, per-iteration samples, hashes and hardware metadata are preserved in `raw-results.json`.

## Findings

The current Rust implementation is **not faster** than the optimized C++
reference. At equal F32 precision on CPU it is 11.98× slower
(`2954.256 / 246.683`) and uses 1.69× the peak resident memory. Profiling locates
the gap primarily in the Rust backbone (~2.29 s) and depth head (~0.78 s), not
preprocessing (~5 ms). The earlier portfolio hypothesis of a 20–30% Rust
speedup is rejected by this experiment and must not be published as a result.

C++ F32 is 2.1% faster than official PyTorch F32 on CPU. Their 95% confidence
intervals do not overlap, but the absolute difference is only 5.211 ms on this
host. On GPU, official PyTorch F32 is 1.41× faster than C++ F32. These are
runtime-specific results, not evidence that one programming language is
generally faster.

On CPU, C++ Q8_0 is 2.3% faster than C++ F32 and reduces peak RSS by 40.7%.
Q4_K reduces RSS by 47.6% but is 14.3% slower than F32 on this workload. On the
RTX 5080, Q8_0/Q4_K are slightly faster and smaller than C++ F32, although
official PyTorch F32 remains the fastest measured GPU arm.

## Fidelity and quantization drift

| Comparison (four images) | Mean Pearson r | Mean MAE | Declared gate |
|---|---:|---:|---|
| C++ F32 vs official PyTorch F32 | 0.9999767 | 0.0025500 | pass |
| Rust F32 vs C++ F32 | 0.9999824 | 0.0020204 | pass |
| C++ CUDA F32 vs C++ CPU F32 | 0.9999995 | 0.0005563 | pass |
| C++ Q8_0 vs C++ F32 | 0.9999772 | 0.0024849 | pass |
| C++ Q4_K vs C++ F32 | 0.9961865 | 0.0628504 | descriptive only |

Every thresholded comparison passes Pearson r ≥0.9999 and MAE ≤0.005 on every
individual image. Q4_K shows materially larger drift and is therefore presented
only as an explicit memory/quality trade-off. These values demonstrate
implementation fidelity to the same model; they do not measure depth accuracy
against real ground truth.

See [PROTOCOL.md](PROTOCOL.md) for the locked measurement boundary, statistical
method and limitations, [raw-results.json](raw-results.json) for all 900 timing
samples, and [quality-results.json](quality-results.json) for all per-image
fidelity values.
