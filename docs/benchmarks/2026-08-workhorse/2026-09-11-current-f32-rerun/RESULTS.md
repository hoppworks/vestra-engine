# Current CPU-F32 remeasurement — Ryzen 9 9950X

## Result

This is a fresh, randomized direct comparison of the accepted Vestra Engine
route and the current public C++/ggml reference. It replaces neither the
historical 20-trial release record nor its raw evidence; it is the current
ten-trial portfolio snapshot for the same locked workload.

| Runtime | Mean of 10 trial medians (ms) | 95% CI (ms) | Trial-median SD (ms) |
|---|---:|---:|---:|
| Vestra Engine (Rust) | 158.126 | [156.196, 160.056] | 2.698 |
| C++/ggml | 197.036 | [196.325, 197.748] | 0.995 |

Vestra has **19.75% lower latency** and **24.61% higher throughput** in this
study. The confidence intervals do not overlap. This is a same-machine,
single-image CPU-F32 result; it is not a GPU, multiview, quantized, or complete
reconstruction performance claim.

## Locked workload

- Hardware: AMD Ryzen 9 9950X, 16 benchmark threads; 96 GiB host memory.
- Model: `depth-anything-base-f32.gguf`.
- Image: `mountains.jpg`; 504x336 inference resize.
- Timed work: preprocessing, backbone, depth/confidence head, and host
  postprocessing. Model loading and image decoding are excluded.
- Each independent process: one untimed warm-up followed by ten timed
  iterations. Ten trials per arm were randomized with seed `20260911` and a
  three-second arm cooldown.
- Rust source: `107ddf3c554baa390e5ae1937cd3ef11d3c3825d`; C++ source:
  `739992d10bf9472c46dcd4622b14d2b20766c58d`; ggml:
  `eced84c86f8b012c752c016f7fe789adea168e1e`.

The exact binaries, hashes, model/input hashes, trial order, per-iteration
samples, host fingerprint, and enabled runtime switches are preserved in
[`raw-results.json`](raw-results.json).

## Fidelity gate

The Rust F32 output was regenerated against the materialized C++ F32 reference
on the four locked images after the timing study:

| Image | Pearson r | MAE |
|---|---:|---:|
| canyon | 0.9999936282 | 0.0018124970 |
| desk | 0.9999782565 | 0.0017728067 |
| mountains | 0.9999855792 | 0.0036749614 |
| street | 0.9999721254 | 0.0008209983 |

Every image meets the required `r >= 0.9999` and `MAE <= 0.005` threshold.
