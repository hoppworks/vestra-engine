# Benchmark protocol

## Research question

How do the official PyTorch implementation, the optimized C++/ggml port and
the Rust reimplementation compare when they execute the same DA3-BASE
single-image depth workload on the same machine?

The primary comparison is F32 against F32 on the same device. Q8_0 and Q4_K
are secondary C++ compression arms and are reported with fidelity drift; they
must not be used as evidence for a language-only speedup.

## Locked environment

- Host: AMD Ryzen 9 9950X (16 physical/32 logical cores), 96 GiB RAM,
  NVIDIA GeForce RTX 5080 16 GiB.
- CPU concurrency: 16 threads for every implementation.
- Input: `mountains.jpg`, SHA-256
  `936d60f43c51fe99156563a0d3c5da69cf84a39cbde5e443bea7662500b8c969`.
- Resolution policy: official DA3 upper-bound resize to 504; the input becomes
  504×336 in every implementation.
- C++: localai-org/depth-anything.cpp commit
  `2028b47ac75a8659c6a9aa617baf09be193eb55f`, ggml commit
  `eced84c86f8b012c752c016f7fe789adea168e1e`.
- Official model code: ByteDance Depth Anything 3 commit
  `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`.
- PyTorch: 2.12.1+cu130. C++ CUDA: CUDA Toolkit 13.0.
- Model and executable hashes are stored in `raw-results.json`.

## Timed work

Each process loads its model once. The input image is decoded once outside the
timed region. One complete untimed warm-up runs before measurement. Every
sample then includes:

1. image resize and ImageNet normalization;
2. patch embedding and token preparation;
3. DA3-BASE backbone;
4. depth and confidence head;
5. host-side depth/confidence postprocessing.

Model load, JPEG decode, camera-pose inference and output-file writing are not
timed. The C++ benchmark hook is patched by `scripts/cpp-steady-state-bench.patch`
to enforce this same boundary and emit raw iteration samples. PyTorch is patched
by `scripts/torch-raw-samples.patch` only to emit those already measured samples.
Rust uses `Engine::infer_depth`, so it does not perform extra pose work.

## Experimental design

- 10 independent process trials per arm.
- 10 timed samples per trial after one warm-up (100 raw samples per arm).
- CPU and GPU are separate suites.
- Arm order is shuffled independently inside every trial with fixed seed
  `20260812` to reduce order/thermal bias.
- Three-second cooldown between process trials.
- Peak resident memory is captured by `/usr/bin/time -v`.
- The primary estimator is the arithmetic mean of the 10 trial medians.
- Dispersion is the sample standard deviation of trial medians.
- The interval is a two-sided 95% Student-t confidence interval (`df=9`,
  critical value 2.262).
- Pooled p95 is the nearest-rank 95th percentile over all 100 raw samples.

The full randomized order, commands and every raw sample are retained in
`raw-results.json`. Re-run inside the pinned benchmark container with:

```bash
python depth-anything-rs/scripts/run_scientific_benchmark.py \
  --trials 10 --repeat 10 --threads 16 --cooldown 3 --seed 20260812
```

## Fidelity protocol

Runtime equivalence is checked on four heterogeneous repository images:
`canyon`, `desk`, `mountains`, and `street`. All outputs are 504×336.

The operational implementation-fidelity gate is Pearson correlation ≥0.9999
and mean absolute error ≤0.005 on every image. It applies to official PyTorch
F32 vs C++ F32, Rust F32 vs C++ F32, C++ CUDA F32 vs C++ CPU F32, and C++ Q8_0 vs C++ F32. Q4_K is reported
without a pass/fail threshold as a compression trade-off. Complete per-image
values are in `quality-results.json`. This gate was fixed for the durable report
after exploratory parity runs, so it is a verification threshold rather than a
preregistered statistical hypothesis test.

This is an implementation-parity test, not a claim about real-world depth
accuracy. Dataset-level accuracy would require ground-truth depth benchmarks
such as NYU Depth V2 or KITTI and is outside this same-model runtime question.

## Threats to validity

- One hardware configuration and one timed image limit external validity.
- Ten process trials quantify run-to-run variation but do not capture variation
  across operating systems, compilers or CPUs/GPUs.
- F32 implementations can still use different lower-level math kernels and
  operation orders; the fidelity suite bounds observed output drift.
- RSS measures process high-water memory, not allocator fragmentation or GPU
  VRAM. GPU memory should be added in a future dedicated measurement.
- The Rust source was benchmarked from a working-tree snapshot; executable
  SHA-256, source files and scripts in this repository are therefore the
  reproducibility anchor, not the base commit alone.
