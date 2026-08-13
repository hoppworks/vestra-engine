# DA-2K / Depth Anything V2 study protocol

## Purpose

This study has two products that must never be collapsed into one leaderboard:

1. **Model and quality comparison.** Reproduce the public DA-2K comparison
   around Depth Anything V2 with the official V1, V2, Marigold, GeoWizard and
   DepthFM inference pipelines. This answers which model produces what
   relative-depth quality under a pinned profile.
2. **Same-model runtime comparison.** Run the identical Depth Anything
   V2-Small checkpoint and graph through the official PyTorch runtime and the
   Rust runtime. This is the only comparison allowed to say that one runtime
   is faster than another.

DA3-BASE CPU-F32 has a separate, completed 44%-faster study. It is not a
substitute for a V2-Small same-model benchmark and must not be merged into the
V2 table.

## Study arms

### DA-2K quality/model arms

| Arm | Pinned official profile | What is reported |
|---|---|---|
| Depth Anything V1 | official checkpoint and runner | DA-2K pairwise accuracy, latency, memory |
| Depth Anything V2 Small | official checkpoint and runner | DA-2K pairwise accuracy, latency, memory |
| Depth Anything V2 Large | official checkpoint and runner | DA-2K pairwise accuracy, latency, memory |
| Marigold | official-quality plus separately labelled fast profile | DA-2K pairwise accuracy, latency, memory |
| GeoWizard | official-quality plus separately labelled fast profile | DA-2K pairwise accuracy, latency, memory |
| DepthFM | official-quality plus separately labelled fast profile | DA-2K pairwise accuracy, latency, memory |

Each arm owns its model and weights. The table may compare quality and cost,
but must say “different models” rather than imply a language-only speed race.
Every checkpoint, runner commit, diffusion step count, ensemble size, input
policy and licence is pinned in the machine-readable manifest before a first
timing run.

### V2-Small same-model runtime arms

| Arm | Required identity |
|---|---|
| Official V2 PyTorch | `depth_anything_v2_vits.pth`, official OpenCV pipeline |
| Hugging Face Transformers | official conversion, separately labelled because Pillow/OpenCV upsampling differs |
| Core ML | Apple V2-Small package, Apple Silicon suite only |
| ONNX Runtime | project-owned export, accepted only after output parity |
| Rust | exact V2-Small graph and weights used by the official reference |

The Rust V2 arm does not exist yet. Implementing/exporting it, proving output
parity, and only then timing it is a prerequisite—not a reporting shortcut.

## Hardware suites

| Suite | Host | Allowed arms |
|---|---|---|
| Linux CUDA | Ryzen 9 9950X + RTX 5080 | all official model arms and V2-Small runtime arms with CUDA support |
| Linux CPU | Ryzen 9 9950X | PyTorch V1/V2 and Rust V2-Small; CPU-supported model arms only |
| Apple Silicon | separate named Mac | PyTorch, Core ML and Rust V2-Small where available |

CPU, CUDA and Apple Silicon results are separate populations. A table never
mixes them or treats a different processor as noise. Every result names the
processor, GPU, RAM, OS, driver, runtime, compiler, thread budget and power
state.

## Fair timing boundary

For an image-latency trial, every arm receives the same pre-extracted RGB
frame files. The harness reports these independently:

1. decode;
2. preprocess;
3. model inference;
4. postprocess and depth export; and
5. end-to-end, the sum of all four.

Model download, checkpoint conversion, model loading, compilation and initial
GPU graph capture are excluded from steady-state latency and reported in a
separate setup table. Video decoding is also a separate end-to-end video
suite; it is never silently included for one image arm but not another.

Each process runs one unmeasured warm-up. The primary result is the mean of
ten independent process-trial medians, where each trial has ten timed
inferences. Arm order is randomized with a recorded seed and a fixed cooldown.
Report 95% Student-t confidence intervals, pooled p95, peak RSS and peak VRAM.
No outlier is removed unless a rule was published before the run.

## Quality and parity gates

The DA-2K score is pairwise relative-depth accuracy from the official
annotations. It is a model-quality metric, not implementation parity.

For any runtime that is intended to execute the same V2-Small model, compare
its raw depth output with the official V2 OpenCV runner on a fixed four-image
corpus before timing. The output tolerance is recorded with the runner and
resampling implementation. A runtime without an accepted parity result may
appear only as an unqualified prototype, never in the direct speed table.

## Reproducibility manifest

Before execution, create one immutable JSON manifest per suite with:

- repository URL and commit for every runner;
- checkpoint URL, licence, SHA-256 and byte size;
- container image or lockfile hashes;
- exact command, image manifest and SHA-256 hashes;
- model input resolution and resampling policy;
- precision, device, effective thread count, CUDA/driver/runtime versions;
- warm-up, iteration, trial, cooldown and seed settings;
- binary/source-tree hashes and host fingerprint.

The harness stores raw per-iteration data and produces the report from that
data. It must fail closed if a required hash, checkpoint, resolution or output
shape differs from the manifest.

## Execution order

1. Pin repositories, checkpoints and DA-2K dataset; write manifests.
2. Implement the common frame manifest, result schema and validation tool.
3. Run each official quality arm unchanged and validate DA-2K scoring.
4. Run official V2-Small PyTorch as the runtime reference on each supported
   hardware suite.
5. Implement Rust V2-Small, prove raw-depth parity, then run the same runtime
   protocol.
6. Add Transformers, Core ML and ONNX only after each has a documented output
   tolerance and a clean manifest.
7. Publish a model table and a same-model runtime table with raw artifacts.

## Publication language

Allowed: “On the named Ryzen 9 9950X CPU-F32 suite, Rust executes the same
pinned V2-Small model X× faster than the official PyTorch reference.”

Not allowed: “Rust is faster than Marigold/GeoWizard/DepthFM” without saying
that those are different models, quality settings and model sizes. Similarly,
do not compare a V2-Small runtime number to the existing DA3-BASE result.
