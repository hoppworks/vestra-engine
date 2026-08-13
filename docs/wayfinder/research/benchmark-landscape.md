# Benchmark landscape: Depth Anything V2

## Decision

Build **two explicitly separate benchmark suites**. A single leaderboard would
otherwise imply a false comparison:

1. **Model benchmark**: the four public baselines used in the official Depth
   Anything V2 DA-2K comparison, plus V2 and our engine.
2. **Runtime benchmark**: implementations of the *same* V2-Small model. This
   is the only suite allowed to make a claim such as “our runtime is X faster”.

The official V2 paper's DA-2K table lists exactly Marigold, GeoWizard, DepthFM,
and Depth Anything V1 as community-model baselines. The paper and project page
make a qualitative `>10x` efficiency claim versus diffusion models, but neither
ships a timing harness, timing data, nor a hardware/precision protocol to
reproduce that claim. Therefore the project must publish its own pinned harness
instead of calling an execution a reproduction of an official speed benchmark.

Sources: [V2 paper, Table 3](https://arxiv.org/html/2406.09414),
[V2 project page](https://depth-anything-v2.github.io/),
[V2 repository](https://github.com/DepthAnything/Depth-Anything-V2).

## A. Model benchmark (the public V2 comparison set)

| Arm | Official runner / pipeline | Weights and compatibility | macOS status | License / caveat |
|---|---|---|---|---|
| Depth Anything V1 | `run_video.py --encoder vits|vitb|vitl` | Own V1 checkpoints; not compatible with V2 checkpoints | PyTorch runner; no official MPS test matrix | Repo is Apache-2.0; pin checkpoint licence separately. [Source](https://github.com/LiheYoung/Depth-Anything) |
| Depth Anything V2 | `run_video.py --encoder vits|vitb|vitl` | V2 `.pth` checkpoints; Small 24.8M, Base 97.5M, Large 335.3M | Official code selects CUDA, then MPS, then CPU | Small model Apache-2.0; Base/Large/Giant CC-BY-NC-4.0. [Source](https://github.com/DepthAnything/Depth-Anything-V2) |
| Marigold Depth v1.1 | `script/depth/run.py` on an image directory | Its own Stable-Diffusion-derived checkpoint; cannot share V2 weights | `--apple_silicon` is explicit, but official tested platform is Ubuntu/CUDA | Code Apache-2.0; model is RAIL++-M. Quality knobs: ensemble and denoise steps. [Source](https://github.com/prs-eth/marigold) |
| GeoWizard | `run_infer.py` on an image directory | Own diffusion checkpoint; cannot share V2 weights | Officially tested only on Ubuntu 22.04 / CUDA 11.8 | CC BY 4.0. Default 3 ensembles × 10 denoise steps; academic setting 10 × 50. [Source](https://github.com/fuxiao0719/GeoWizard) |
| DepthFM | `inference.py --num_steps --ensemble_size --img` | Own flow-matching/SD 2.1-derived checkpoint; cannot share V2 weights | Officially tested only on Ubuntu 22.04 / CUDA 12.4; runner directly calls CUDA | MIT code; checkpoint terms must be pinned when downloaded. [Source](https://github.com/CompVis/depth-fm) |
| Our engine | one V2-Small arm and, separately, V2-Large research arm | Must consume the exact pinned V2 checkpoint/export that the reference arm uses | Target: macOS CPU and Apple Silicon; GPU results separate | V2-Small is the commercial-safe default. |

### Implications

- These are **model competitors**, not six interchangeable runtimes. The
  weights and architectures differ. Never compare raw latency and imply equal
  accuracy; report DA-2K/quality results and latency side by side.
- All baseline repositories are image-first except DA V1/V2's official video
  runners. The harness must decode the same pre-extracted frame manifest for
  every arm. Video decode, inference, depth export, and point-cloud assembly
  must be independently timed.
- Diffusion/flow methods have quality-speed knobs. Each gets two pinned
  profiles: `official-quality` and `fast`. V2 gets `small` and `large` as
  distinct model rows, not profiles of the same model.
- GPU-only arms belong in a Linux/CUDA report. Do not force GeoWizard or
  DepthFM onto a Mac CPU and present it as a fair product ranking.

## B. V2-Small runtime benchmark (the speed claim)

| Runtime arm | Model identity | Platform role | Important qualification |
|---|---|---|---|
| Official V2 PyTorch | Official `depth_anything_v2_vits.pth` | Reference for CPU, MPS and CUDA where available | Includes a runnable `run_video.py` and selects CUDA → MPS → CPU. [Source](https://github.com/DepthAnything/Depth-Anything-V2/blob/main/run_video.py) |
| Hugging Face Transformers | `depth-anything/Depth-Anything-V2-Small-hf` conversion | Public PyTorch baseline | The V2 authors explicitly warn results can differ because Transformers uses Pillow while their runner uses OpenCV for upsampling. It is a runtime/API comparison, not bit-identical verification. [Source](https://github.com/DepthAnything/Depth-Anything-V2#use-our-models) |
| Apple Core ML | Apple's V2-Small Core ML package | Native Apple Silicon reference | Officially linked by V2 maintainers, but Small only; verify the downloaded package licence and numerical tolerance in the harness. [Source](https://huggingface.co/apple/coreml-depth-anything-v2-small) |
| ONNX Runtime | Project-owned export of pinned V2-Small | Optional cross-platform runtime arm | V2 links an ONNX community implementation, not an official V2 ONNX runner. Treat export parity as a prerequisite and label this arm as project-adapted. [Source](https://github.com/DepthAnything/Depth-Anything-V2#community-support) |
| Our Rust engine | The same pinned V2-Small graph/weights as the reference arm | Product candidate | This is the only arm for which a direct speedup statement is meaningful. |

## Required reproducibility contract

- Pin Git commit, Python/Rust/runtime versions, checkpoint SHA-256, input
  frames, decoded pixel format, input resolution, precision, device, thread
  count, warm-up count, measured iterations, and seed.
- Emit median, p95, frames/s, peak RSS/VRAM, and four stage timings:
  `decode`, `preprocess`, `model`, `postprocess/export`. A separate
  `end_to_end` number includes all four.
- Use fixed frame files for model latency; run a second video end-to-end suite
  for the actual browser use case. Never mix initial model download/compile
  time into steady-state latency.
- Validate each adapted runtime against the official V2 runner on a fixed
  image corpus before timing. The threshold must be recorded per runtime, not
  assumed identical across the OpenCV/Pillow pipeline difference.
- Publish macOS CPU, Apple Silicon acceleration, and Linux/CUDA results as
  separate tables. Hardware names and power mode are mandatory metadata.

## Scope recommendation for the first milestone

1. Official V2-Small PyTorch vs our V2-Small engine on the current macOS host.
2. Local browser video → depth frames → point-cloud viewer, using that engine.
3. Add Core ML and Transformers V2-Small only after parity contracts exist.
4. Add V1, Marigold, GeoWizard, and DepthFM in a Linux/CUDA model-comparison
   report; use their official quality and fast profiles.

Video Depth Anything is deliberately out of scope for this comparison: V2's
official page describes V2 as image-based and says video is used for display;
Video Depth Anything is a later, temporally-consistent model family.
[V2 project page](https://depth-anything-v2.github.io/),
[Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything).
