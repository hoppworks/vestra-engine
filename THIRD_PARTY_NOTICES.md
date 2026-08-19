# Third-party notices

Vestra Engine's original source is available under the repository's MIT
license. The model semantics and compatibility surfaces below derive from
third-party projects and remain subject to their respective licenses.

## Depth Anything 3

- Project: [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3)
- Source revision used for parity and benchmarks:
  `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`
- Copyright: `Copyright 2025 The Depth Anything 3 Team`
- License: Apache License 2.0; a complete copy is in
  [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt).

Depth Anything 3 supplies the model architecture, tensor semantics, and the
official DA3-BASE checkpoint against which this Rust implementation is
qualified. Vestra Engine is a modified, independently optimized Rust
implementation; it is not the official Depth Anything 3 runtime.

Model weights are not distributed in this repository. The specific
[`depth-anything/DA3-BASE`](https://huggingface.co/depth-anything/DA3-BASE)
checkpoint used by the recorded studies is Apache-2.0. This notice does not
claim that every DA3 checkpoint has the same license: the official model table
publishes different terms for DA3-LARGE, DA3-GIANT, and Nested variants.
Converting DA3-BASE weights to GGUF does not replace the checkpoint's original
terms.

## depth-anything.cpp

- Project: [localai-org/depth-anything.cpp](https://github.com/localai-org/depth-anything.cpp)
- Pinned merged PR #2 revision:
  `2028b47ac75a8659c6a9aa617baf09be193eb55f`
- Copyright: `Copyright (c) 2026 the depth-anything.cpp authors`
- License: MIT; the required notice and permission text are in
  [`LICENSES/MIT-depth-anything.cpp.txt`](LICENSES/MIT-depth-anything.cpp.txt).

The model-configuration keys, preprocessing contract, patch/position token
semantics, ViT execution order, DPT head, UV embedding, camera-pose decoder,
and ordered multi-view schedule in `crates/da-engine` contain direct Rust ports
or structure-preserving translations of the pinned C++ implementation. Vestra
changes data ownership, scheduling, kernels, caching, error handling, and Rust
APIs. Source-level comments identify the relevant C++ files at the ported
sections.

Benchmark scripts and historical patches also exercise this pinned C++ runtime
as the same-workload reference. Vestra does not claim authorship of the C++
runtime or its original benchmark interface.

## ggml

- Project: [ggml-org/ggml](https://github.com/ggml-org/ggml)
- Reference revision used by the recorded C++ studies:
  `eced84c86f8b012c752c016f7fe789adea168e1e`
- Copyright: `Copyright (c) 2023-2026 The ggml authors`
- License: MIT; the required notice and permission text are in
  [`LICENSES/MIT-ggml.txt`](LICENSES/MIT-ggml.txt).

`crates/da-gguf` implements compatibility with the public GGUF container and
GGML tensor-type/Q8_0 wire layouts. The benchmark corpus also records ggml as
the backend of the pinned C++ reference. Vestra Engine neither vendors nor
links the ggml source tree in its Rust runtime, but includes this notice because
the compatibility code and published comparison depend on those definitions.

The project names above are used only to identify provenance. Their licenses
do not grant trademark rights or imply endorsement of Vestra Engine.
