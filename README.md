# Vestra Engine

Vestra Engine is a native Rust inference runtime for video-to-world
reconstruction. It is the optimized neural engine imported by the Vestra
product repository; scene stitching, TSDF fusion, exports, and the browser
studio live in Vestra itself.

## Scope

- GGUF model loading
- calibrated preprocessing
- single-view depth, confidence, and camera pose
- ordered multi-view local/global transformer execution
- optimized CPU execution on AVX-512 x86-64
- a future native CUDA backend through `vestra-kernels`

The public product name is Vestra Engine. Legacy `da-*` source-directory and
internal dependency aliases are retained temporarily to preserve the verified
code history while public Cargo packages use `vestra-*` names.

## Multi-view status

The first parity tracer bullet is implemented:

- local blocks execute independently per view;
- global blocks attend over one view-major flattened sequence;
- reference and source views receive distinct camera-token slots;
- RoPE special-token boundaries repeat correctly for every view;
- the `S=1` ordered multi-view path is bitwise equal to single-view execution;
- a synthetic test proves that a second view affects the first view at global
  attention layers.

Saddle-balanced reference selection for `S>=3`, real-model C++ parity for
`S=2,3,12`, CUDA, and streaming-window orchestration remain open work.

## CPU-F32 baseline

The current durable 20-trial same-machine study on an AMD Ryzen 9 9950X,
16 benchmark threads, DA3-BASE F32, and 504×336 measured:

| Runtime | Mean of trial medians | 95% CI |
|---|---:|---:|
| Vestra Engine | 171.141 ms | [168.042, 174.241] ms |
| C++ / ggml | 238.789 ms | [237.406, 240.172] ms |

That is 28.3% lower latency, or 39.5% higher throughput. It is a single-image
CPU-F32 result and must not be presented as proof of multi-view, quantized, GPU,
or complete reconstruction performance. Raw trials and provenance live under
`docs/benchmarks/2026-08-workhorse/`.

## Build and test

```bash
cargo test --workspace
cargo run -p vestra-cli -- infer \
  --model /path/to/model.gguf \
  --image /path/to/image.jpg \
  --output /tmp/depth.pfm
```

Target-hardware release builds use `-C target-cpu=znver5` on the Ryzen 9
9950X. Benchmark commands and accepted environment switches are documented in
the benchmark bundle rather than hidden in this README.

## Repository boundary

- `vestra-engine`: model semantics and execution
- `vestra-kernels`: qualified CPU/CUDA kernels
- `vestra`: video reconstruction, scene format, local service, CLI, and studio

## Development dependency

The checked-in Cargo configuration uses a sibling `vestra-kernels` checkout as
a development patch while the package is prepared for publication. This is not
a source copy: it is the sole kernel implementation. See the
[repository split migration](docs/REPOSITORY_SPLIT.md) and
[ADR-001](docs/ADR-001-engine-kernel-repository-split.md) for the boundary and
release requirement.
