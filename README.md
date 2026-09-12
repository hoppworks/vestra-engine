# Vestra Engine

Vestra Engine is a native Rust inference runtime for video-to-world
reconstruction. It is the optimized neural engine imported by the Vestra
product repository; scene stitching, TSDF fusion, exports, and the browser
studio live in Vestra itself.

The engine is a structure-preserving Rust port of, and is benchmarked against,
[`localai-org/depth-anything.cpp`](https://github.com/localai-org/depth-anything.cpp)
(C++17/ggml). The Depth Anything 3 model and checkpoints are upstream work by
ByteDance; ggml is upstream work as well. This repository is not a separate
"Depth Anything in Rust" project; it is one of three repositories behind the
single Vestra case study.

## Scope

- GGUF model loading
- calibrated preprocessing
- single-view depth, confidence, and camera pose
- ordered multi-view local/global transformer execution
- optimized CPU execution on AVX-512 x86-64
- opt-in native CUDA parity slices through `vestra-kernels`

The public product name is Vestra Engine. Legacy `da-*` source-directory and
internal dependency aliases are retained temporarily to preserve the verified
code history while public Cargo packages use `vestra-*` names.

## Multi-view status

The ordered multi-view path is implemented and independently qualified:

- local blocks execute independently per view;
- global blocks attend over one view-major flattened sequence;
- reference and source views receive distinct camera-token slots;
- RoPE special-token boundaries repeat correctly for every view;
- the `S=1` ordered multi-view path is bitwise equal to single-view execution;
- a synthetic test proves that a second view affects the first view at global
  attention layers;
- the automatic path performs the preliminary local CLS pass, selects the
  saddle-balanced reference view for eligible windows, runs reference-first,
  and restores the caller's original view order;
- canonical RGB24 C++ oracle comparisons are accepted for `S=2`, `S=3`, and
  `S=12`, including depth, confidence, W2C extrinsics, and intrinsics.

| Window | Worst depth r | Worst depth MAE | Worst W2C MAE | Worst intrinsics error |
|---|---:|---:|---:|---:|
| `S=2` | 0.999999999982 | 0.0000015403 | 0.0000019950 | 0.003965 px |
| `S=3` | 0.999999999985 | 0.0000026587 | 0.0000082050 | 0.004754 px |
| `S=12` | 0.999999999741 | 0.0000211174 | 0.0000199768 | 0.032229 px |

The canonical-input contract, thresholds, provenance, and repeatable commands
are in [the multi-view oracle gate](docs/MULTIVIEW_ORACLE.md). Streaming-window
scheduling, cross-window registration, and fusion belong to the Vestra product
repository rather than this inference engine.

## CUDA status

The `cuda-residual-oracle` feature is a native CUDA integration and parity
surface, not a production speed backend. On an RTX 5080 it has qualified:

- device-side patch lowering and cached patch projection;
- a device-resident transformer tail from the first Q/K-normalized block
  through the final block;
- single-image and ordered multi-view agreement with the CPU F32 path.

Preprocessing, early token preparation and transformer blocks, feature
captures, DPT, and pose still execute on or cross through the CPU. CUDA is
therefore opt-in, is not selected by the product CLI, and carries no current
end-to-end performance claim. The executable gates live in
[`cuda_residual_parity.rs`](crates/da-engine/tests/cuda_residual_parity.rs).

## Canonical CPU-F32 baseline

The current portfolio comparison is a fresh 10-trial same-machine study against
the public `localai-org/depth-anything.cpp` revision. It uses an AMD Ryzen 9
9950X, 16 benchmark threads, DA3-BASE F32, and 504×336:

| Runtime | Mean of trial medians | 95% CI |
|---|---:|---:|
| Vestra Engine | 158.126 ms | [156.196, 160.056] ms |
| C++ / ggml | 197.036 ms | [196.325, 197.748] ms |

That is 19.75% lower latency, or 24.61% higher throughput; both percentages
describe the same run and must not be added. The confidence intervals do not
overlap. It is a single-image CPU-F32 result (2026-09-11, C++ commit
`739992d`) and must not be presented as proof of multi-view, quantized, GPU, or
complete reconstruction performance.

Scope of that number: the Rust arm ran the qualified workhorse configuration,
meaning the AOCL BLIS bridge plus the documented opt-in kernel switches
(`DA3_STRIP_MLP`, `DA3_RN1_F2_OC32`, `DA3_STRIP_MLP_HOIST_DISPATCH`,
`DA3_KERNELS_ENABLE_OUT1_F2_128X64`, `DA3_FUSED_FINAL_RESIZE_ROW_RING`) and an
OpenMP thread policy. It is not the default `cargo run` build. The trade-off is
memory: mean per-process peak RSS was 1038.1 MiB for Vestra Engine against
619.1 MiB for C++/ggml. The Vestra product repository pins an older engine
revision, so this is an engine measurement, not a property of the product
binary.

Earlier studies on the same workload are historical evidence, not the
headline, and must not be combined with it: 2026-08-30, N=20, same C++ build,
without the extra switches, measured 188.137 ms versus 197.939 ms (4.95% lower
latency; bundle not published); 2026-08-13, N=20, against an older and slower
pinned C++ build, measured 171.141 ms versus 238.789 ms (28.3% lower latency).
Raw trials, executable hashes, fidelity results, and provenance for the current
number are in the
[2026-09-11 remeasurement bundle](docs/benchmarks/2026-08-workhorse/2026-09-11-current-f32-rerun/RESULTS.md).

## Build and test

```bash
cargo test --locked --workspace
cargo run --release --locked -p vestra-cli --bin vestra-engine -- infer \
  --model /path/to/model.gguf \
  --image /path/to/image.jpg \
  --out-depth /tmp/depth.pfm \
  --out-pose /tmp/pose.json
```

The qualified workhorse benchmark build used `-C target-cpu=znver5` on the
Ryzen 9 9950X. Benchmark commands and accepted environment switches are
documented in the benchmark bundle rather than hidden in this README.

The [optimization carry-over audit](docs/audits/2026-08-30-optimization-carryover-audit.md)
explains why the old optimization branch was not merged wholesale. The
[whole-repository optimization audit](docs/audits/2026-08-30-code-optimization.md)
records remaining hypotheses separately from qualified speedups.

## Repository boundary

- `vestra-engine`: model semantics and execution
- `vestra-kernels`: qualified CPU/CUDA kernels
- `vestra`: video reconstruction, scene format, local service, CLI, and studio

## Reproducible dependency

The checked-in Cargo configuration resolves `vestra-kernels` from an exact
Git revision, and `Cargo.lock` records the complete CLI/benchmark dependency
graph. A clone needs no sibling checkout once that remote revision is readable,
while every engine revision still names the kernel source it qualified.
Contributors may use an uncommitted local Cargo source override while changing
both repositories; that override must never replace the committed revision.
See the
[repository split migration](docs/REPOSITORY_SPLIT.md) and
[ADR-001](docs/ADR-001-engine-kernel-repository-split.md) for the boundary and
release requirement. The exact local and benchmark identities are recorded in
[the reproducibility record](docs/REPRODUCIBILITY.md).

## Licensing and model weights

Vestra Engine's Daniel-owned original contributions are licensed under
Apache-2.0. Direct ports and adaptations remain subject to their upstream
notices: in particular, the `depth-anything.cpp` and ggml-derived portions
retain their MIT copyright and permission notices. Model weights are not
distributed in this repository. The benchmarked `depth-anything/DA3-BASE`
checkpoint is Apache-2.0; that statement is specific to DA3-BASE and must not
be generalized to DA3-LARGE, DA3-GIANT, or Nested checkpoints, whose model
cards publish different terms. See [NOTICE](NOTICE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for pinned sources, affected
modules, and complete license copies.

Part of Daniel Hopp's portfolio: https://daniel.hoppworks.de/work/vestra/ ·
https://www.linkedin.com/in/hoppworks
