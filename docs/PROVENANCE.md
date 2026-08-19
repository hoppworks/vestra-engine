# Vestra Engine provenance

Vestra Engine was extracted from the optimized Rust implementation developed
inside the versioned
[`hoppworks/depth-anything.cpp-master`](https://github.com/hoppworks/depth-anything.cpp-master)
repository.

## Source snapshot

- Source repository: `https://github.com/hoppworks/depth-anything.cpp-master`
- Source commit: `b326a9c2f6f7c80c85e72bda050ceb41c83cac17`
- Extracted history commit: `14452b6f70fe16dbfa5e2564a8b8a17170e934d0`
- Source subtree: `depth-anything-rs/`
- Source tracked-diff SHA-256: `a8b6bcac930cbd625d059a9041530e0e02b3a88d0abb9c6ef0f62a187a88d19d`
- Source status-list SHA-256: `b0330a8f8ac4ae95586c7bd78f701d90992b8ecd686d2e6e5a89f202633dea36`
- Snapshot date: 2026-08-13

The extraction preserved the subtree's Git history. The source working tree
contained verified but uncommitted optimization work, so the new repository
first records that exact source snapshot before applying the Vestra naming and
multi-view changes.

The authoritative CPU-F32 benchmark artifacts are retained under
`docs/benchmarks/2026-08-workhorse/`. They identify the Ryzen 9 9950X workload,
binary hashes, model and image hashes, raw trials, and confidence intervals.

## Dependency boundary

Vestra Engine owns model semantics, model loading, preprocessing, tensor
scheduling, single-view and multi-view inference, and backend selection.
Fixed-shape CPU and CUDA kernels live in the separately versioned
`vestra-kernels` repository.

## Third-party source boundary

The model topology originates in ByteDance Seed's Depth Anything 3. Several
model-semantic sections are direct Rust ports of the pinned
`localai-org/depth-anything.cpp` implementation, while GGUF/Q8_0 compatibility
follows the ggml format. The exact revisions, affected modules, copyright
notices, and licenses are recorded in
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md). Model weights are not
part of this source repository.
