# Code optimization audit — 2026-08-30

## Purpose

This audit separates three different questions that must not be mixed in a
portfolio claim:

1. Is the single-image CPU-F32 benchmark measuring the current binaries?
2. Did the repository split lose an already-qualified optimization?
3. Which new opportunities still require their own hypothesis and benchmark?

The intended reader is a future maintainer preparing a benchmark release. After
reading this document, they should know which work is required for trustworthy
evidence and which findings are only unqualified opportunities.

## Executive summary

The highest-impact defect was in the evidence pipeline, not the inference
kernel. The benchmark helpers still referenced the pre-split package and binary
names. A stale `target/release/da` could therefore be measured while the current
binary is `target/release/vestra-engine`. The scientific and profiling runners
also assumed that the Rust workspace lived inside the C++ checkout.

The evidence path was corrected first. It now resolves the engine repository
from the runner location, accepts an explicit C++ root, records the current
Rust and C++ binary hashes plus Rust build switches, rejects missing artifacts,
bounds each arm with a timeout, and publishes result files atomically.

The old optimization branch does not contain a single patch that should be
merged wholesale. Its important mechanisms are either present or superseded by
the current direct execution, external-kernel, BLIS, Winograd, packed-attention,
and 16-thread paths. The carry-over mapping is recorded in the dated benchmark
bundle.

## Findings by priority

### Evidence integrity — fixed in this audit

| Severity | Location | Finding | Resolution |
|---|---|---|---|
| High | `scripts/compare_e2e.sh` | Pre-split package `da-cli` and binary `target/release/da` could select stale code. | Build `vestra-cli`/`vestra-engine` with `--locked`; require an explicit C++ checkout. |
| High | `scripts/run_scientific_benchmark.py` | Runner derived the Rust checkout from the C++ root and used the legacy binary. | Resolve the engine root from the script and allow an explicit binary override. |
| High | `scripts/run_hardware_profile.py` | Profile provenance used the removed in-tree kernel path and legacy binary. | Make engine, kernel, C++ root, and Rust binary explicit artifacts. |
| Medium | Scientific runner | A hung arm could block the complete randomized study. | Add a configurable per-arm timeout and terminate the complete process group. |
| Medium | Scientific runner | Interrupted writes could leave truncated or mutually inconsistent evidence. | Flush, fsync, and atomically replace partial and final artifacts. |

### Single-image CPU opportunities — require isolated qualification

| Severity | Location | Finding | Required experiment |
|---|---|---|---|
| High | Transformer direct path | Q/K/V, attention, LayerNorm, MLP, and residual buffers are allocated repeatedly. | Introduce one typed `VitWorkspace`; compare allocation counts and whole-model latency before and after. |
| High | Transformer RoPE path | Grid positions are rebuilt and converted for every layer. | Cache typed local/global positions per inference geometry and verify all parity gates. |
| High | DPT handoff | Feature captures are cloned before the head, and the fused token-normalization/CHW path is opt-in. | Remove borrowed-data copies and run an alternating A/B study before changing the default. |
| Medium | Preprocessing | Resize plans and intermediate image buffers are rebuilt for repeated fixed-resolution frames. | Cache the plan and reusable buffers; measure phase timing and end-to-end impact. |
| Medium | Weight lookup | Hot operations repeatedly format and hash string weight names. | Resolve typed layer-weight handles at load time, then benchmark the full model. |

### Multi-view and CUDA opportunities — outside the headline benchmark

| Severity | Location | Finding | Required experiment |
|---|---|---|---|
| High | Multi-view backbone | Independent local blocks, preprocessing, DPT, and pose execute serially by view. | Add one explicit thread-budget policy before introducing view-level parallelism. |
| High | Multi-view global blocks | Every global layer flattens all views into a new allocation and copies them back. | Keep states contiguous or reuse a persistent flatten workspace. |
| High | CUDA local/global blocks | Per-view launches, device copies, host-zero buffers, and repeated uploads serialize work. | Add persistent device scratch and batched execution; qualify as a separate CUDA claim. |
| Medium | Reference-view selection | Pairwise similarity is `O(V²·E)`. | Replace it with a summed normalized embedding in `O(V·E)`. |
| Medium | Positional/UV caches | Resolution-keyed caches have no byte or entry bound. | Add a small measured LRU before using the engine in a long-lived service. |

### Robustness and maintenance — not latency claims

| Severity | Location | Finding | Recommended action |
|---|---|---|---|
| High | GGUF reader | File-controlled counts, alignment, dimensions, offsets, and byte lengths are insufficiently bounded. | Centralize checked range arithmetic and reject malformed models before allocation. |
| High | Engine model contract | Missing or shape-incompatible weights can reach panic paths in an aborting release build. | Validate the required typed weight set during model load. |
| High | GGUF load | Rank-2 tensors are materialized and then transposed into another allocation. | Decode or map directly into the final representation; this is a cold-load optimization. |
| High | CLI dry-run | A retired scan path copies the complete source video even in dry-run mode. | Remove or relocate the retired floorplan surface rather than optimize dead code. |
| High | Dependency graph | Default `image` and `faer` features compile unused formats and support crates. | Narrow features in a dedicated dependency-only change and verify the lockfile. |
| Medium | CUDA fallback | Runtime errors and unsupported routes share the same fallback signal. | Distinguish `Ok(None)` from an execution error; benchmark mode must fail closed. |
| Medium | Dead surface | Retired capture, scan, finish, floorplan, and speculative quantization APIs remain visible. | Move product code to the owning repository or remove it with history as recovery. |

## Decision rule

No item in the opportunity tables is an accepted optimization merely because
it looks plausible or wins a microbenchmark. Promotion requires one isolated
change, a correctness oracle, alternating same-binary smoke runs, the four-image
F32 fidelity gate when numerical execution changes, and a randomized full
study on the named workhorse.
