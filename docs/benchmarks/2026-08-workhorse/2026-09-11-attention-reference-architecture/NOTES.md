# Attention reference architecture: C++/ggml versus Vestra

## Purpose

This is a diagnostic design note, not a parity-valid performance comparison.
It records the C++ reference's attention topology and a deliberately incorrect
skip-mode measurement to bound attention's contribution before another kernel
implementation is attempted.

## C++ reference findings

The current reference builds `ggml_flash_attn_ext` with F32 K/V and F32
precision in `src/attention.cpp`. In its CPU implementation:

- Query tiles are 64 rows and key/value tiles are 64 columns.
- Each dynamic CPU task can cover a 64-row tile; work is flattened across all
  heads and query tiles by ggml's thread pool.
- The scratch format is `Q[64,64]`, `KQ[64,64]`, `V[64,64]`, and
  `VKQ[64,64]` per worker.
- Both Q×K^T and score×V use ggml's 4-row × 64-column SIMD microkernel on
  AVX-512 builds.

Source evidence on the Workhorse checkout:

- `src/attention.cpp:65-75`
- `third_party/ggml/src/ggml-cpu/ops.cpp:8585-8855`
- `third_party/ggml/src/ggml-cpu/common.h:9-10`
- `third_party/ggml/src/ggml-cpu/simd-gemm.h:10-108`

## Diagnostic timing

The same C++ binary, DA3 BASE F32 GGUF, `mountains.jpg`, 504×336 and 16
threads was run with one warm-up and ten timed iterations.

| C++ mode | Median inference | Interpretation |
|---|---:|---|
| normal Flash attention | 199.4 ms | Valid reference output |
| `DA_ATTN=skip` | 180.6 ms | Intentionally wrong output; diagnostic only |

The difference is approximately 18.8 ms. It is **not** transferable as a
Rust speedup estimate: the runtimes have different non-attention costs and
the skip mode removes rather than accelerates computation.

## Consequence

Vestra's current 8×32 flash path should not be replaced with materialized
BLIS attention (measured +35.285 ms). Its prior 64-row port was also slower.
Any further attention work must demonstrate why it avoids the Rust 64-row
route's overhead while preserving ggml's flat dynamic scheduling and
per-worker scratch locality. A direct, profiled port of the ggml task layout
is a higher-quality next experiment than another query-tile size sweep.
