# depth-anything-rs

Rust rebuild work for Depth Anything inference, parity, and reproducible
performance measurement.

The repository-level entry point is [the case-study README](../README.md).
For the current architecture, accepted/rejected decisions and the exact
workhorse reproduction route, read the [CPU-F32 status guide](docs/DA3_CPU_F32_STATUS.md).

## Active direction: reproducible runtime case study

The active deliverable is a same-model comparison of the official PyTorch
DA3-BASE runtime, the optimized C++/ggml port, and this Rust reimplementation.
It reports performance only together with numerical fidelity and raw evidence.

Video, point-cloud and floorplan work remains preserved as future exploration;
it is not part of the benchmark claim or the primary portfolio story.

## Benchmark direction

The canonical host is a Ryzen 9 9950X plus RTX 5080. CPU and CUDA results are
separate. Direct runtime claims use identical DA3-BASE F32 weights; C++ Q8_0
and Q4_K are separately labelled compression/quality trade-offs.

- [Locked protocol](docs/benchmarks/2026-08-workhorse/PROTOCOL.md)
- [Original baseline study](docs/benchmarks/2026-08-workhorse/RESULTS.md)
- [Raw timing data](docs/benchmarks/2026-08-workhorse/raw-results.json)
- [Fidelity data](docs/benchmarks/2026-08-workhorse/quality-results.json)
- [Sources and attribution](docs/benchmarks/2026-08-workhorse/SOURCES.md)

The existing `da bench` command provides the current Rust timing primitive:

```bash
cargo run -p da-cli -- bench --model /path/to/model.gguf --image /path/to/image.png
```

The original baseline report is intentionally retained as historical evidence;
it is not the current performance headline. The current qualified CPU-F32
result and the corresponding workhorse artifact are described in the status
guide.

## Future work

The DA3/COLMAP 3D-floorplan experiment and point-cloud browser are preserved but
must not be presented as completed product functionality or included in this
runtime leaderboard.

Parity is gated against the C++ repo's reference dumps in `../dumps/`
(read-only).
