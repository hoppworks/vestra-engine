# Iteration 55 provenance

This artifact is the durable copy of the qualified 2026-08-13 CPU-F32 study.

- Raw result SHA-256:
  `5736716c67981b3080ee95ab3040245ba5ece5ff347478337c792f051ee5f85c`
- Human-readable summary SHA-256:
  `a4d726830551dd41ceff4b658f00e05947b9acbe172594cc69446e7a3d2059dc`
- Model SHA-256:
  `1b13b166e8a8b4f2c862f42d36edb2f9aab995a18cc527a52b9f160b99c6b8da`
- Timed image SHA-256:
  `936d60f43c51fe99156563a0d3c5da69cf84a39cbde5e443bea7662500b8c969`
- Host: AMD Ryzen 9 9950X, 16 physical cores / 32 logical CPUs, 96 GiB memory;
  NVIDIA GeForce RTX 5080 with 16,303 MiB VRAM; Linux 7.1.5; rustc 1.97.1.
- Protocol: 16 threads; ten randomized process trials per arm; one warm-up and
  ten timed inferences per trial; three-second cooldown.

The measurement runner preserved commands, model and input hashes, host data,
trial order and all 200 raw timing samples. It did **not** resolve C++, ggml or
Rust Git commits because the workhorse benchmark tree was a filesystem snapshot
rather than a Git checkout. The result is still useful evidence for this exact
snapshot, but a future public rerun must add binary hashes and source-tree or
commit hashes before replacing this artifact.
