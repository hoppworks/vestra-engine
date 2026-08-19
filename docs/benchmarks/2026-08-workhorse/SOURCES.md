# Sources and attribution

- **Original model and official PyTorch implementation:** ByteDance Seed,
  [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3), pinned
  for this experiment at commit `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`.
  The code is Apache-2.0. The specific
  [`depth-anything/DA3-BASE`](https://huggingface.co/depth-anything/DA3-BASE)
  checkpoint used here is also Apache-2.0. Other DA3 checkpoints publish
  different terms, including CC BY-NC 4.0, and are outside this claim.
- **Optimized reference runtime and inherited benchmark:** LocalAI contributors,
  [depth-anything.cpp](https://github.com/localai-org/depth-anything.cpp), pinned
  at commit `2028b47ac75a8659c6a9aa617baf09be193eb55f`, MIT license.
- **Converted GGUF weights:**
  [mudler/depth-anything.cpp-gguf](https://huggingface.co/mudler/depth-anything.cpp-gguf),
  using the DA3-BASE weights above. Conversion does not replace the original
  checkpoint terms. Exact file hashes are in `raw-results.json`; weights are
  not distributed by Vestra Engine.

Complete source notices and license copies are in the repository root's
[`THIRD_PARTY_NOTICES.md`](../../../THIRD_PARTY_NOTICES.md).

The work attributable to this portfolio repository is the Rust
reimplementation, parity fixes, fair timing boundary, experimental harness,
target-machine execution, statistical analysis and interpretation. It does not
claim authorship of DA3, the C++ port, the original benchmark concept or model
weights.
