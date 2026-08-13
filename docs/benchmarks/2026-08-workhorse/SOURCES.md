# Sources and attribution

- **Original model and official PyTorch implementation:** ByteDance Seed,
  [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3), pinned
  for this experiment at commit `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`.
  Code and official checkpoints are Apache-2.0; cite the original paper/project
  when using the model.
- **Optimized reference runtime and inherited benchmark:** LocalAI contributors,
  [depth-anything.cpp](https://github.com/localai-org/depth-anything.cpp), pinned
  at commit `2028b47ac75a8659c6a9aa617baf09be193eb55f`, MIT license.
- **Converted GGUF weights:**
  [mudler/depth-anything.cpp-gguf](https://huggingface.co/mudler/depth-anything.cpp-gguf),
  Apache-2.0. Exact file hashes are in `raw-results.json`.

The work attributable to this portfolio repository is the Rust
reimplementation, parity fixes, fair timing boundary, experimental harness,
target-machine execution, statistical analysis and interpretation. It does not
claim authorship of DA3, the C++ port, the original benchmark concept or model
weights.
