# Repository split migration

Reader: an engineer migrating a pre-split checkout or adding a backend.
After reading: they can place new code in the correct repository and build a
local engine/kernel pair without source duplication.

| Responsibility | Destination | Stable boundary |
|---|---|---|
| GGUF parsing, weights, model configuration | Vestra Engine | Engine-owned model types |
| Preprocess, letterboxing, depth, confidence, pose, multi-view | Vestra Engine | Public `vestra-engine` API |
| CLI, parity corpus, end-to-end benchmark runner | Vestra Engine | CLI and benchmark documents |
| GEMM, attention, softmax, LayerNorm, RoPE | Vestra Kernels | explicit slices and dimensions |
| convolution, Winograd, resize, transpose/copy | Vestra Kernels | explicit slices and dimensions |
| AVX dispatch, fixed-shape DA3 paths, kernel oracles | Vestra Kernels | CPU feature dispatch |

The historical `da-kernels` crate was removed from the engine workspace. Its
generic operators, ISA dispatch, tests, and microbenchmarks now belong to
Vestra Kernels. Dump-backed operator checks remain engine integration tests
because they require engine-owned parity fixtures.

## Local development

Vestra Engine declares `vestra-kernels` as a versioned crate and applies a
repository-local Cargo patch to the sibling checkout during development. This
is intentionally the only local override. For a released build, remove the
patch and resolve the published/pinned kernel release recorded in the lockfile.

## Known release gap

The public GitHub repository for `hoppworks/vestra-kernels` did not exist when
this split was prepared. The local repositories are committed and buildable;
publishing the kernel repository, pushing its baseline commit, and replacing
the local development patch with that immutable revision are required before a
network-only clone can build.

The committed local pair was fresh-cloned into sibling directories and checked
successfully at engine `0c65739` and kernel `b35a917`. That validates the
documented local-development topology without relying on the original working
trees.
