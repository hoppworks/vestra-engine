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

Vestra Engine declares `vestra-kernels` at exact Git revision
`1ad85305de14ea76ddd878af6dac80f19bdf2bc3`, also recorded in `Cargo.lock`.
Contributors changing both repositories may use an uncommitted local Cargo
source override to a sibling checkout. This is intentionally the only local
override; it must be removed before qualification so release builds resolve the
committed revision with `--locked`.

## Immutable release boundary

The engine no longer depends on a sibling path or a committed Cargo patch. A
network-only public clone requires the pinned `hoppworks/vestra-kernels`
revision to remain publicly readable; repository visibility is an operational
release setting, not a source-code fallback.

The committed local pair was fresh-cloned into sibling directories and checked
successfully at engine `0c65739` and kernel `b35a917`. That validates the
documented local-development topology without relying on the original working
trees.
