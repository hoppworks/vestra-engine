# Split reproducibility record

This record distinguishes the local repository split from the earlier
performance study. The split is mechanical; it does not replace the dated
benchmark provenance.

| Item | Recorded value |
|---|---|
| Engine snapshot before split | `1f016ac` |
| Engine split commit | `2d0820d` |
| Kernel snapshot before rename | `b70af69` |
| Kernel split/rename commit | `022b85a` |
| Rust 2024 AVX-512 compatibility fix | `b35a917` |
| Pinned kernel source revision | `1ad85305de14ea76ddd878af6dac80f19bdf2bc3` |
| Local validation toolchain | rustc 1.93.0, Cargo 1.93.0 |
| Target benchmark toolchain | rustc 1.97.1, LLVM 22.1.6 |
| Release profile | opt-level 3, Thin LTO, one codegen unit, abort panic |
| Target CPU policy | `-C target-cpu=znver5` on AMD Ryzen 9 9950X |
| Model/input identity | recorded in the dated CPU-F32 protocol and raw results |

The versioned engine dependency is `vestra-kernels` 0.1.0 at the immutable Git
revision above. `Cargo.lock` is committed because this workspace ships CLI and
benchmark binaries; CI resolves it with `--locked`. An uncommitted local Cargo
source override is permitted while changing the engine and kernels together,
but it must never replace the repository pin or qualify a release result.
Never substitute a benchmark result from a different model, input, precision,
target CPU, or thread budget.
