# Profile-guided Rust build experiment — 2026-09-10

## Hypothesis

The CPU path is already built with `opt-level=3`, ThinLTO, one codegen unit,
`panic=abort`, and `target-cpu=znver5`. Profile-guided optimization (PGO) was
tested as a compiler-only experiment: it could improve inlining, block layout,
and hot/cold branch placement without changing model data, arithmetic, or the
benchmark workload.

## Method

The current source revision was instrumented with Rust 1.93.0 and exercised
on the locked four-image corpus (`canyon`, `desk`, `mountains`, `street`) at
504 x 336. Each image used the standard one warm-up plus three timed passes;
the instrumentation output was merged using the matching Rust-1.93 LLVM
`llvm-profdata` tool. A second release binary was built with the resulting
13-MiB profile. The control was built from the same source with identical
BLIS link flags, `target-cpu=znver5`, and runtime candidate flags, without
`profile-use`.

## Result

Five interleaved trials on `desk.jpg` used one warm-up plus ten timed
iterations and varied only the compiler profile-use flag:

| Trial | Plain median (ms) | PGO median (ms) |
|---:|---:|---:|
| 1 | 161.130 | 244.286 |
| 2 | 157.575 | 237.651 |
| 3 | 162.086 | 238.257 |
| 4 | 161.951 | 232.744 |
| 5 | 159.042 | 238.954 |
| **Mean** | **160.357** | **238.378** |

PGO is 78.021 ms (48.7%) slower in this controlled smoke. **Decision: reject.**
No PGO build, profile, or result is part of the qualified Vestra benchmark.
