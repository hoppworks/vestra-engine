# Profile-guided optimization build — rejected

## Hypothesis

An LLVM PGO build trained on the four locked DA3 corpus images at the fixed
504×336 shape could improve cross-crate inlining, code layout and branch
prediction without changing any inference work.

## Build procedure

The candidate used the same Rust 1.93.0 toolchain as the control. An
instrumented release was compiled with `-C profile-generate`, then ran one
warm-up plus five inferences on each of canyon, desk, mountains and street.
The resulting 13 MiB merged LLVM profile was consumed by a separate release
build using `-C profile-use`. Model loading and the profile-training runs were
outside the benchmark boundary.

## Admission result

The first interleaved locked-workload pair on the Ryzen 9 9950X was already
decisive:

| Arm | Trial median (ms) |
| --- | ---: |
| Thin-LTO control | 162.439 |
| PGO candidate | 245.442 |

A second candidate process was also slow at 236.489 ms. The queued smoke
series was terminated immediately rather than consuming further Workhorse
time. This is not a numerical or workload change; it is an invalid performance
candidate, so no PFM or full randomized study is warranted.

## Decision

Reject PGO for the current Rust/LLVM toolchain and profile generation mode.
Keep the ordinary deterministic Thin-LTO release profile. Training artifacts
are retained under `/var/tmp/vestra-pgo-prof-20260911/` and
`/var/tmp/vestra-pgo-20260911.profdata` for investigation, but are not part of
the product build.
