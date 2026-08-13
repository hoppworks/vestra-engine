# Repository-split baseline

The split is behavior-preserving. Its fresh 10-trial validation is recorded in
the dated split-validation bundle. The acceptance baseline remains the CPU-F32 DA3-BASE study on an AMD Ryzen 9
9950X with 16 benchmark threads, a 504×336 input, one warm-up, ten measured
iterations per trial, and randomized fresh-process trials.

| Study | Vestra Engine | C++/ggml | Result |
|---|---:|---:|---|
| 20-trial revalidation | 171.141 ms | 238.789 ms | 39.5% higher throughput for Vestra Engine |

The model and input hashes, raw trial values, confidence intervals, compiler
flags, and parity evidence are retained in the dated benchmark bundle. A
post-split smoke benchmark is required before treating the split as a release
candidate. The full 10-trial study must be repeated only on a quiet Workhorse
with the published kernel revision pinned.
