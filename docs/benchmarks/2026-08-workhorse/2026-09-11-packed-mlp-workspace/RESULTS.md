# Packed MLP workspace reuse — rejected

## Hypothesis

The accepted `DA3_STRIP_MLP=1` path allocates a fresh 865×768 normalized
activation and a fresh 865×768 FC2 output for each of DA3 BASE's twelve
transformer blocks. Reusing the existing `VitWorkspace.norm` and
`VitWorkspace.branch` capacities could remove about 63 MiB of transient
allocation and initialization per inference without changing any arithmetic.

## Isolated candidate

Only the `MlpExecutor` ownership boundary was changed. It wrote normalized
tokens and FC2 output into caller-owned block workspace buffers. The hidden
strip ordering, BLIS/panel kernels, bias/GELU/LayerScale operations, input,
model, thread budget and benchmark procedure remained unchanged.

The candidate was built from Vestra Engine `800a1a4` with the one local
candidate patch. The control was an independently built, unmodified
`800a1a4` binary. Both used Vestra Kernels `d3a9b8b`, the locked
`depth-anything-base-f32.gguf`, `mountains.jpg`, 504×336 preprocessing,
16 threads, one warm-up and ten timed iterations per pair.

## Paired smoke result

Five randomized interleaved arm pairs on the Ryzen 9 9950X produced the
following per-run medians in milliseconds:

| Pair | Control | Candidate |
| --- | ---: | ---: |
| 1 | 163.264 | 161.909 |
| 2 | 161.873 | 160.449 |
| 3 | 161.232 | 157.738 |
| 4 | 157.651 | 162.520 |
| 5 | 166.695 | 162.217 |
| Mean | 162.143 | 160.967 |

The candidate was 1.176 ms lower on this smoke sample, but won only four of
five pairs and did not meet the predeclared admission threshold of at least
3 ms and eight wins in ten interleaved pairs. It was therefore not promoted
to a full ten-trial study or the product path.

## Numerical parity

Candidate output was compared with the locked C++ F32 PFM corpus:

| Image | Pearson r | MAE |
| --- | ---: | ---: |
| canyon | 0.9999936282 | 0.0018124970 |
| desk | 0.9999782565 | 0.0017728067 |
| mountains | 0.9999855792 | 0.0036749614 |
| street | 0.9999721254 | 0.0008209983 |

Every image clears r ≥ 0.9999 and MAE ≤ 0.005. This confirms the candidate
was semantically equivalent, but its benefit was too small and noisy to
justify added ownership complexity.

## Decision

Reverted cleanly. Raw candidate outputs remain at
`/var/tmp/vestra-mlp-workspace-parity-20260911/` on the Workhorse; the
candidate and control build directories are respectively
`/var/tmp/vestra-engine-mlp-workspace-20260911/` and
`/var/tmp/vestra-engine-mlp-baseline-20260911/`.
