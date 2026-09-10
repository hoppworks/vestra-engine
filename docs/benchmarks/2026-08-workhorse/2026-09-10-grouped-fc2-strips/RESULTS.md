# Grouped FC2 strips — rejected

## Hypothesis

The hidden-strip MLP performs FC2 as 48 consecutive 64-wide input strips.
The existing route reloads and stores each six-row FC2 output tile after every
strip. `DA3_STRIP_MLP_GROUP_FC2=1` grouped four consecutive strips, keeping
the output tile resident while preserving the exact ascending FMA order for
every output value. The predicted benefit was fewer partial-output transfers
and better output-panel weight locality.

## Change

- Kernel candidate: `24b61ea` / `eb7d6c7` in `vestra-kernels`.
- Engine candidate: `8d66569`, with the compatible kernel pin in `7c88d42`.
- New opt-in only: `DA3_STRIP_MLP_GROUP_FC2=1`.
- The control route remained available unchanged, in the same binary.

## Correctness gates

The kernel test compared grouped accumulation against four established serial
FC2 calls using non-zero output, signed values, four and six row tiles. It
passed locally and on the Ryzen 9 9950X AVX-512/FMA target.

The four-image Rust F32 versus locked C++ F32 gate also passed with the
candidate enabled:

| Image | Pearson r | MAE |
|---|---:|---:|
| canyon | 0.9999936282 | 0.0018124970 |
| desk | 0.9999782565 | 0.0017728067 |
| mountains | 0.9999855792 | 0.0036749614 |
| street | 0.9999721254 | 0.0008209983 |

All values meet the locked `r >= 0.9999`, `MAE <= 0.005` rule.

## Admission measurement

Hardware: AMD Ryzen 9 9950X. Both arms used the same released binary, model,
`desk` image resized to 504x336, 16 benchmark threads, one warm-up and ten
timed iterations per arm. The ten pairs alternated start order. Common enabled
candidate flags were `DA3_STRIP_MLP`, `DA3_RN1_F2_OC32`,
`DA3_STRIP_MLP_HOIST_DISPATCH`, `DA3_KERNELS_ENABLE_OUT1_F2_128X64`,
`DA3_FUSED_FINAL_RESIZE_ROW_RING`, `DA3_KERNELS_BLIS_LINEAR`, and
`DA3_HEAD_BLIS_GEMM`. Only `DA3_STRIP_MLP_GROUP_FC2` differed.

| Pair | Control median ms | Grouped median ms |
|---:|---:|---:|
| 1 | 161.304 | 154.324 |
| 2 | 150.890 | 165.983 |
| 3 | 160.607 | 157.510 |
| 4 | 160.761 | 158.001 |
| 5 | 163.137 | 166.209 |
| 6 | 157.071 | 158.666 |
| 7 | 150.487 | 161.343 |
| 8 | 150.629 | 162.675 |
| 9 | 154.744 | 160.387 |
| 10 | 159.093 | 164.979 |
| **Mean** | **156.872** | **161.008** |

Grouped FC2 was 4.135 ms slower and won 3/10 pairs. This fails the
predeclared admission rule of at least 2 ms E2E reduction and at least 8/10
faster pairs. The code is reverted; no public benchmark number changes.

