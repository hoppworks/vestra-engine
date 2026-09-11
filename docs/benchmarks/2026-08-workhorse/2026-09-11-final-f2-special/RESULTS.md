# Final 64→32 F(2) microkernel — rejected

## Hypothesis

The final fused head convolution has a fixed 64-input/32-output F(2)
Winograd product. Its optional static-bounds AVX-512 microkernel might remove
generic loop and bounds overhead while preserving the same transformed layout
and FMA order.

## Method

The accepted Rust binary, BASE F32 GGUF, `mountains.jpg`, 504×336
preprocessing, 16 threads, one warm-up and ten timed iterations were held
constant. The only differing flag was
`DA3_KERNELS_ENABLE_FINAL_F2_TILES4_SPECIAL=1`. Five interleaved pairs on the
Ryzen 9 9950X measured:

| Pair | Production F(2) (ms) | Static microkernel (ms) |
| --- | ---: | ---: |
| 1 | 154.354 | 161.106 |
| 2 | 156.320 | 156.474 |
| 3 | 162.141 | 163.714 |
| 4 | 158.169 | 162.274 |
| 5 | 161.146 | 162.399 |
| Mean | 158.426 | 161.193 |

The candidate wins no matched pair and is 2.767 ms slower on average.

## Decision

Keep the generic final F(2) product path. The optional specialized kernel
remains disabled for reproducibility only; it receives neither a PFM gate nor
a full randomized benchmark.
