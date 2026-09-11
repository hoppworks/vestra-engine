# Flash-attention 32-query superblocks — rejected

## Hypothesis

The optional 32-query Flash scheduler packs K once per head and processes four
QT8 subtiles in one task. It could reduce nested Rayon scheduling while
retaining the established online-softmax and FMA order for every query.

## Method

The accepted Rust binary, BASE F32 GGUF, `mountains.jpg`, 504×336
preprocessing, 16 threads, one warm-up and ten timed iterations were held
constant. The only candidate flag was `DA3_KERNELS_FLASH_SUPERBLOCK32=1`.
Five interleaved pairs ran on the Ryzen 9 9950X.

| Pair | QT8 production path (ms) | 32-query candidate (ms) |
| --- | ---: | ---: |
| 1 | 163.818 | 176.155 |
| 2 | 159.900 | 176.377 |
| 3 | 157.622 | 169.535 |
| 4 | 155.983 | 172.905 |
| 5 | 160.895 | 171.010 |
| Mean | 159.644 | 173.196 |

The superblock route loses every pair and is 13.552 ms slower.

## Decision

Reject the scheduler. The existing QT8 nested task structure remains the
qualified attention path; no parity or full randomized promotion is warranted.
