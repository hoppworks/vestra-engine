# Head-only Flash-attention scheduler — rejected

## Hypothesis

Scheduling one Rayon task per attention head and executing its QT8 query tiles
serially could remove nested task overhead while preserving the existing
packed-K, QT8 and online-softmax arithmetic.

## Method

The accepted binary, BASE F32 model, `mountains.jpg`, 504×336 preprocessing,
16 threads, one warm-up and ten timed iterations were fixed. The only
candidate flag was `DA3_KERNELS_FLASH_HEAD_ONLY=1`. Five interleaved pairs ran
on the Ryzen 9 9950X.

| Pair | QT8 production path (ms) | Head-only candidate (ms) |
| --- | ---: | ---: |
| 1 | 156.444 | 170.954 |
| 2 | 155.862 | 176.295 |
| 3 | 160.768 | 170.986 |
| 4 | 157.978 | 169.752 |
| 5 | 159.592 | 175.191 |
| Mean | 158.129 | 172.636 |

The candidate loses every pair and is 14.507 ms slower.

## Decision

Reject head-only scheduling. The qualified QT8 nested scheduler remains the
only viable existing attention task structure; no parity or full-study
promotion is warranted.
