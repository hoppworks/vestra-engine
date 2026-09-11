# Smaller BLIS thread teams — rejected

## Hypothesis

Vestra alternates Rayon transformer/Winograd phases with serial 16-thread
BLIS head projections. Reducing only the BLIS OpenMP team could lower
synchronization and cross-CCD costs while keeping the process-level maximum
at the locked 16 benchmark threads.

## Method

The accepted Rust binary, BASE F32 GGUF, `mountains.jpg`, 504×336
preprocessing, `RAYON_NUM_THREADS=16`, one warm-up and ten timed iterations
were fixed. The only varying setting was `OMP_NUM_THREADS` for BLIS/OpenMP.
Five interleaved pairs ran per candidate on the Ryzen 9 9950X.

### Eight threads

| Pair | OMP 16 (ms) | OMP 8 (ms) |
| --- | ---: | ---: |
| 1 | 164.361 | 163.996 |
| 2 | 155.114 | 160.145 |
| 3 | 156.677 | 161.140 |
| 4 | 155.869 | 154.048 |
| 5 | 167.743 | 154.488 |
| Mean | 159.953 | 158.763 |

OMP 8 wins 3/5 pairs but improves the mean by only 1.190 ms; the final pair
dominates the mean and does not establish a stable improvement.

### Twelve threads

| Pair | OMP 16 (ms) | OMP 12 (ms) |
| --- | ---: | ---: |
| 1 | 155.723 | 161.596 |
| 2 | 160.577 | 156.450 |
| 3 | 158.513 | 160.887 |
| 4 | 156.099 | 162.017 |
| 5 | 161.411 | 156.044 |
| Mean | 158.465 | 159.399 |

OMP 12 loses by 0.934 ms on average and wins 2/5 pairs.

## Decision

Keep `OMP_NUM_THREADS=16`. Neither reduced team reaches the 3 ms / 8-of-10
admission rule, so no parity or full randomized study is justified.
