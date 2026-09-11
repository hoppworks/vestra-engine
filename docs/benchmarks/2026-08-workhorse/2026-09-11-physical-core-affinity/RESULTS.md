# Fixed physical-core affinity — rejected

## Hypothesis

The Ryzen 9 9950X exposes 16 physical cores and 32 SMT logical CPUs. Pinning
the 16-thread DA3 workload to CPU IDs 0–15 (one logical thread per physical
core) might reduce scheduler migration and SMT contention.

## Method

The accepted Rust binary, model, `mountains.jpg` input, 504×336 preprocessing,
16 Rayon/BLIS threads and one-warm-up/ten-timed-iteration contract were held
constant. Only `taskset -c 0-15` differed. Three interleaved pairs on the
otherwise idle Workhorse measured:

| Pair | Scheduler-managed (ms) | `taskset -c 0-15` (ms) |
| --- | ---: | ---: |
| 1 | 159.484 | 196.411 |
| 2 | 158.888 | 201.525 |
| 3 | 160.814 | 209.925 |
| Mean | 159.729 | 202.620 |

## Decision

Reject fixed affinity. It is 42.891 ms slower, so no C++ follow-up is
justified. The fair benchmark continues to use the normal Linux scheduler for
both arms; no affinity mask is part of the published contract.
