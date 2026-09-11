# Iteration 100 — fair physical-core affinity check (rejected)

## Question

The Workhorse exposes CPU IDs 0–15 as the first hardware thread of each of
the Ryzen 9 9950X's 16 physical cores, and 16–31 as their SMT siblings.  A
prior Rust-only pinning experiment was negative.  This check pins *both* arms
to CPUs 0–15 to establish whether an identical physical-core policy changes
the fair comparison.

## Protocol

Both commands ran under `taskset -c 0-15`, with the locked DA3 BASE F32 GGUF,
`mountains.jpg`, 504x336 preprocessing, and 16 runtime threads.  Rust used
the accepted configuration; C++ used `da3-cli depth --threads 16`.  Each run
used one warm-up plus ten timed iterations.

| Pair | Rust median (ms) | C++ median (ms) |
|---:|---:|---:|
| 1 | 193.206 | 199.3 |
| 2 | 203.602 | 201.3 |
| 3 | 210.259 | 196.7 |

The C++ reference remains near its unpinned latency while Rust loses the
accepted route's roughly 158–163 ms behavior.  Pinning is therefore not a
fair performance improvement and cannot be used for qualification.

## Decision

Rejected.  Keep the fixed 16-thread budget but let the operating-system
scheduler place those threads for both runtime arms.  The result also rules
out SMT/core-selection policy as the explanation for the remaining gap.
