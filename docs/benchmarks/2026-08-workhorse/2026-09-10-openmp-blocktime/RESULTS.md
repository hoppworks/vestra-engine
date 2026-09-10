# Candidate execution policy: LLVM OpenMP `KMP_BLOCKTIME=0`

## Hypothesis

Vestra alternates 16-thread BLIS/OpenMP GEMMs with Rayon-driven attention and
Winograd work. The linked LLVM OpenMP runtime had its default 200 ms active
worker wait interval. That is longer than a whole established inference, so
BLIS workers can compete with Rayon workers after a GEMM has returned.

The candidate changes no numerical operation: set `KMP_BLOCKTIME=0` before
process start, preserving the locked 16-thread budget for both runtimes.

## Verified runtime settings

`KMP_SETTINGS=1` on Workhorse reports:

```text
KMP_BLOCKTIME=0ms
OMP_DYNAMIC=false
OMP_NUM_THREADS='16'
```

All candidate runs also set:

```text
RAYON_NUM_THREADS=16
OMP_NUM_THREADS=16
OMP_DYNAMIC=FALSE
OMP_MAX_ACTIVE_LEVELS=1
```

## Numerical check

The four-image C++ F32 comparison is unchanged and passes the locked gate:

| Image | Pearson r | MAE |
| --- | ---: | ---: |
| canyon | 0.99999362795732 | 0.0018125212874871735 |
| desk | 0.9999782568224567 | 0.001772925931360581 |
| mountains | 0.9999855775226433 | 0.003675203580509249 |
| street | 0.9999721239891701 | 0.0008210032855921501 |

## Alternating smoke measurements

Each arm used one warm-up and five timed iterations on `canyon`; arms were
alternated at process level. The Workhorse was running unrelated workloads, so
these are diagnostic only and **not** a qualified final result.

| Arm | Median |
| --- | ---: |
| control | 205.406 ms |
| `KMP_BLOCKTIME=0`, `OMP_WAIT_POLICY=PASSIVE` | 194.251 ms |
| control | 207.091 ms |
| `KMP_BLOCKTIME=0`, `OMP_WAIT_POLICY=PASSIVE` | 185.607 ms |
| control | 205.884 ms |
| `KMP_BLOCKTIME=0` only | 190.458 ms |
| control | 216.144 ms |
| `KMP_BLOCKTIME=0` only | 189.060 ms |

The isolated `KMP_BLOCKTIME=0` runs reproduce the direction, so passive wait
policy is not required for the candidate. The candidate is not yet promoted:
it needs an idle-host randomized study under the locked protocol, and the
scientific runner must record the complete OpenMP environment for both arms.
