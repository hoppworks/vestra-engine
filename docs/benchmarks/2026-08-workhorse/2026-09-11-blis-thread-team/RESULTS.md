# Iteration 102 — BLIS thread-team width (rejected)

## Hypothesis

Vestra's accepted head route uses BLIS for selected GEMMs while the surrounding
runtime uses a 16-thread Rayon/OpenMP budget.  Explicitly selecting a smaller
BLIS team might avoid an unfavorable nested team shape without exceeding the
locked 16-thread limit.

## Screening and admission measurement

A one-run screen covered `BLIS_NUM_THREADS=1,2,4,8,16`; `4` showed a single
154.591 ms result and was therefore the only setting admitted to alternating
measurement.  Both arms used the same accepted binary, DA3 BASE F32 model,
`mountains.jpg`, 504x336 preprocessing, 16 Rayon/OpenMP threads, one warm-up
and ten timed iterations.

| Pair | Default median (ms) | BLIS 4 median (ms) |
|---:|---:|---:|
| 1 | 156.785 | 159.998 |
| 2 | 163.104 | 161.475 |
| 3 | 159.233 | 161.181 |
| 4 | 156.355 | 156.793 |
| 5 | 159.274 | 160.929 |

Default mean: 158.950 ms. BLIS-4 mean: 160.075 ms. The candidate loses three
pairs and is 1.125 ms (0.71%) slower, so the initial screen was noise.

## Decision

Rejected. No explicit `BLIS_NUM_THREADS` setting belongs in the accepted Rust
environment; the library's normal team selection remains in force.
