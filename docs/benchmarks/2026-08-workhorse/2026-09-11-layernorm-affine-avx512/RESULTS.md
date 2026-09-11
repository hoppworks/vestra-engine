# LayerNorm affine AVX-512 suffix — rejected

## Hypothesis

DA3-BASE executes LayerNorm in every transformer block.  The reduction is
numerically sensitive and deliberately remains scalar, but the following
per-channel affine suffix (`normalized * gamma + beta`) is independent for
each element.  Vectorizing only that suffix with AVX-512 could reduce scalar
epilogue work without changing the reduction order or numerical contract.

## Isolated change

The candidate is Vestra Kernels commit `23178422b1aea42802e1b1a3673a6cc5c52b225e`
(`feat: vectorize layernorm affine suffix`).  Engine source is exactly
`107ddf3c554baa390e5ae1937cd3ef11d3c3825d` in both arms.  The control resolves
kernel commit `d3a9b8b5ad0cb590721eda5d88f05e512a5dfa5d`; the candidate resolves
the same manifest revision to the candidate Git object only.  Both binaries
were rebuilt on the Ryzen 9 9950X workhorse with the identical Zen 5, BLIS and
16-thread environment.

The kernel oracle `layernorm_simd_affine_keeps_scalar_reduction_bits` passed
locally, and the target candidate built successfully with all 72
`vestra-engine` library tests passing.  This candidate was not promoted to the
four-image C++-F32 gate because it did not pass the pre-registered timing
admission rule.

## Alternating target-hardware smoke

The model, `mountains.jpg` input, 504x336 resize, timed boundary, one warm-up,
ten timed iterations, and all accepted runtime flags were identical.  Ten
control/candidate process pairs were alternated; each number is the process
median.  `candidate - control < 0` is a candidate win.

| Pair | Control (ms) | Candidate (ms) | Delta (ms) |
|---:|---:|---:|---:|
| 1 | 162.399 | 155.764 | -6.635 |
| 2 | 158.860 | 156.604 | -2.256 |
| 3 | 151.725 | 156.143 | 4.418 |
| 4 | 161.147 | 161.650 | 0.503 |
| 5 | 158.394 | 161.082 | 2.688 |
| 6 | 158.407 | 162.046 | 3.639 |
| 7 | 155.884 | 160.126 | 4.242 |
| 8 | 161.337 | 152.463 | -8.874 |
| 9 | 161.609 | 159.507 | -2.102 |
| 10 | 156.336 | 153.632 | -2.704 |

Mean delta: **-0.708 ms**; candidate wins: **5/10**.

## Decision

Reject.  The candidate's mean is directionally lower, but it wins only half of
the matched pairs and is far below the pre-registered admission threshold of
at least 2.0 ms reduction and 8/10 wins.  The large pair variance means this
is not evidence of a robust end-to-end improvement.  Keep the Engine pinned to
`d3a9b8b`; do not claim a new benchmark result, run four-image C++ parity, or
start a randomized full study for this variant.

The raw pair table is retained in [`raw-pairs.tsv`](raw-pairs.tsv).
