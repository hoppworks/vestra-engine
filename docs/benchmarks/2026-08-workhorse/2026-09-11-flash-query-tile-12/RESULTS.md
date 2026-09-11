# Flash-attention 12-query scheduler candidate

## Status

**Rejected after smoke admission.** The candidate changes only the query-tile
scheduler from the accepted eight rows to twelve rows. It does not alter model
weights, input, F32 arithmetic within a tile, or the 16-thread budget.

## Hypothesis

The 24 DA3 attention calls dominate the remaining backbone time. A twelve-row
query task could reduce scheduler overhead relative to the eight-row default
without introducing the register pressure of wider older variants.

## Method

On AMD Ryzen 9 9950X with the locked DA3 BASE F32 `mountains.jpg` contract,
each arm used one warm-up plus ten timed inferences. Control and
`DA3_KERNELS_FLASH_QUERY_TILE=12` candidate alternated over five trials;
all accepted runtime switches and 16 benchmark threads were fixed.

## Results

| Trial | QT8 control median (ms) | QT12 median (ms) | QT12 - QT8 (ms) |
|---:|---:|---:|---:|
| 1 | 158.106 | 174.779 | +16.673 |
| 2 | 165.400 | 176.935 | +11.535 |
| 3 | 159.323 | 170.961 | +11.638 |
| 4 | 160.780 | 167.783 | +7.003 |
| 5 | 162.803 | 169.202 | +6.399 |
| mean | 161.282 | 171.932 | +10.650 |

QT12 lost every pair and is 10.650 ms slower on the mean trial median. It
fails admission decisively; no source change or full qualification is allowed.

Raw runner output is retained at
`/var/tmp/vestra-flash-qt12-smoke-20260911/runner.log`.
