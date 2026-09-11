# Zen-5 BLIS runtime-library candidate

## Status

**Rejected after smoke admission.** No source or model change was made. The
candidate swapped only the dynamically loaded BLIS library while preserving
the existing DA3 BASE F32 binary, input, 16-thread contract and timed work.

## Hypothesis

The Workhorse contains a separate Zen-5-specialized BLIS build at
`/var/roothome/da3-blis-zen5-20260910/lib`. Loading it instead of the accepted
`/var/roothome/da3-blis-install-9212/lib` build could improve the large linear
projections in the DA3 backbone without affecting numerical output.

## Method

- Hardware: AMD Ryzen 9 9950X, 16 fixed benchmark threads.
- Model: `depth-anything-base-f32.gguf`.
- Input: `mountains.jpg`, 504×336.
- Work: one warm-up, then ten timed inferences; model loading and decode
  excluded by the established `vestra-engine bench` contract.
- Arms were alternated per trial. All accepted runtime switches were held
  fixed; only `LD_LIBRARY_PATH` selected the BLIS library.

## Results

| Trial | Accepted BLIS median (ms) | Zen-5 BLIS median (ms) | Zen-5 - accepted (ms) |
|---:|---:|---:|---:|
| 1 | 156.469 | 164.558 | +8.089 |
| 2 | 166.771 | 160.207 | -6.564 |
| 3 | 157.444 | 164.082 | +6.638 |
| 4 | 162.941 | 165.200 | +2.259 |
| 5 | 164.047 | 162.494 | -1.553 |
| mean | 161.534 | 163.308 | +1.774 |

The Zen-5 library won 2/5 pairs and was 1.774 ms slower on mean trial median.
It fails the pre-registered smoke admission rule (at least 3 ms reduction and
8/10 pair wins), so no full qualification study was run and the runtime path
remains unchanged.

Raw runner output is retained at
`/var/tmp/vestra-blis-zen5-smoke-20260911/runner.log`.
