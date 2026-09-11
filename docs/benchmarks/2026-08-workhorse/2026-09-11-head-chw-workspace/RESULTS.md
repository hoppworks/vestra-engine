# Iteration 98 — reuse CHW head-handoff buffers (reverted)

## Hypothesis and isolated change

The production DPT head already owns a pool for temporary maps, but the four
token-major-to-CHW handoffs allocated fresh `[768, grid_h, grid_w]` buffers
on every inference.  This candidate wrote those fully-overwritten buffers
into `HeadWorkspace` storage and returned them immediately after their 1x1
projection.  Arithmetic, operation order, and the public debug route were
unchanged.

## Correctness

- The local Engine library suite passed (72 tests), including its existing
  bitwise workspace-reuse test.
- C++ F32 parity passed on every locked image:

| Image | Pearson r | MAE |
|---|---:|---:|
| canyon | 0.9999936282 | 0.0018124970 |
| desk | 0.9999782565 | 0.0017728067 |
| mountains | 0.9999855792 | 0.0036749614 |
| street | 0.9999721254 | 0.0008209983 |

## Locked smoke measurement

The Ryzen 9 9950X Workhorse ran the accepted route and candidate in ten
alternating pairs.  Both used the same Rust 1.93.0 toolchain, DA3 BASE F32
GGUF, `mountains.jpg`, 504x336 resize, 16 fixed threads, accepted flags, and
one warm-up plus ten timed iterations per arm.

| Pair | Control median (ms) | CHW workspace median (ms) |
|---:|---:|---:|
| 1 | 156.752 | 154.680 |
| 2 | 158.622 | 160.810 |
| 3 | 157.609 | 157.684 |
| 4 | 164.155 | 157.388 |
| 5 | 158.485 | 154.992 |
| 6 | 164.693 | 168.543 |
| 7 | 158.467 | 157.766 |
| 8 | 158.010 | 157.244 |
| 9 | 161.917 | 154.591 |
| 10 | 161.076 | 161.400 |

Control mean: 159.379 ms. Candidate mean: 158.510 ms. The 0.869 ms (0.55%)
reduction wins only 6/10 pairs. It fails the 2 ms and 8/10 admission rule,
so it cannot be claimed as a reliable optimization.

## Decision

Rejected and reverted.  Recycled allocation alone is below the measurement
noise at the locked workload; future head work must reduce major convolution
or refinement computation rather than allocator traffic.
