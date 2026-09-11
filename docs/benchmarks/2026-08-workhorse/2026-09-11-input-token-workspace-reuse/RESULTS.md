# Input and token workspace reuse — rejected

## Hypothesis

`Engine::forward_depth` allocated new CHW input and token vectors on every
inference, even though the locked CPU F32 workload always processes the same
504x336 resolution. Keeping those two buffers on `Engine` and returning their
capacity after preprocessing/backbone execution might reduce allocator traffic
without changing tensor values or operation order.

## Isolated change and local regression

The candidate changed only `Engine`: it took the two reusable vectors from
engine-owned fields, used the existing `preprocess` and `prepare_tokens`
functions unchanged, then returned each buffer after its final use. No kernel,
shape, flag, model, input or thread setting changed. `cargo fmt --check` and
`cargo test --locked -p vestra-engine --lib` passed locally (72 tests).

## C++ F32 parity

The Workhorse candidate passed the four locked 504x336 images against the
materialized C++ F32 reference:

| Image | Pearson r | MAE |
|---|---:|---:|
| canyon | 0.9999936282 | 0.0018124970 |
| desk | 0.9999782565 | 0.0017728067 |
| mountains | 0.9999855792 | 0.0036749614 |
| street | 0.9999721254 | 0.0008209983 |

All values satisfy `r >= 0.9999` and `MAE <= 0.005`.

## Locked paired smoke measurement

Both binaries used the accepted DA3 BASE F32 flags, Rust 1.93.0, 16 threads,
the same GGUF and `mountains.jpg`, one warm-up and ten timed iterations. Five
alternating pairs on the Ryzen 9 9950X produced:

| Pair | Control median (ms) | Reuse candidate median (ms) |
|---|---:|---:|
| 1 | 153.664 | 161.873 |
| 2 | 161.385 | 163.570 |
| 3 | 160.336 | 158.826 |
| 4 | 155.302 | 160.018 |
| 5 | 162.778 | 156.156 |
| Mean | 158.693 | 160.089 |

The candidate won only two of five pairs and was 1.396 ms slower on average.
It fails the predeclared smoke admission rule (at least 2 ms reduction and
eight wins in ten pairs), so a full randomized study is not justified.

## Decision

Reverted. The allocator was not the limiting factor; retaining the large
buffers on the long-lived engine likely worsened cache behavior around the
backbone/head boundary. The accepted allocation route remains unchanged.
