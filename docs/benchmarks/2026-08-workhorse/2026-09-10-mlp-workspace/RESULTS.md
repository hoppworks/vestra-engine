# Iteration 59 — reusable transformer MLP activations

## Hypothesis and isolated change

The single-view DA3-BASE CPU backbone executes twelve blocks at one fixed
activation shape. The existing workspace retained attention buffers, but each
block still allocated LN1, LN2, FC1's 865×3072 activation, and FC2's
865×768 residual branch. Reusing those buffers should reduce allocator and
zero-initialisation traffic without changing matrix multiplication, elementwise
order, precision, model data, or thread budget.

Commit `18baef1` adds reusable `norm`, `mlp_hidden`, and `branch` buffers to
`VitWorkspace`, turns LayerNorm into a destination-writing helper, and routes
the CPU MLP through destination-writing FC1/FC2 calls. Commit `4f808f9`
removes the retired allocation wrapper from release builds.

## Correctness

- `cargo test --locked -p vestra-engine --lib`: 70 tests passed before the
  post-change capacity assertion; the focused workspace test also passes after
  it.
- The workspace regression test runs fresh, reused, and NaN-poisoned reuse
  routes, compares every output float bitwise, and verifies the three MLP
  activation capacities remain stable on the second fixed-shape pass.
- Four-image Rust versus C++ F32 parity remains inside the locked gate:

| Image | Pearson r | MAE |
|---|---:|---:|
| canyon | 0.9999936280 | 0.0018125201 |
| desk | 0.9999782568 | 0.0017729246 |
| mountains | 0.9999855775 | 0.0036752027 |
| street | 0.9999721240 | 0.0008210033 |

## Smoke measurement — not a qualified comparison

The Workhorse was still occupied by unrelated CPU-heavy work. One exact
protocol-shaped diagnostic (`1` warm-up plus `5` timed calls) measured a
206.275-ms median, with individual measurements from 193.762 to 236.499 ms.
That spread makes it unsuitable for an before/after claim and it is not
compared to C++ here. The next valid action is an alternating smoke and then
the locked randomized 20-trial Rust/C++ study only while the machine is idle.

## Next hypothesis

The persistent MLP allocation path is now closed. The remaining MLP time is
dominated by the FC1/FC2 arithmetic itself, so the next candidate must improve
the fixed 865×768×3072 and 865×3072×768 GEMM execution rather than revisit
allocator micro-optimizations. Before replacing BLIS, its actual Zen-5
subconfiguration and packing behaviour must be profiled on an idle Workhorse.
