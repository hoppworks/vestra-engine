# Hidden-strip MLP candidate — 2026-09-10

## Hypothesis

The old experimental packed MLP materialized the complete `865 × 3072` FC1
activation before FC2.  It scheduled narrow six-token tiles independently,
so it could not retain a 64-neuron FC1 weight panel and its corresponding
activation strip in cache across a larger token slab.

The candidate visits hidden channels in ascending 64-wide strips.  Each
55-token Rayon slab computes one FC1 strip, applies the existing F32
bias/GELU implementation, and immediately accumulates it into FC2.  It never
materializes the full hidden activation.  FC2 partial sums are carried in
ascending hidden-channel order; no quantization, pruning, or approximate
math is introduced.

## Change

- `vestra-kernels` `cf387d606ef37452e368ef2e472100ef288187a4`:
  panel-local FC1 and FC2 accumulation primitives.
- `vestra-kernels` `92a4bef8`: exact-projection regression test for those
  primitives.
- `vestra-engine` `408facc`: opt-in `DA3_STRIP_MLP=1` executor.
- `vestra-engine` `55436b4`: pins the tested kernels revision.

The normal qualified route remains unchanged.  This candidate is selected
only by `DA3_STRIP_MLP=1`.

## Local verification

`cargo test -p vestra-kernels packed_gemm --lib`: 4/4 passed.

`cargo test -p vestra-engine --lib`: 71/71 passed.

The new kernel test compares the panel-local FC1 and carried FC2 accumulation
against the pre-existing packed projection bit-for-bit when AVX-512/FMA is
available.  On non-x86 hosts it intentionally skips the ISA-specific body.

## Workhorse smoke and parity

Hardware: AMD Ryzen 9 9950X, 16 configured Rayon/OpenMP threads.  Both arms
used `DA3_KERNELS_BLIS_LINEAR=1`, `DA3_HEAD_BLIS_GEMM=1`,
`RAYON_NUM_THREADS=16`, `OMP_NUM_THREADS=16`, `OMP_DYNAMIC=FALSE`,
`OMP_MAX_ACTIVE_LEVELS=1`, and `KMP_BLOCKTIME=0`.

The host had unrelated active workloads (Flutter, Chrome, VM).  The following
numbers are diagnostic smoke data, **not** benchmark claims and must not enter
the locked randomized study.

| Arm | Protocol | Median |
|---|---:|---:|
| Current qualified route, no packed MLP | 1 warm-up + 3 timed | 248.746 ms |
| Hidden-strip candidate | 1 warm-up + 3 timed | 210.188 ms |

The candidate was therefore 15.5% lower latency in this noisy paired smoke.
It is promising but unqualified: the arms ran sequentially on a busy host,
and no randomized trial confidence interval was computed.

Four-image C++ F32 parity (`cpp_{image}.pfm`) passed the required per-image
gate (`r >= 0.9999`, `MAE <= 0.005`):

| Image | Pearson r | MAE |
|---|---:|---:|
| canyon | 0.9999936282 | 0.0018124970 |
| desk | 0.9999782565 | 0.0017728067 |
| mountains | 0.9999855792 | 0.0036749614 |
| street | 0.9999721254 | 0.0008209983 |

## Decision

Keep the candidate opt-in.  Its next admission gate is an idle-host,
alternating control/candidate smoke with the same execution-policy settings.
Only a reproducible gain followed by the locked ten-independent-trial study
can promote it or change the public Rust-versus-C++ result.
