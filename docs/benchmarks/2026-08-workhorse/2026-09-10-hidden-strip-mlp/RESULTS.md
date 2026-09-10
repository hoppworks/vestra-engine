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
available.  On the Ryzen 9 9950X, its AVX-512/FMA body ran and passed:
`cargo test hidden_strip_primitives_preserve_full_projection_bits --lib`.
On non-x86 hosts it intentionally skips the ISA-specific body.

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
| Hidden-strip + existing opt-in Out1 F(2) kernel | 1 warm-up + 3 timed | 192.350 ms |

The candidate was therefore 15.5% lower latency in this noisy paired smoke.
The combined candidate was 22.7% lower latency than the same busy-host
standard-route smoke. It is promising but unqualified: the arms ran
sequentially on a busy host, and no randomized trial confidence interval was
computed.

### Rejected composition: commuting Fusion resize and 1x1

The existing `DA3_COMMUTE_FUSION_RESIZE_1X1=1` candidate was tested only on
top of the combined hidden-strip + Out1 configuration. Its 1-warm-up +
3-timed smoke median was **195.712 ms**, versus **192.350 ms** without that
flag. It therefore did not improve this candidate composition and remains
off. This result is also diagnostic-only because the Workhorse was busy.

### Final resize row-ring candidate

`vestra-kernels` `07b0edbe5839a2d96b050940f94c0d74fae6eddf` adds the opt-in
`DA3_FUSED_FINAL_RESIZE_ROW_RING=1` route for the exact final 64→32 F(2)
operation. It retains four resized+UV rows per channel and worker rather than
recomputing overlapping bilinear samples for every tile. The operation-level
timed profile reduced final resize/Out2a from the prior approximately
15.338 ms diagnostic observation to **7.225 ms** in its timed iteration.

With hidden-strip MLP, Out1 F(2), and the row-ring candidate enabled, the
busy-host 1-warm-up + 3-timed smoke produced 187.211 ms median
(`187.211, 230.879, 175.377` ms). The 230.879 ms outlier confirms that this
host is not suitable for a publishable comparison; treat the result only as
admission evidence. The complete four-image C++ F32 parity gate again passed
with the same per-image values recorded above.

The kernel unit suite includes a bitwise fused-route comparison with signed
UV/bias, boundary coverage through more than four tile rows, NaN overwrite
checking, and consecutive-input scratch reuse. The first attempt exposed a
stale top-border row-tag bug; it was fixed before this candidate was pushed.

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
