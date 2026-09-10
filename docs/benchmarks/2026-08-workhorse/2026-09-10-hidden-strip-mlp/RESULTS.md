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

### Rejected admission: streamed rn1 resize → pointwise projection

`vestra-kernels` `95673a97bb296c784175fa1c6200d6a6f4b8018a` and
`vestra-engine` `750834e` add `DA3_STREAM_FUSION_RESIZE_1X1=1`, an exact-order
candidate for rn1's `96×144 → 192×288`, 128→128 resize plus 1×1 projection.
It streams six destination pixels at a time into the existing packed AVX-512
projection microkernel and does not invoke nested BLIS workers.

It failed the first performance admission: a same-binary diagnostic head
profile measured the combined rn1 resize/projection at about **4.914 ms**
with the candidate, compared with **4.217 ms** for the established route in
the adjacent control measurement. This misses the predeclared 30% reduction
target. The candidate remains opt-in for future investigation and is not part
of the composed benchmark configuration. No F32 parity claim is made for it.

Four-image C++ F32 parity (`cpp_{image}.pfm`) passed the required per-image
gate (`r >= 0.9999`, `MAE <= 0.005`):

| Image | Pearson r | MAE |
|---|---:|---:|
| canyon | 0.9999936282 | 0.0018124970 |
| desk | 0.9999782565 | 0.0017728067 |
| mountains | 0.9999855792 | 0.0036749614 |
| street | 0.9999721254 | 0.0008209983 |

### Candidate: remove nested Rayon bias scheduling in hidden-strip FC1

**Hypothesis.** Each 55-token hidden-strip slab already owns its output buffer
inside an outer Rayon worker. The generic `scalar::add_bias_rows` helper starts
another Rayon traversal for inputs with at least 32 rows. All 16 slabs meet
that threshold (15 × 55 rows and one 40-row tail), and each has 48 FC1 strips,
so the executor unnecessarily starts **9,216 nested scheduling traversals per
inference**. Replacing only this local bias epilogue with a serial row loop
should remove scheduling and worker-stealing overhead without changing F32
arithmetic.

**Change.** `vestra-engine` `c6e273b` uses the worker-local helper only for
the FC1 hidden strip. Global bias operations, GELU, FC1/FC2 kernels and all
execution flags are unchanged.

**Verification.** The new regression test compares the worker-local helper
bit-for-bit with `scalar::add_bias_rows` for both the 40-row tail and a
55-row slab. `cargo test -p vestra-engine --lib` passed 72/72. A four-image
C++ F32 parity gate and a controlled idle-host A/B smoke are still required
before promotion; the Workhorse is currently busy, so no timing result is
recorded here.

**Predeclared admission.** Keep this change only if a controlled A/B shows at
least 2 ms end-to-end reduction or at least 5% reduction in the complete MLP
phase. This is deliberately a low-single-digit-ms hypothesis, not a claimed
large kernel win.

## Decision

Keep the candidate opt-in.  Its next admission gate is an idle-host,
alternating control/candidate smoke with the same execution-policy settings.
Only a reproducible gain followed by the locked ten-independent-trial study
can promote it or change the public Rust-versus-C++ result.
