# Iteration 57 — fused DPT token-normalization handoff

## Hypothesis and change

The previous DPT route cloned each captured token-major feature, normalized
it, and then materialized a second CHW buffer. The existing fused routine
writes the normalized values directly to CHW. Commit `a7993f6` makes that
bitwise-tested route the default, with
`DA3_DISABLE_FUSE_HEAD_TOKEN_NORM_CHW=1` retained as its diagnostic control.

## Correctness

The existing unit oracle for the fused and materialized paths passes bitwise.
On the Workhorse, current Rust output remains within the locked C++ F32 gate
for canyon, desk, mountains, and street: every Pearson r is at least 0.9999
and every MAE is at most 0.005. No arithmetic, model, resize policy, or thread
budget changed.

## Measurement status

Alternating ten-sample smoke runs were directionally favorable but were run
while unrelated CPU-heavy processes were active on the Workhorse. They are
therefore intentionally not reported as a speedup and do not promote this
iteration to the full 20-trial study. The final randomized comparison must be
rerun only after the machine is idle.
