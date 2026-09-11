# Stage-0 composed transpose/lateral candidate

## Hypothesis

The DA3 DPT head performs, without an intervening nonlinearity:

1. a learned `96 -> 96`, 4x4 stride-4 transposed convolution with bias; then
2. a bias-free `96 -> 128`, 3x3 pad-1 lateral convolution.

For stage 0 these linear operations can be composed into one prepared
`96 -> 128`, 6x6 stride-4, pad-1 transposed convolution. This removes the
materialized `96 x 96 x 144` activation (about 5.1 MiB) and reduces the
estimated pairwise arithmetic from roughly 807M MACs to 382M MACs.

## Constraints

- Do not change LayerNorm, the `768 -> 96` projection, UV addition, or later
  RefineNet arithmetic.
- The resize bias is not a single channel bias after composition. Its lateral
  convolution produces nine interior/edge/corner spatial classes; the
  candidate must prepare and apply those exact spatial corrections.
- The composed weights change F32 association. Bitwise equality is not
  expected; four-image C++ F32 parity is mandatory.
- The materialized route remains the default control behind an explicit
  opt-in candidate switch.

## Required tests and admission

1. Compare candidate and materialized pair on signed random tensors, impulses,
   zero input with nonzero resize bias, every edge/corner class, and tiny
   rectangles.
2. Confirm output dimensions and all 96-to-128 channel mappings for both
   504x336 orientations.
3. Run four-image C++ F32 parity: each image must meet `r >= 0.9999` and
   `MAE <= 0.005`.
4. Run ten alternating 1-warm-up/10-timed-iteration pairs on idle Workhorse;
   require at least 3 ms end-to-end reduction and at least 8/10 wins before a
   full randomized C++/Rust qualification.

This candidate can only save part of the remaining 26.754 ms gap by itself,
but it is a different dataflow architecture rather than another microtile
variation.

