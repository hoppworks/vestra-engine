# Contributing to Vestra Engine

Vestra Engine accepts changes that preserve the model contract, numerical
parity, and benchmark protocol.

Before submitting a change, run the workspace tests and the relevant CLI
smoke test. Changes that affect inference arithmetic must also run the
four-image F32 parity corpus. A performance-path change additionally needs an
alternating smoke benchmark before it can be considered for the full
randomized trial study.

Do not vendor kernel source into this repository. Add or change low-level CPU
work in Vestra Kernels, publish or pin its revision, then consume its stable
public API here. Keep model IDs, weights, inputs, compiler flags, and trial
raw data out of commits unless they are already public benchmark fixtures.
