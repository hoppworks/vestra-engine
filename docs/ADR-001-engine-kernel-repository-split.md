# ADR-001: Engine and kernel repository split

Status: accepted

## Context

Vestra Engine combines model semantics with CPU-specific implementation work.
That made it difficult to qualify low-level changes independently, reuse them
for another backend, and keep the engine API free from machine-specific types.

## Decision

Vestra Engine owns GGUF loading, calibrated preprocessing, model topology,
depth/confidence/pose inference, multi-view orchestration, the CLI, parity
fixtures, and the public inference API. Vestra Kernels owns primitive-slice
CPU kernels, ISA dispatch, microbenchmarks, and kernel-level numerical
oracles.

The dependency is one-way: Vestra Engine depends on Vestra Kernels. Kernels
must not import engine types, GGUF readers, model configuration, or CLI code.
The engine depends on an exact `vestra-kernels` Git revision recorded in
`Cargo.lock`. An uncommitted local Cargo source override is permitted only for
development; it must not replace the versioned dependency in the manifest.

## Consequences

Kernel APIs use explicit dimensions and primitive buffers. Engine public types
do not expose kernel implementation structs. CUDA or another backend can be
introduced behind the kernel API without rewriting model semantics. A kernel
change must be qualified in the kernel repository and then re-qualified
end-to-end by the engine benchmark protocol.

## Rejected alternative

Keeping an in-tree compatibility kernel crate would preserve short-term import
paths but leaves two implementations and violates the single ownership rule.
