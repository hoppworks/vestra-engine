# Select the reconstruction stack

Label: `wayfinder:research`

## Question

Which maintainable, local-first stack should own video decoding, calibrated camera tracking, multi-view reconstruction/depth fusion, mesh extraction, and GLB/SVG export for this Rust-based repository, while reusing the DA3 depth engine only where it strengthens the result?

## Resolution

Closed — see [the source-backed research](../research/reconstruction-stack.md).

COLMAP owns the canonical global, calibrated camera trajectory. The official
DA3 Python sidecar supplies pose-conditioned depth and confidence. The owned
Rust pipeline owns scale, quality validation, fusion, structural extraction,
and GLB/SVG exports. DA3-NESTED/GIANT is benchmark-only because its checkpoint
is non-commercial; DA3-BASE + DA3METRIC-LARGE is the commercial-safe default.
