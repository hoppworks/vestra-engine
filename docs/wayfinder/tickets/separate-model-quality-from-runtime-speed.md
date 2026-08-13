# Separate model quality from runtime speed

Label: `wayfinder:research`

## Question

How can the public benchmark compare Depth Anything V2 with the models in its
official DA-2K table without making a false same-model performance claim?

## Resolution

Closed — see [the benchmark landscape research](../research/benchmark-landscape.md).

The project has two suites: a DA-2K model-quality comparison for Depth Anything
V1, V2, Marigold, GeoWizard, DepthFM, and any future owned model; and an
exact-weight V2-Small runtime comparison. Only the latter can say that one
runtime is faster than another. V2-Base is reported separately because its
checkpoint is non-commercial.
