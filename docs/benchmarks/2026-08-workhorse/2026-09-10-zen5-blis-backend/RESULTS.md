# Iteration 61 — AOCL-BLIS Zen-5 backend audit

## Hypothesis

The Ryzen 9 9950X is Zen 5, but the linked AOCL-BLIS build reported
`CONFIG_NAME := zen4` and `bli_arch_query_id() == zen4`. A dedicated AOCL 5.3
Zen-5 build might improve the fixed DA3 projection GEMMs without altering
Vestra arithmetic or the benchmark workload.

## Experiment

On the Workhorse, AOCL-BLIS 5.3 source revision `9212e3b` was built separately
with `./configure --enable-threading=openmp … zen5`. The resulting library
reported `bli_arch_query_id() == zen5`; a separately linked Vestra binary was
used with the same model, image, F32 switches, and sixteen-thread budget.

## Result and decision

The Workhorse was not idle, so these alternating five-sample diagnostics are
not publishable benchmark evidence:

| Arm | Median 1 | Median 2 |
|---|---:|---:|
| existing Zen-4 AOCL-BLIS | 196.413 ms | 198.279 ms |
| dedicated Zen-5 AOCL-BLIS | 198.459 ms | 201.043 ms |

The candidate was not directionally favorable and one run had a 265.863-ms
p95. It is therefore rejected as the default backend. The independently built
library is retained on the Workhorse only as an auditable diagnostic artifact;
no source or benchmark contract changed because of this experiment.

## Follow-up

The remaining GEMM work must target prepacking/layout or a Vestra-specific
kernel rather than merely swapping AOCL configuration families. Any such
candidate must exceed this failed library substitution on an idle-machine,
whole-model measurement.
