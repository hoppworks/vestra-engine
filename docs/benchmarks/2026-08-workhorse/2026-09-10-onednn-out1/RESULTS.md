# Rejected experiment: strict-F32 oneDNN `head.scratch.out1`

## Hypothesis

A prepared oneDNN CPU convolution could beat Vestra's prepared F(2x2, 3x3)
Winograd operation for the isolated DA3-BASE output-head convolution:

```text
NCHW 1×128×192×288 → 1×64×192×288, OIHW 64×128×3×3,
stride 1, padding 1, bias, no activation
```

The measured route included the full operation boundary. Immutable weights were
reordered once during primitive preparation; every timed invocation bound an
NCHW activation and output, executed the primitive with a user scratchpad, and
waited for completion.

## Native build and fairness controls

- Workhorse: AMD Ryzen 9 9950X; target thread budget 16.
- oneDNN `v3.9.2`, source commit
  `fef486592e40c9e907e615e747118620b4611e04`.
- Clang `22.1.8`, `DNNL_CPU_RUNTIME=OMP` and LLVM `libomp.so`.
- The existing BLIS backend resolves to the same LLVM `libomp.so`; no
  `libgomp`/`libomp` mixed process was measured.
- `RAYON_NUM_THREADS=16`, `OMP_NUM_THREADS=16`, `OMP_DYNAMIC=FALSE`, and
  `OMP_MAX_ACTIVE_LEVELS=1`.
- Primitive attributes: strict F32 fpmath, strict accumulation, and user
  scratchpad.
- Built library SHA-256:
  `f9847f2590610ae2a58c277c739bdc5e23553e07a2a28bfd9831c16530954fb7`.

## Result

oneDNN verbose output confirmed an OpenMP CPU runtime with `nthr:16`. Its
actual selected convolution was `jit_uni_ncsp_convolution`.

| Measurement | Existing path | oneDNN candidate |
| --- | ---: | ---: |
| OneDNN selected operation | n/a | 1,258.78 ms |
| Canyon smoke, 1 warm-up + 2 timed calls, median | 239.212 ms | 1,468.881 ms |

The host was not idle, so neither smoke figure is a qualified benchmark. The
candidate is nevertheless unambiguously noncompetitive: the oneDNN operation
alone is several times slower than the entire existing end-to-end smoke.

## Numerical check

The candidate's canyon output still passed the locked C++ F32 gate:

```text
Pearson r = 0.9999936279922501
MAE       = 0.001812517168690209
```

This result does not warrant a four-image or randomized-trial study because
the performance acceptance criterion failed first.

## Decision

Rejected. The experimental bridge and CLI feature were reverted from both
repositories after this result; normal builds and the established Winograd
route are unchanged. A future oneDNN revisit must first show an isolated
operation improvement versus the existing prepared Winograd route on this Zen
5 host, including all required reorders, before it is reintegrated.
