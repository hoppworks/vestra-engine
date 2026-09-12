# Optimization carry-over audit

## Question

Did the repository split discard the July CPU optimizations, requiring the old
branch to be merged into Vestra Engine?

## Decision

No wholesale merge is justified. The old branch was never merged into the old
C++ repository's `main`, but its important mechanisms are present or superseded
in the extracted engine and kernel repositories. Applying the branch directly
would reintroduce an older graph architecture, duplicate later direct kernels,
and mix a single-thread strategy with the qualified 16-thread path.

## Mechanism mapping

| Old branch mechanism | Current implementation | Status |
|---|---|---|
| AVX-512 tiled online-softmax attention | External AVX-512 packed/flash attention plus contiguous QKV layouts | Superseded and further optimized |
| Zero-copy graph weights and one compiled ViT plan | Direct transformer execution with fixed-shape kernels and reusable model-owned state | Superseded; the generic graph is not the qualified hot path |
| DPT transpose GEMM, Winograd convolution, SIMD resize, reusable head workspace | Qualified Winograd/filter caches, blocked external kernels, fused final resize path, and `HeadWorkspace` | Retained and extended |
| `faer` 0.24 as the large-projection accelerator | Zen-targeted fixed-shape kernels and explicit BLIS bridges; `faer` remains a fallback | Replaced by faster target-specific routes |
| GEMM-backed attention and fused bias/GELU | Packed attention, fused projection epilogues, fused FC1 bias/GELU, and fused QK normalization/RoPE | Retained and extended |
| Opt-in thread count and persistent worker gang | Global 16-thread execution across projection, attention, convolution, LayerNorm, RoPE, and epilogue work | Replaced by the qualified multi-kernel thread policy |

## Historical numbers are not interchangeable

The old `0.613×` result was a single-thread measurement. It means 38.7% lower
latency, or 63.1% higher throughput, for that old machine/protocol pairing. It
cannot be combined with the later 16-thread headline.

The 2026-08-12 result used a pinned C++ revision and measured 208.889 ms for
Rust versus 241.203 ms for C++/ggml. Later work improved the Rust path further,
but those results still belong to their recorded compiler, kernel revision, and
C++ binary. The current-upstream study in this directory is the only evidence
that may support a claim against public C++ commit
`739992d10bf9472c46dcd4622b14d2b20766c58d` as checked on 2026-08-30.

## Merge policy

Use the old branch only as a hypothesis ledger. A mechanism may be reintroduced
only when it is absent from the current architecture, still relevant to the
same benchmark scope, numerically verified, and faster in a fresh whole-model
study. Commit-level cherry-picking is not an acceptable shortcut across the
repository split.
