# Iteration 40: temporary AOCL-BLIS backend candidate

This records the state omitted by the generic scientific-runner schema. The
candidate was enabled only for the Rust arm by inheriting:

```text
DA3_KERNELS_BLIS_LINEAR=1
LD_LIBRARY_PATH=/tmp/da3-blis-install/lib:/tmp/da3-clang64/usr/lib64
RUSTFLAGS=--cfg da3_blis -L native=/tmp/da3-blis-install/lib -C target-cpu=znver5
RAYON_NUM_THREADS=16
```

The C++ arm had none of those variables. Both arms otherwise used the locked
F32 model and Mountains input recorded in `raw-results.json`.

## Result

| Runtime | Mean of 10 trial medians | 95% CI |
|---|---:|---:|
| Rust + BLIS candidate | 188.005 ms | [181.437, 194.574] ms |
| C++/ggml F32 | 238.809 ms | [237.230, 240.387] ms |

The previous accepted Rust workspace study was 195.986 ms, so the point
estimate is 7.981 ms (4.07%) lower. The candidate does not meet the
sporting -30 ms GEMM target and its Rust confidence interval is wide; it is a
qualified intermediate candidate, not a final claim.

## Parity

Candidate Rust F32 versus C++ F32 PFM output:

| Image | Pearson r | MAE |
|---|---:|---:|
| canyon | 0.9999936278 | 0.0018125444 |
| desk | 0.9999782570 | 0.0017729213 |
| mountains | 0.9999855778 | 0.0036751658 |
| street | 0.9999721236 | 0.0008210414 |

Every image passes the project threshold r >= 0.9999 and MAE <= 0.005.

## Provenance

| Artifact | SHA-256 / version |
|---|---|
| Rust candidate binary | `8b1bf1ec72a33a65b898499e82c3ee5bea73dce51338a502c6fc99b8d0e4cc68` |
| C++ binary | `eba42df633ebc5f4f6c178e0c39e80054124a3591a49e4e7f8da1d73e81aece5` |
| external `da3-kernels/src/lib.rs` | `3b2d2c57ed66d5da9bb9b055472d6761ed198538fde6cd5909ba6d09cff37c89` |
| `vit_block.rs` | `60d3ef7c64ca174aab4e3e0f7067eb8a2d3cf1126c31188abe7ba451b4e4cfe7` |
| `dpt_head.rs` | `3e4894379a68bd97d2efb6807fc4f8c67ae32701b6d516ff1338725502cba5f2` |
| `engine.rs` | `92a276a5c0d77ca50a352a7cfc6e9af764b233f87de400fa6c11599b3601a317` |
| BLIS shared library | `bac67908ea7da77022ff826068bfa359ab57c684bef3a5b7855e321df7e4038a` |
| BLIS source | `9212e3b464ef3310b093d4405222ec79afd147b4` |
| Compiler used for BLIS | Clang 22.1.8, temporary `/tmp` extraction |
| Raw benchmark bundle | `ccf34ea6c044b79cf85b5f00957603ee912a5f29a4fe5c6c2066b3075469f48e` |

BLIS was built as a temporary OpenMP `zen4` configuration from AMD's BLIS
source. No system package or persistent Workhorse runtime was changed.
