# Iteration 41: BLIS transformer + DPT head projections

The Rust candidate extends iteration 40's temporary BLIS backend to DPT
head GEMMs. It is enabled only by the following Rust-arm environment:

```text
DA3_KERNELS_BLIS_LINEAR=1
DA3_HEAD_BLIS_GEMM=1
LD_LIBRARY_PATH=/tmp/da3-blis-install/lib:/tmp/da3-clang64/usr/lib64
RUSTFLAGS=--cfg da3_blis -L native=/tmp/da3-blis-install/lib -C target-cpu=znver5
RAYON_NUM_THREADS=16
```

The C++ arm had none of those variables. The locked model and input hashes are
in `raw-results.json`.

## Qualified result

| Runtime | Mean of 10 trial medians | 95% CI |
|---|---:|---:|
| Rust candidate | 181.138 ms | [175.623, 186.653] ms |
| C++/ggml F32 | 238.513 ms | [236.701, 240.325] ms |

This is a 31.67% throughput advantage (`238.513 / 181.138 - 1`) and 14.848
ms lower latency than the previous 195.986-ms accepted workspace baseline.
The 40%-speed target at this C++ result is 170.367 ms, so 10.771 ms remain.

## F32 parity

| Image | Pearson r | MAE |
|---|---:|---:|
| canyon | 0.9999936279 | 0.0018125425 |
| desk | 0.9999782569 | 0.0017729257 |
| mountains | 0.9999855778 | 0.0036751673 |
| street | 0.9999721236 | 0.0008210428 |

All four images satisfy r >= 0.9999 and MAE <= 0.005 against C++ F32.

## Additional provenance

| Artifact | SHA-256 |
|---|---|
| Rust binary | `458d9c9e3ffb3c561e7eaa2fbda5ad52abe45e75932bfdd503177e424e800b96` |
| C++ binary | `eba42df633ebc5f4f6c178e0c39e80054124a3591a49e4e7f8da1d73e81aece5` |
| external `da3-kernels/src/lib.rs` | `019c2bc1172fae74652707d81497142a822993b923802e6eb73ac1e34aca9386` |
| Rust `gemm.rs` | `c8a5c7c341eb6b31fc29d5e20128a0e4a531ddb29f93b32933db8519ecfc1fa3` |
| Rust `dpt_head.rs` | `a2c2be635e5eadebed92e79b242ced61bda389eb954cb198e2a2e086d5327319` |
| BLIS library | `bac67908ea7da77022ff826068bfa359ab57c684bef3a5b7855e321df7e4038a` |
| Raw bundle | `9b884847da2dc91ed55dbf55884c9ed1345eeb9f0cefa54e6532f62b0471288a` |

The candidate remains an explicit experimental build. Its 95% interval is
still wider than the C++ interval, so it does not prove the final 40% claim.
