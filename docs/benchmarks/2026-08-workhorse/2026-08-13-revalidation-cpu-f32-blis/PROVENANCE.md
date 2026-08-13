# CPU F32 revalidation provenance

This is a fresh revalidation of the locked DA3-BASE CPU F32 comparison, run
on 13 August 2026 on the AMD Ryzen 9 9950X Workhorse. It is an independent
result bundle and does not overwrite the earlier qualified iteration 55
study.

## Result

| Runtime | Mean of 10 trial medians | 95% CI | Pooled p95 |
|---|---:|---:|---:|
| Rust | 171.542 ms | [166.078, 177.006] ms | 186.065 ms |
| C++/ggml | 238.136 ms | [236.287, 239.986] ms | 248.357 ms |

The Rust arm has 38.8% higher throughput than C++/ggml
(`238.136 / 171.542 - 1`), equivalently 28.0% lower latency. The confidence
intervals do not overlap.

## Locked conditions

- Model: `depth-anything-base-f32.gguf` (the raw file includes its SHA-256).
- Image: `mountains.jpg`, resized to 504x336 by every arm.
- CPU budget: 16 threads for both arms.
- Per arm: one untimed warm-up and 10 timed iterations.
- Study: 10 fresh, randomized process trials per arm; 100 timed samples per
  arm; seed `20260812`; three-second cooldown between process trials.
- Timed work: preprocessing, DA3 backbone, depth/confidence head and host
  postprocessing. Model loading and JPEG decoding are excluded.

## Runtime environment

The Rust executable is the BLIS-enabled Zen 5 build used by the accepted
runtime configuration. Its required shared libraries and runtime switches
were explicitly supplied for this run:

```bash
export LD_LIBRARY_PATH=/tmp/da3-blis-install/lib:/tmp/da3-clang64/usr/lib64
export DA3_KERNELS_BLIS_LINEAR=1
export DA3_HEAD_BLIS_GEMM=1
```

The executable hashes captured before the run were:

- Rust: `ce493321daede87e68bcc023c4f163c0ac111593466223a46142b150d5b2ced4`
- C++/ggml: `eba42df633ebc5f4f6c178e0c39e80054124a3591a49e4e7f8da1d73e81aece5`

The Workhorse checkout is a mounted source snapshot rather than a Git worktree;
therefore the raw harness records executable hashes as the primary build
identity. The external kernel source fingerprint captured before the run was
`03441a85c9c601a7a40e6676211b766372c1050f0118e02f560cbe801dce4c20`.

No timed sample was recorded during the initial preflight attempt: it failed
before inference because the dynamic linker could not find the BLIS library.
The final bundle was started afresh after restoring the documented library
path, so it contains only valid measurements.

## Integrity

- `raw-results.json` SHA-256:
  `6b78c181a68f9fcac488cc8dbeec360cf0b3f9f5effa18201be2154babda8675`
- `RESULTS.md` SHA-256:
  `920c7862dadb6828ed10cdc4af6b511412a3b7b893e4ab5c73e396bad73a2eb0`

This run re-establishes performance only. It relies on the already accepted
four-image C++ F32 parity gate; it does not substitute for a new parity study
when numerical implementation changes.
