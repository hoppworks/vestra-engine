# CPU F32 revalidation — 20 trials

This is the higher-confidence revalidation requested after the initial
10-trial run showed appreciable Rust variation. It is a new, fully randomized
20-trial study, not a statistical merge of two separate studies.

## Result

| Runtime | Mean of 20 trial medians | 95% CI | Trial-median SD | Pooled p95 |
|---|---:|---:|---:|---:|
| Rust | 171.141 ms | [168.042, 174.241] ms | 6.622 ms | 185.641 ms |
| C++/ggml | 238.789 ms | [237.406, 240.172] ms | 2.956 ms | 248.874 ms |

Rust has **39.5% higher throughput** than C++/ggml
(`238.789 / 171.141 - 1`), equivalently **28.3% lower latency**. The 95%
confidence intervals remain clearly separated.

The expanded sample lowers Rust's standard error from the preceding N=10
revalidation while retaining the same conclusion. Twenty independent trials
are sufficient for this comparison at the agreed decision point; no further
N increase is warranted unless the environment or implementation changes.

## Contract

- Hardware: AMD Ryzen 9 9950X Workhorse.
- Model: `depth-anything-base-f32.gguf`, F32 for both arms.
- Input: `mountains.jpg`; every arm resizes to 504x336.
- CPU budget: 16 threads for both arms.
- Per process: one untimed warm-up, then 10 timed iterations.
- Study: 20 fresh randomized trials per arm, 200 timed samples per arm,
  seed `20260812`, and a three-second inter-process cooldown.
- Timed work: preprocessing, backbone, depth/confidence head and host
  postprocessing. Model loading and JPEG decoding are excluded.
- Interval: two-sided 95% Student-t interval over 20 trial medians
  (`df=19`, critical value 2.093).

## Runtime environment and identity

The Rust executable is the Zen 5 BLIS-enabled build. The run explicitly used:

```bash
export LD_LIBRARY_PATH=/tmp/da3-blis-install/lib:/tmp/da3-clang64/usr/lib64
export DA3_KERNELS_BLIS_LINEAR=1
export DA3_HEAD_BLIS_GEMM=1
```

- Rust executable SHA-256:
  `ce493321daede87e68bcc023c4f163c0ac111593466223a46142b150d5b2ced4`
- C++/ggml executable SHA-256:
  `eba42df633ebc5f4f6c178e0c39e80054124a3591a49e4e7f8da1d73e81aece5`
- External kernel source SHA-256:
  `03441a85c9c601a7a40e6676211b766372c1050f0118e02f560cbe801dce4c20`

The Workhorse checkout is a mounted source snapshot rather than a Git
worktree, so executable and source-file hashes are the reproducibility anchor.

## Integrity and scope

- `raw-results.json` SHA-256:
  `352000fde9c382164d2d4625c5e751a303cab02b036fdb98d1e0726e37c654a7`
- `RESULTS.md` SHA-256:
  `c8cb580943ce49114e4f00f004fad3ecadff12910681992014a0bf49bf7dafab`

This is a performance revalidation only. It relies on the accepted four-image
C++ F32 parity gate and must not substitute for a new parity study after any
numerical implementation change.
