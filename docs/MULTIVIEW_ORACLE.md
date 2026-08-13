# Multi-view C++ oracle gate

Vestra Engine's automatic multi-view path is accepted only when it matches
the pinned `localai-org/depth-anything.cpp` PR #2 revision recorded by Vestra.
The compared work is identical: same GGUF, ordered input images, production
resize path, view count, and F32 precision.

Build the pinned C++ checkout, then produce the reference artifacts:

```bash
da3-cli depth --model depth-anything-base-f32.gguf \
  --input frame-00.png --input frame-01.png \
  --out-prefix /tmp/cpp-window
```

Run the equivalent Vestra Engine pass:

```bash
cargo run -p vestra-cli -- infer-multi \
  --model depth-anything-base-f32.gguf \
  --image frame-00.png --image frame-01.png \
  --out-prefix /tmp/rust-window
```

Compare the artifacts and retain the JSON report with the fixture provenance:

```bash
python3 scripts/compare_multiview_oracle.py \
  --cpp-prefix /tmp/cpp-window \
  --rust-prefix /tmp/rust-window \
  --views 2 \
  --output multiview-s2.json
```

Repeat for ordered `S=3` and `S=12` windows. Every view must satisfy depth
Pearson `r >= 0.9999`, depth MAE `<= 0.005`, W2C extrinsics MAE `<= 0.005`,
and intrinsics max absolute error `<= 1.5` pixels. Camera tensors are measured
separately because a raw concatenated "pose MAE" mixes unitless rotation and
translation with focal lengths in pixels, which is not physically meaningful.
The script does not suppress outliers or average failures away: one failed
view makes the run fail.

## Diagnosing a failed window

Set `VESTRA_TRACE_DIR` for a temporary instrumented C++ oracle and the Vestra
Engine invocation. Both write `block-0` through `block-11` tensors in
`[view][token][channel]` F32 order. Locate the first divergent block with:

```bash
python3 scripts/compare_block_trace.py \
  --cpp-dir /tmp/cpp-trace --rust-dir /tmp/rust-trace --blocks 12 \
  --output block-trace.json
```

Do not tune a later block until the earliest divergent block is understood.
