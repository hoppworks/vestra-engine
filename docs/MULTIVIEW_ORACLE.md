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
Pearson `r >= 0.9999`, depth MAE `<= 0.005`, and pose MAE `<= 0.005`.
The script does not suppress outliers or average failures away: one failed
view makes the run fail.
