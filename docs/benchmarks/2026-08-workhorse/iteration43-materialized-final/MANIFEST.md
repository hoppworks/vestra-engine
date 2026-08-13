# Iteration 43 manifest

- Date: 2026-08-13
- Hardware: AMD Ryzen 9 9950X, 16 benchmark threads
- Model: `depth-anything-base-f32.gguf`, SHA-256
  `1b13b166e8a8b4f2c862f42d36edb2f9aab995a18cc527a52b9f160b99c6b8da`
- Input: `mountains.jpg`, SHA-256
  `936d60f43c51fe99156563a0d3c5da69cf84a39cbde5e443bea7662500b8c969`
- Protocol: 10 fresh-process randomised trials per arm; one warm-up and ten
  measured iterations per trial; three-second cooldown; seed `20260815`.
- Candidate environment in addition to the accepted BLIS-head environment:
  `DA3_DISABLE_FUSE_FINAL_RESIZE_WINO=1`.
- The scientific runner records only `RAYON_NUM_THREADS`; the complete
  inherited candidate environment is recorded here to prevent ambiguity.
