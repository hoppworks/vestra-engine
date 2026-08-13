# Continue — DA3 CPU-F32 performance goal

## Last action

Iteration 38 accepted an engine-owned DPT activation workspace. The current
fully verified result is Rust **195.986 ms** [194.540; 197.431] versus
C++/ggml **238.840 ms** [237.337; 240.343] (17.94% faster). The direct 10×
control without workspace was Rust **200.788 ms** [199.538; 202.039], so the
workspace's isolated gain is 4.803 ms. The 40% goal still requires a Rust
result below **172.091 ms** under the locked reference contract.

## Next action

Do not retry allocators, generic workspaces, QKV layout switches, Flash tile
sizes, full-score attention, or PGO: they are measured and rejected. The
hardware cycle sample puts 45% of cycles in the fixed 6×64 projection/FC
family, 20% in Flash and about 16% in Winograd. Next pursue either:

1. a genuinely different Zen-5 packed GEMM macrokernel/BLIS-AOCL-style
   backend for the exact DA3 projections, after a hardware-counter shape
   shootout; or
2. an arithmetically fused DPT pipeline, not another buffer pool.

Every new candidate needs a ten-pair alternating smoke gain of at least 5 ms
before four-image PFM parity and the full 10× study.

## Why

The current Rust path already uses direct AVX-512 QKV, projection, MLP, and
Winograd kernels. Recent evidence rules out the tempting smaller changes:
full ggml-style 64×64 attention tiling was 1.90% slower, direct HND output
projection was 0.76% slower, QKV-to-QK-norm/RoPE fusion was 5.07% slower,
and direct stage-0 DPT convolution composition was 1.74ms slower in the
head profile. The remaining 40%-credible route must change a dominant
algorithm, not merely avoid allocations.

## Verification contract

- Workhorse: `ssh -S /private/tmp/da3-workhorse-control workhorse`.
- Build Rust with `RUSTFLAGS="-C target-cpu=znver5" cargo build --release -p da-cli`.
- Locked smoke: `RAYON_NUM_THREADS=16 ./target/release/da bench --model /var/roothome/da3-bench/models/depth-anything-base-f32.gguf --image /var/roothome/da3-bench/src/depth-anything.cpp/assets/samples/mountains.jpg --warmup 1 --repeat 10`.
- Before accepting any candidate: run all four C++-PFM comparisons with
  `scripts/compare_pfm.py`; required per image: Pearson r >= 0.9999 and MAE
  <= 0.005.
- Only a serious candidate earns the randomized 10× full study using
  `scripts/run_scientific_benchmark.py --cpu-f32-direct`.

## Important evidence

- C++ fair reference uses one fused ggml graph; its current full result is
  240.928ms. Its unfused diagnostic median is preprocess 3.7ms, backbone
  141.7ms, head 101.1ms.
- Rust warm diagnostics are roughly preprocess 5ms, backbone 135ms, head
  60ms. Allocation reuse is now accepted but remains a 4.8ms-class change;
  it cannot close the remaining ~24ms gap.
- `docs/optimization-log.md` contains all hypotheses, commands, measurements,
  parity results, and rejected paths through Iteration 33.
- PGO is conclusively bad: 285.726ms vs 198.628ms (+43.85%). Native CPU and
  Fat-LTO have no proven win. Retain `znver5` and Thin-LTO.
- Hardware-profile runner: `scripts/run_hardware_profile.py`, committed as
  `b326a9c`; it records exact source, binary, model and image hashes and
  delays counters past loading/warm-up. Iteration-38 raw files remain on the
  Workhorse under `/tmp/da3-cpu-f32-{expanded,no-}workspace-20260813/`.

## Do not

- Do not claim speed from a smoke, a microbenchmark, quantization, or one
  lucky median. Use alternating trials and the locked F32 contract.
- Do not enable opt-in experimental environment switches by default without
  four-image parity and a stable A/B win.
- Do not reset or bulk-clean the worktree. It is intentionally shared and
  dirty with unrelated floorplan/UI work and prior benchmark changes.
- Do not use C++/ggml FFI as a Rust performance win; the requested result is
  a real Rust implementation.
