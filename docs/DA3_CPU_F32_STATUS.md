# DA3-BASE CPU-F32: architecture, evidence and continuation guide

## Who this is for

This document is for the engineer resuming the Rust DA3 performance work on
the Ryzen 9 9950X workhorse. After reading it, they should be able to run the
fair comparison, understand which runtime path is authoritative, and choose
the next experiment without repeating rejected work or weakening the fidelity
contract.

## Canonical public result

The conservative release baseline was generated on 2026-08-13 with twenty
randomized process trials per arm and ten timed iterations per process:

| Arm | Mean of trial medians | 95% CI | Peak RSS |
|---|---:|---:|---:|
| Rust DA3-BASE F32 | 171.141 ms | 168.042–174.241 ms | 804.0 MiB |
| C++/ggml DA3-BASE F32 | 238.789 ms | 237.406–240.172 ms | 618.2 MiB |

Rust therefore delivers 39.5% higher throughput than this C++/ggml reference.
Equivalently, its mean latency is 28.3% lower. These two percentage conventions
must never be mixed.

### Later exploratory result

After the canonical run, iteration 55 evaluated a packed Flash-attention
candidate with ten randomized process trials per arm:

| Arm | Mean of trial medians | 95% CI | Peak RSS |
|---|---:|---:|---:|
| Rust DA3-BASE F32 | 165.751 ms | 161.038–170.464 ms | 804.1 MiB |
| C++/ggml DA3-BASE F32 | 238.647 ms | 236.902–240.392 ms | 618.5 MiB |

That later run corresponds to 44.0% higher throughput and 30.5% lower
latency. It is preserved as serious optimization evidence, but its smaller
sample and incomplete source-commit capture make it exploratory rather than
the public release headline.

### Hardware and software fingerprint

| Component | Measured environment |
|---|---|
| CPU | AMD Ryzen 9 9950X, 16 physical cores / 32 logical CPUs |
| CPU settings used | 16 benchmark threads; boost enabled; AVX-512 available |
| Memory | 96 GiB installed (91 GiB reported available to the host) |
| GPU | NVIDIA GeForce RTX 5080, 16,303 MiB VRAM, driver 610.43.03 |
| Operating system | Linux 7.1.5-ogc5.1.fc44.x86_64 |
| Rust compiler | rustc 1.97.1 (2026-07-14) |
| Code generation | `target-cpu=znver5`, release profile, thin LTO, one codegen unit |

The GPU is installed in the workhorse but is not used for the CPU-F32 result.
The CPU has two L3-cache CCDs and one NUMA node; benchmark trials are not
claimed to be a universal result for every Ryzen, operating system or CPU
affinity policy.

The canonical evidence is the
[N=20 summary](benchmarks/2026-08-workhorse/2026-08-13-revalidation-cpu-f32-blis-n20/RESULTS.md)
and its
[raw trials](benchmarks/2026-08-workhorse/2026-08-13-revalidation-cpu-f32-blis-n20/raw-results.json).
The later candidate has a separate
[iteration-55 summary](benchmarks/2026-08-workhorse/iteration55-packed-flash/RESULTS.md)
and [raw trials](benchmarks/2026-08-workhorse/iteration55-packed-flash/raw-results.json).
Record binary and source-tree hashes alongside any future rerun before
presenting a new number outside this repository. The iteration-55 runner could
not resolve Git commits from its workhorse snapshot; that provenance limitation
must not be hidden.

### Fidelity gate

Each Rust output was compared with C++ F32 at 504×336. The required threshold
is Pearson r ≥ 0.9999 and MAE ≤ 0.005 on every image.

| Image | Pearson r | MAE |
|---|---:|---:|
| canyon | 0.999993628 | 0.001812542 |
| desk | 0.999978257 | 0.001772926 |
| mountains | 0.999985578 | 0.003675167 |
| street | 0.999972124 | 0.000821043 |

All four pass. This is implementation fidelity, not a depth-accuracy claim
against ground truth.

## What the production path does

The Rust implementation owns model loading, preprocessing, ViT backbone, DPT
depth/confidence head and postprocessing. An independently versioned local
kernel crate supplies the DA3-BASE fixed-shape CPU kernels.

The performance-critical execution decisions are:

1. **Locked workload.** The direct comparison is DA3-BASE F32 at 504×336,
   one decoded image, 16 CPU threads, one warm-up and ten timed iterations.
   Model loading and image decode are out of the timed region.
2. **Backbone projections.** The accepted experimental workhorse build routes
   selected fixed DA3 projection shapes through a Zen-oriented BLIS SGEMM
   bridge. It is deliberately explicit rather than the portable default.
   Direct QKV stays on the specialized native kernel because the BLIS QKV
   experiment regressed end-to-end latency.
3. **DPT head.** The large serial 1×1 head projections use the same BLIS
   bridge. Spatial 3×3 work uses the prepared F(2,3×3) Winograd route; the
   final resize and output convolution are fused. Reintroducing a materialized
   final resize is known to be slower.
4. **Flash attention.** The production attention path uses eight query rows
   and two 32-output AVX-512 panels. For complete eight-query tiles, it uses a
   dedicated persistent-K kernel that omits generic diagnostic fallback
   scratch. Its result is bit-identical to the prior QT8 kernel. The final
   short query tile remains on the generic path.
5. **Memory.** The head has an accepted safe workspace for buffers that the
   immediately following operation fully overwrites. It is not a whole-model
   graph arena and must not be described as one.

## Why this architecture was chosen

The original eager Rust path spent most of its time rebuilding mini graphs,
planning arenas, copying weights and allocating temporary activation buffers.
Removing that overhead and specializing the true fixed DA3-BASE shapes changed
the bottleneck from framework overhead to GEMM, Winograd and attention math.

The current design intentionally keeps these boundaries explicit. It makes
numeric parity inspectable and prevents an apparent speedup obtained by
silently dropping work, changing precision, or changing the resize policy.

The external kernel crate is a deliberate seam: architecture-level Rust code
owns model semantics and orchestration; fixed-shape AVX-512 kernels can evolve
independently with their own bitwise oracles.

## Accepted decisions

| Decision | Evidence | Consequence |
|---|---|---|
| Compare CPU F32 only for direct language/runtime claims | identical F32 GGUF model and input contract | quantized C++ results are descriptive only |
| Use 16 threads for both arms | locked benchmark contract | never change threads to improve a headline |
| Keep BLIS bridges explicit | qualified whole-model studies beat the earlier native fallback | portable builds may have different speed |
| Keep native direct QKV | BLIS QKV regressed by about 2.8% in smoke A/B | no QKV staging or BLIS transpose handoff |
| Keep fused final resize + F2 Winograd | materialized route lost a full study | avoid 41 MiB final resize materialization |
| Keep packed QT8 Flash as a candidate | exploratory 10-trial study reduced Rust to 165.751 ms | revalidate at N=20 before promoting its number to the release headline |
| Require parity before qualifying speed | four-image C++ F32 gate | faster incorrect output is rejected |

## Rejected paths worth not repeating blindly

- Full attention score matrices with Faer GEMM: about 30% slower than online
  Flash at the real shape.
- Larger Flash scheduler or tile experiments: flat, superblock and GGML64
  variants did not show a stable win. A 6×64 kernel was slower through register
  pressure.
- BLIS QKV: layout staging and head-major conversion outweighed the SGEMM win.
- OpenBLAS as a generic substitute: no material whole-model gain.
- F(4,3×3) Winograd: transform and scheduling costs dominated.
- Algebraic stage composition, direct attention-HND projection and residual-add
  fusion: all removed apparently useful buffers but regressed the whole model.
- PGO, whole-process affinity and compiler-target tuning: no reliable gain;
  some variants were materially slower.

The rationale and raw smoke/full-study values live in the optimization ledger.
Do not reinterpret a rejected microbenchmark as an end-to-end opportunity
without new profiling evidence.

## Reproducing the canonical comparison

Run only on a quiet workhorse. Stop other CPU-intensive jobs first; thermal
and scheduler interference widened Rust variance substantially during this
work.

The workhorse build uses the DA3 BLIS bridge and a Zen 5 code-generation
target:

```bash
cd /var/roothome/da3-bench/depth-anything-rs
export LD_LIBRARY_PATH=/tmp/da3-blis-install/lib:/tmp/da3-clang64/usr/lib64
RUSTFLAGS="--cfg da3_blis -L native=/tmp/da3-blis-install/lib -C target-cpu=znver5" \
  cargo build --release -p da-cli
```

The fair study is then:

```bash
env \
  LD_LIBRARY_PATH=/tmp/da3-blis-install/lib:/tmp/da3-clang64/usr/lib64 \
  DA3_KERNELS_BLIS_LINEAR=1 \
  DA3_HEAD_BLIS_GEMM=1 \
  DA3_BENCH_ROOT=/var/roothome/da3-bench \
  DA3_BENCH_IMAGE=/var/roothome/da3-bench/src/depth-anything.cpp/assets/samples/mountains.jpg \
  RAYON_NUM_THREADS=16 \
  RUSTFLAGS="--cfg da3_blis -L native=/tmp/da3-blis-install/lib -C target-cpu=znver5" \
  python3 scripts/run_scientific_benchmark.py \
    --cpu-f32-direct --trials 20 --repeat 10 --threads 16 --cooldown 3 \
    --seed 20260812 --output /tmp/da3-cpu-f32-<label>
```

Run the four-image PFM gate before accepting a candidate. The C++ F32 output is
the reference; compare one output per canyon, desk, mountains and street. The
benchmark contract gives the exact scope, statistics and exclusions. A
10-trial run remains suitable for candidate evaluation, but only the N=20
revalidation is the canonical public number.

## Safe next steps

If performance work resumes, begin by reproducing the 171.141 ms canonical
bundle on an idle host, preserving the same model, input, thread budget and
timed boundary. Then revalidate the 165.751 ms iteration-55 candidate at N=20
before considering a headline update.

Only then choose a new experiment from a profile. The most plausible remaining
area is online Flash softmax/accumulator work, but its upside is unproven and
previous attempts at scheduling, full scores and simple rescale fusion did not
qualify. Any new candidate must have:

1. a written hypothesis and one isolated change;
2. a local and target-hardware correctness oracle;
3. at least ten alternating same-binary smoke pairs with a quiet-host record;
4. four-image C++-F32 fidelity results if the smoke is meaningfully faster;
5. a randomized ten-trial full study before becoming the default.

## Source attribution and limits

Depth Anything 3, the DA3-BASE weights, depth-anything.cpp, and ggml are
third-party work. This repository does not distribute model weights. The
specific DA3-BASE checkpoint used by this study is Apache-2.0; that statement
does not cover every DA3 checkpoint. This project contributes the Rust path,
the kernel work, the benchmark infrastructure, parity gates and the documented
analysis. See [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) for exact
source and license provenance. The result does not generalize automatically to
other models, image resolutions, CPUs, compilers, operating systems or GPU
inference.
