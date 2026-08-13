//! The DPT (Dense Prediction Transformer) decoder head: takes the 4 backbone
//! out-layer features (`feat_5/7/9/11` from Task 17's `Backbone::forward`,
//! each `[n_patch, C]` token-major) and produces a dense depth map (`exp`
//! activation) plus a confidence map (`exp(x)+1`, "expp1").
//!
//! Ported line-for-line from the real C++ reference —
//! `../src/dpt_head.cpp::DptHead::build_depth_graph` (the shared graph
//! builder for depth/conf, aux-ray, and the metric-relative variants) and
//! `../src/dpt_blocks.cpp` (`conv2d`/`conv_transpose2d_p0`/
//! `residual_conv_unit`/`feature_fusion`/`interp_bilinear_ac` helpers) —
//! read directly during this task's investigation, not reverse-engineered.
//!
//! ## Design choice: direct `vestra_kernels` calls, not a `da_graph::Op`/`Plan` mini-graph
//!
//! Task 17's `vit_block` module composes via small single-purpose
//! `da_graph::Plan`s (documented there as a known perf escape hatch, not a
//! blocker). This module goes one step further and skips `da_graph`
//! entirely, calling `vestra_kernels::conv2d`/`conv_transpose2d`/
//! `bilinear_resize_align_corners`/`scalar::*` directly:
//!
//! - `da_graph::Op` has no `ConvTranspose2d` or align-corners-resize variant
//!   yet (only `Op::Conv2d`, matching Task 17's Attention/LayerNorm/Gemm
//!   needs) — adding two new op variants (with their `CpuBackend::execute`
//!   arms, arena-lifetime plumbing, and aliasing-safety comments) purely to
//!   route this task's math through the graph, when nothing else in this
//!   task needs a *compiled, reusable* multi-call graph (this head runs
//!   once per image, not once per token/layer like `vit_block`), is scope
//!   the task brief explicitly leaves to this module's judgment.
//! - The fusion pyramid (4 stages x reassemble/resize, 4 refinenet fusion
//!   blocks, 2 output convs) is a fixed, non-looped sequence of ~30 conv/
//!   resize/elementwise calls on host-owned `Vec<f32>` buffers — a `Plan`
//!   buys nothing here that a straight-line function doesn't already give
//!   (no repeated re-execution at the same shape within one `dpt_head`
//!   call).
//!
//! If a future task needs this head to run inside a single fused `da_graph`
//! `Plan` alongside the backbone (e.g. for arena-reuse across a whole
//! forward pass), extending `Op` with `ConvTranspose2d`/`ResizeBilinearAc`
//! variants mirroring `Op::Conv2d`'s pattern is the natural next step.
//!
//! ## Honesty note
//!
//! NOT numerically cross-checked against the `head_stage{0..3}`/
//! `head_fused`/`head_depth`/`head_depth_conf` reference dumps in this
//! environment (no `../dumps/reference.gguf` present) — `tests/dpt_parity.rs`
//! skips cleanly until that dump exists. The math is transcribed directly
//! from the C++ source (see per-function doc comments for the
//! line-by-line correspondence), so confidence is high on structure and the
//! two "hard rules" (align_corners resize, `expp1` activation), but full
//! numerical parity end-to-end is UNVERIFIED here. Two explicit, documented
//! assumptions (not directly verifiable without a real GGUF/dumps in this
//! environment):
//! - **`head.out_channels` GGUF override not implemented**: the real model
//!   config can override the per-stage projection channel counts via a
//!   `depthanything3.head.out_channels` array-of-4-u32 KV
//!   (`include/da_gguf_keys.h`); this module always uses the DA3-BASE
//!   default `[96, 192, 384, 768]` (matching `../src/dpt_head.cpp`'s own
//!   default when that KV is absent, `int oc[4] = {96,192,384,768};`).
//!   Adding a `ModelConfig` field for this one head-only override was
//!   judged out of proportion to this task's scope; if a real GGUF ever
//!   sets a non-default `head.out_channels`, this module would silently use
//!   the wrong per-stage channel counts (weight-shape asserts would panic
//!   loudly rather than corrupt output, since the loaded projection weight
//!   tensors would then mismatch the hardcoded `oc[]`).
//!
//! ## Weight tensor names (verified against the real GGUF converter)
//!
//! Under the `head.*` prefix: `norm.weight`/`.bias` (optional, presence-gated),
//! `proj.{0..3}.weight`/`.bias`, `resize.{0..3}.weight`/`.bias`,
//! `scratch.layer{1..4}_rn.weight` (no bias), `scratch.rn{1..4}.rc1.c{1,2}.weight`/`.bias`
//! (rc1 absent/unused for `rn4` — no lateral there), `scratch.rn{1..4}.rc2.c{1,2}.weight`/`.bias`,
//! `scratch.rn{1..4}.out.weight`/`.bias`, `scratch.out1.weight`/`.bias`,
//! `scratch.out2a.weight`/`.bias`, `scratch.out2b.weight`/`.bias`.

use da_graph::Weights;
use rayon::prelude::*;
use std::cell::RefCell;
use std::collections::HashMap;
use vestra_kernels::conv::{
    conv2d, conv3x3_winograd_f2_prepared, conv3x3_winograd_f2_prepared_relu_input,
    conv3x3_winograd_f2_prepared_resize_align_corners, conv3x3_winograd_f4_prepared,
    conv_transpose2d, prepare_winograd_f2_filter, prepare_winograd_f4_filter, WinogradF2Filter,
    WinogradF4Filter,
};
use vestra_kernels::gemm::{BlisOrFaerGemm, Gemm};
use vestra_kernels::{bilinear_resize_align_corners, scalar, Kernels};

use crate::uv_embed::UvEmbedCache;
use crate::ModelConfig;

/// `head.norm` LayerNorm epsilon. **Not** `cfg.ln_eps`: the C++ reference
/// hardcodes this to torch's default `nn.LayerNorm` eps
/// (`../src/dpt_head.cpp`: `const float eps = 1e-5f; // head.norm is
/// nn.LayerNorm default`), the same "torch-default, not the block eps" trap
/// documented for `vit_block::QK_NORM_EPS`.
pub const HEAD_NORM_EPS: f32 = 1e-5;

/// Default per-stage projection output channel counts (DA3-BASE), matching
/// `../src/dpt_head.cpp`'s `int oc[4] = {96,192,384,768};` default — see
/// module doc comment's "`head.out_channels` GGUF override not implemented"
/// note.
const DEFAULT_OC: [usize; 4] = [96, 192, 384, 768];

/// Fixed channel width of every `layer{i}_rn`/refinenet fusion stage.
/// Matches the C++ reference's `head.scratch.*` tensor shapes (all
/// `layer{i}_rn`/`rn{i}.rc*`/`rn{i}.out` convs are 128-channel).
const FUSION_C: usize = 128;

/// Per-`Engine` cache for the model's immutable 3x3 DPT filters. Keeping it
/// here binds transformed filters to the owning loaded model, avoiding both
/// per-image content hashing and unsafe global pointer identity.
#[derive(Default)]
pub struct WinogradFilterCache {
    filters: HashMap<String, WinogradF2Filter>,
    f4_filters: HashMap<String, WinogradF4Filter>,
}

impl WinogradFilterCache {
    pub fn new() -> Self {
        Self::default()
    }

    fn get_or_prepare(
        &mut self,
        weights: &Weights,
        name: &str,
        in_c: usize,
        out_c: usize,
    ) -> &WinogradF2Filter {
        self.filters
            .entry(name.to_owned())
            .or_insert_with(|| prepare_winograd_f2_filter(get_weight(weights, name), in_c, out_c))
    }

    fn get_or_prepare_f4(
        &mut self,
        weights: &Weights,
        name: &str,
        in_c: usize,
        out_c: usize,
    ) -> &WinogradF4Filter {
        self.f4_filters
            .entry(name.to_owned())
            .or_insert_with(|| prepare_winograd_f4_filter(get_weight(weights, name), in_c, out_c))
    }
}

/// Per-engine pool for DPT activation buffers that are wholly overwritten by
/// a following kernel.  The production 504×336 head creates many large
/// temporary NCHW maps; returning their backing allocations after each
/// inference avoids faulting fresh pages again on the next video frame.
///
/// This is intentionally not a general arena: externally visible depth and
/// confidence maps never enter it, debug captures keep their owned buffers,
/// and a buffer may only be obtained through [`take_overwrite`] when the
/// caller's next operation initializes every element before reading it.
#[derive(Default)]
pub struct HeadWorkspace {
    free: RefCell<Vec<Vec<f32>>>,
}

impl HeadWorkspace {
    pub fn new() -> Self {
        Self::default()
    }

    fn take_overwrite(&self, len: usize) -> Vec<f32> {
        let mut free = self.free.borrow_mut();
        let index = free.iter().position(|buffer| buffer.capacity() >= len);
        let mut buffer = index
            .map(|index| free.swap_remove(index))
            .unwrap_or_else(|| Vec::with_capacity(len));
        // SAFETY: this private method is used only for output buffers of
        // kernels that overwrite the full slice before any read.  Keeping the
        // existing allocation avoids both allocator work and page faults.
        unsafe { buffer.set_len(len) };
        buffer
    }

    fn recycle(&self, mut buffer: Vec<f32>) {
        buffer.clear();
        self.free.borrow_mut().push(buffer);
    }
}

fn overwrite_buffer(workspace: Option<&HeadWorkspace>, len: usize) -> Vec<f32> {
    workspace.map_or_else(|| vec![0.0; len], |workspace| workspace.take_overwrite(len))
}

fn recycle_buffer(workspace: Option<&HeadWorkspace>, buffer: Vec<f32>) {
    if let Some(workspace) = workspace {
        workspace.recycle(buffer);
    }
}

/// Output of [`dpt_head`]: dense depth (`exp(logit)`) and confidence
/// (`exp(logit)+1`, "expp1") maps, each `[h*w]` row-major, at the requested
/// `(h, w)` pixel resolution.
pub struct DepthOut {
    pub depth: Vec<f32>,
    /// Empty when the loaded `head.scratch.out2b.weight` has `output_dim <
    /// 2` (a single-channel, depth-only head variant — out of scope for
    /// this task's DA3-BASE target but handled gracefully rather than
    /// panicking).
    pub conf: Vec<f32>,
    pub h: usize,
    pub w: usize,
}

/// Debug/parity-capturable intermediates from a `dpt_head` call: the 4
/// post-resize `l[s]` stage tensors (`head_stage{0..3}` dump gate) and the
/// post-`output_conv1` fused tensor (`head_fused` dump gate, `[64,128,128]`
/// at the square-224 fixture). Returned alongside [`DepthOut`] by
/// [`dpt_head_debug`] so a future parity test can capture these exact
/// intermediates without this module needing to change.
pub struct DptDebug {
    /// `stages[s]` is `[oc[s], stage_h[s], stage_w[s]]` CHW, `s` in stage
    /// order (0..3), post-resize (post-`Identity` for `s==2`).
    pub stages: [Vec<f32>; 4],
    /// `[64, 128, 128]` CHW (at the square-224/patch14 fixture — generally
    /// `[64, 8*grid, 8*grid]`), post-`output_conv1`, pre-final-upsample.
    pub fused: Vec<f32>,
}

fn relu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = v.max(0.0);
    }
}

/// Reshapes a `[n_tok, c]` token-major buffer (token order row-major over a
/// `grid_h x grid_w` patch grid) into `[c, grid_h, grid_w]`
/// CHW.
fn tok_major_to_chw(tok: &[f32], n_tok: usize, c: usize, grid_h: usize, grid_w: usize) -> Vec<f32> {
    debug_assert_eq!(tok.len(), n_tok * c);
    debug_assert_eq!(n_tok, grid_h * grid_w);
    let mut out = vec![0f32; c * grid_h * grid_w];
    for t in 0..n_tok {
        let row = t / grid_w;
        let col = t % grid_w;
        for ch in 0..c {
            out[ch * grid_h * grid_w + row * grid_w + col] = tok[t * c + ch];
        }
    }
    out
}

/// Applies the optional head LayerNorm while changing token-major storage to
/// CHW.  This is an ownership-free handoff from the backbone captures: the
/// established path first clones every `[token, channel]` feature, normalizes
/// it in place, then makes a second CHW buffer.  Each worker here owns one
/// input token and writes its disjoint channel slots, retaining exactly the
/// scalar LayerNorm reduction and per-channel arithmetic used by
/// `scalar::layernorm`.
fn token_major_to_chw_with_optional_layernorm(
    tok: &[f32],
    n_tok: usize,
    c: usize,
    grid_h: usize,
    grid_w: usize,
    norm: Option<(&[f32], &[f32])>,
) -> Vec<f32> {
    debug_assert_eq!(tok.len(), n_tok * c);
    debug_assert_eq!(n_tok, grid_h * grid_w);
    let mut out = vec![0.0f32; tok.len()];
    let out_ptr = out.as_mut_ptr() as usize;
    tok.par_chunks(c).enumerate().for_each(|(t, row)| {
        let (mean, inv) = if let Some((gamma, beta)) = norm {
            debug_assert_eq!(gamma.len(), c);
            debug_assert_eq!(beta.len(), c);
            let mean = row.iter().sum::<f32>() / c as f32;
            let variance = row
                .iter()
                .map(|value| {
                    let delta = value - mean;
                    delta * delta
                })
                .sum::<f32>()
                / c as f32;
            (mean, 1.0 / (variance + HEAD_NORM_EPS).sqrt())
        } else {
            (0.0, 0.0)
        };
        for ch in 0..c {
            let value = if let Some((gamma, beta)) = norm {
                (row[ch] - mean) * inv * gamma[ch] + beta[ch]
            } else {
                row[ch]
            };
            // SAFETY: each `(token, channel)` maps to one unique CHW output
            // element. Rayon supplies each source row to exactly one worker.
            unsafe { *((out_ptr as *mut f32).add(ch * n_tok + t)) = value };
        }
    });
    out
}

fn get_weight<'a>(weights: &'a Weights, name: &str) -> &'a [f32] {
    weights
        .get_f32(name)
        .unwrap_or_else(|| panic!("missing weight tensor {name:?}"))
}

/// `out = relu(x); out = conv3x3pad1(out, c1w, c1b); out = relu(out); out =
/// conv3x3pad1(out, c2w, c2b); return out + x`. A residual block, channel
/// count preserved (`c` in, `c` out). Ported from
/// `../src/dpt_blocks.cpp::residual_conv_unit`.
#[allow(clippy::too_many_arguments)]
fn residual_conv_unit(
    x: &[f32],
    c: usize,
    h: usize,
    w: usize,
    c1: &WinogradF2Filter,
    c1b: &[f32],
    c2: &WinogradF2Filter,
    c2b: &[f32],
    workspace: Option<&HeadWorkspace>,
) -> Vec<f32> {
    let mut o1 = overwrite_buffer(workspace, c * h * w);
    conv3x3_winograd_f2_prepared_relu_input(x, c, h, w, c1, c, Some(c1b), &mut o1);
    let mut o2 = overwrite_buffer(workspace, c * h * w);
    conv3x3_winograd_f2_prepared_relu_input(&o1, c, h, w, c2, c, Some(c2b), &mut o2);
    Kernels::detect().add(&mut o2, x);
    recycle_buffer(workspace, o1);
    o2
}

/// RefineNet-style fusion of a `top` feature map (from the deeper stage)
/// with an optional `lateral` skip connection, resized to `(target_h,
/// target_w)`, projected back to `FUSION_C` channels. Ported from
/// `../src/dpt_blocks.cpp::feature_fusion`.
///
/// `top` and `lateral` (when present) are both `[FUSION_C, th, tw]` CHW at
/// the SAME spatial size `(th, tw)` — by construction of this module's
/// fusion chain (see [`dpt_head_debug`]'s call sites), never independently
/// varying.
#[allow(clippy::too_many_arguments)]
fn feature_fusion(
    top: &[f32],
    th: usize,
    tw: usize,
    lateral: Option<&[f32]>,
    rc1: Option<(&WinogradF2Filter, &[f32], &WinogradF2Filter, &[f32])>,
    rc2: (&WinogradF2Filter, &[f32], &WinogradF2Filter, &[f32]),
    out_w: &[f32],
    out_b: &[f32],
    target_h: usize,
    target_w: usize,
    gemm: &impl Gemm,
    workspace: Option<&HeadWorkspace>,
) -> Vec<f32> {
    let c = FUSION_C;
    let profile = std::env::var_os("DA_HEAD_PROFILE").is_some();
    debug_assert_eq!(top.len(), c * th * tw);

    let mut y = overwrite_buffer(workspace, c * th * tw);
    y.copy_from_slice(top);
    let rc1_started = std::time::Instant::now();
    if let (Some(lat), Some((c1, c1b, c2, c2b))) = (lateral, rc1) {
        debug_assert_eq!(lat.len(), c * th * tw);
        let res = residual_conv_unit(lat, c, th, tw, c1, c1b, c2, c2b, workspace);
        Kernels::detect().add(&mut y, &res);
        recycle_buffer(workspace, res);
    }
    let rc1_elapsed = rc1_started.elapsed();
    let (r2c1, r2c1b, r2c2, r2c2b) = rc2;
    let rc2_started = std::time::Instant::now();
    let refined = residual_conv_unit(&y, c, th, tw, r2c1, r2c1b, r2c2, r2c2b, workspace);
    recycle_buffer(workspace, y);
    let y = refined;
    let rc2_elapsed = rc2_started.elapsed();

    let mut out = overwrite_buffer(workspace, c * target_h * target_w);
    let out_started = std::time::Instant::now();
    let resize_started = std::time::Instant::now();
    let mut resized = overwrite_buffer(workspace, c * target_h * target_w);
    bilinear_resize_align_corners(&y, c, th, tw, target_h, target_w, &mut resized);
    conv2d(
        &resized,
        c,
        target_h,
        target_w,
        out_w,
        c,
        1,
        1,
        1,
        0,
        Some(out_b),
        gemm,
        &mut out,
    );
    recycle_buffer(workspace, resized);
    let resize_elapsed = resize_started.elapsed();
    if profile {
        eprintln!(
            "phase: head fusion_detail {th}x{tw}->{target_h}x{target_w} rc1={:.3}ms rc2={:.3}ms resize={:.3}ms out1x1={:.3}ms",
            rc1_elapsed.as_secs_f64() * 1e3,
            rc2_elapsed.as_secs_f64() * 1e3,
            resize_elapsed.as_secs_f64() * 1e3,
            out_started.elapsed().as_secs_f64() * 1e3,
        );
    }
    out
}

/// Runs the full DPT depth+confidence head, also returning the
/// dump-gate-capturable intermediates ([`DptDebug`]) alongside the final
/// [`DepthOut`]. [`dpt_head`] is a thin wrapper discarding the debug output.
///
/// - `feats[s]` (`s` in `0..4`, matching `cfg.out_layers` order) is
///   `[n_patch, C]` token-major (`C = 2*embed_dim` if `cfg.cat_token` else
///   `embed_dim`, matching `BackboneOutputs.feats`'s documented layout).
/// - `(h, w)` is the target *pixel* resolution the final depth/conf maps
///   are upsampled to (the original preprocessed image's `H`/`W`).
/// - `cache` memoizes the (input-independent, geometry-only) UV positional
///   embeddings across calls at the same resolution — see
///   `uv_embed::UvEmbedCache`'s doc comment.
///
/// # Panics
/// - If any `feats[s].len()` isn't a multiple of the expected channel count
///   or does not match the patch grid derived from `(h, w)`.
/// - If a required `head.*` weight tensor is missing from `weights`.
fn dpt_head_impl(
    feats: &[Vec<f32>],
    h: usize,
    w: usize,
    cfg: &ModelConfig,
    weights: &Weights,
    cache: &mut UvEmbedCache,
    wino_cache: &mut WinogradFilterCache,
    workspace: Option<&HeadWorkspace>,
    capture_debug: bool,
) -> (DepthOut, Option<DptDebug>) {
    let head_profile = std::env::var_os("DA_HEAD_PROFILE").is_some();
    let head_started = std::time::Instant::now();
    assert_eq!(
        feats.len(),
        4,
        "dpt_head expects exactly 4 out-layer feats (feat_5/7/9/11)"
    );
    let gemm = BlisOrFaerGemm;

    let c_in = if cfg.cat_token {
        2 * cfg.embed_dim as usize
    } else {
        cfg.embed_dim as usize
    };
    let oc = DEFAULT_OC;

    let has_head_norm = weights.get_f32("head.norm.weight").is_some();
    let norm_w = has_head_norm.then(|| get_weight(weights, "head.norm.weight"));
    let norm_b = has_head_norm.then(|| get_weight(weights, "head.norm.bias"));

    let patch = cfg.patch_size as usize;
    assert!(patch > 0, "patch_size must be non-zero");
    assert_eq!(h % patch, 0, "target height must be patch-aligned");
    assert_eq!(w % patch, 0, "target width must be patch-aligned");
    let grid_h = h / patch;
    let grid_w = w / patch;

    // All four out-layer features share this rectangular patch grid.
    let n_tokens0 = {
        let c = c_in;
        assert_eq!(
            feats[0].len() % c,
            0,
            "feats[0] length not a multiple of channel count {c}"
        );
        feats[0].len() / c
    };
    assert_eq!(
        n_tokens0,
        grid_h * grid_w,
        "feats[0] token count does not match {grid_h}x{grid_w} patch grid"
    );
    // The C++ reference always uses the full target image's pixel aspect
    // ratio for the UV embedding, even at the (smaller, and generally
    // differently-shaped for non-square images) patch-grid resolution used
    // by the per-stage `projects[s]` additions below — see
    // `uv_embed::build_uv_embed_chw`'s doc comment.
    let target_aspect = w as f32 / h as f32;

    // ---- Stage 0..3: reassemble (LN -> reshape -> project -> +UV -> resize) ----
    let stages_started = std::time::Instant::now();
    let mut l: [Vec<f32>; 4] = Default::default();
    let mut l_hw: [(usize, usize); 4] = [(0, 0); 4];
    for s in 0..4 {
        let n_tok = feats[s].len() / c_in;
        assert_eq!(
            n_tok,
            grid_h * grid_w,
            "feats[{s}] token count {n_tok} != patch grid size {}",
            grid_h * grid_w
        );

        let x_chw = if std::env::var_os("DA3_FUSE_HEAD_TOKEN_NORM_CHW").is_some() {
            token_major_to_chw_with_optional_layernorm(
                &feats[s],
                n_tok,
                c_in,
                grid_h,
                grid_w,
                norm_w.zip(norm_b),
            )
        } else {
            let mut tok = feats[s].clone();
            if let (Some(g), Some(b)) = (norm_w, norm_b) {
                scalar::layernorm(&mut tok, n_tok, c_in, g, b, HEAD_NORM_EPS);
            }
            tok_major_to_chw(&tok, n_tok, c_in, grid_h, grid_w)
        };

        // projects[s]: 1x1 conv, c_in -> oc[s]
        let pw_name = format!("head.proj.{s}.weight");
        let pb_name = format!("head.proj.{s}.bias");
        let pweight = get_weight(weights, &pw_name);
        let pbias = get_weight(weights, &pb_name);
        let mut projected = overwrite_buffer(workspace, oc[s] * grid_h * grid_w);
        conv2d(
            &x_chw,
            c_in,
            grid_h,
            grid_w,
            pweight,
            oc[s],
            1,
            1,
            1,
            0,
            Some(pbias),
            &gemm,
            &mut projected,
        );

        if cfg.head_pos_embed {
            let uv = cache.get_or_build_with_aspect(grid_h, grid_w, oc[s], target_aspect);
            debug_assert_eq!(uv.len(), projected.len());
            Kernels::detect().add(&mut projected, &uv);
        }

        // resize_layers[s]
        let (x, oh, ow) = match s {
            0 => {
                let rw = get_weight(weights, "head.resize.0.weight");
                let rb = get_weight(weights, "head.resize.0.bias");
                let oh = (grid_h - 1) * 4 + 4;
                let ow = (grid_w - 1) * 4 + 4;
                let mut out = overwrite_buffer(workspace, oc[0] * oh * ow);
                conv_transpose2d(
                    &projected,
                    oc[0],
                    grid_h,
                    grid_w,
                    rw,
                    oc[0],
                    4,
                    4,
                    4,
                    Some(rb),
                    &mut out,
                );
                recycle_buffer(workspace, projected);
                (out, oh, ow)
            }
            1 => {
                let rw = get_weight(weights, "head.resize.1.weight");
                let rb = get_weight(weights, "head.resize.1.bias");
                let oh = (grid_h - 1) * 2 + 2;
                let ow = (grid_w - 1) * 2 + 2;
                let mut out = overwrite_buffer(workspace, oc[1] * oh * ow);
                conv_transpose2d(
                    &projected,
                    oc[1],
                    grid_h,
                    grid_w,
                    rw,
                    oc[1],
                    2,
                    2,
                    2,
                    Some(rb),
                    &mut out,
                );
                recycle_buffer(workspace, projected);
                (out, oh, ow)
            }
            2 => (projected, grid_h, grid_w), // Identity
            3 => {
                let rw = get_weight(weights, "head.resize.3.weight");
                let rb = get_weight(weights, "head.resize.3.bias");
                let oh = (grid_h + 2 - 3) / 2 + 1;
                let ow = (grid_w + 2 - 3) / 2 + 1;
                let mut out = overwrite_buffer(workspace, oc[3] * oh * ow);
                conv2d(
                    &projected,
                    oc[3],
                    grid_h,
                    grid_w,
                    rw,
                    oc[3],
                    3,
                    3,
                    2,
                    1,
                    Some(rb),
                    &gemm,
                    &mut out,
                );
                recycle_buffer(workspace, projected);
                (out, oh, ow)
            }
            _ => unreachable!(),
        };
        l_hw[s] = (oh, ow);
        l[s] = x;
    }
    if head_profile {
        eprintln!(
            "phase: head stages={:.3}ms",
            stages_started.elapsed().as_secs_f64() * 1e3,
        );
    }

    // ---- Lateral projections: layer{i}_rn (i=1..4 <-> stage s=0..3), 3x3 pad1, NO bias, -> 128ch ----
    let laterals_started = std::time::Instant::now();
    let mut l_rn: [Vec<f32>; 4] = Default::default();
    for s in 0..4 {
        let (gh, gw) = l_hw[s];
        let w_name = format!("head.scratch.layer{}_rn.weight", s + 1);
        let filter = wino_cache.get_or_prepare(weights, &w_name, oc[s], FUSION_C);
        let mut out = overwrite_buffer(workspace, FUSION_C * gh * gw);
        conv3x3_winograd_f2_prepared(&l[s], oc[s], gh, gw, filter, FUSION_C, None, &mut out);
        l_rn[s] = out;
        if !capture_debug {
            recycle_buffer(workspace, std::mem::take(&mut l[s]));
        }
    }
    // The production path no longer needs the large pre-lateral maps. Moving
    // them into an Option lets their allocation be returned before RefineNet
    // fusion starts, while the debug entry point retains the exact snapshots.
    let debug_stages = capture_debug.then_some(l);
    if head_profile {
        eprintln!(
            "phase: head laterals={:.3}ms",
            laterals_started.elapsed().as_secs_f64() * 1e3,
        );
    }

    // ---- Fusion chain: rn4 (deepest, no lateral) -> rn3 -> rn2 -> rn1 ----
    // These tensors are immutable model parameters.  Borrow them directly
    // instead of copying every RefineNet weight for every inference.
    for i in 1..=4 {
        for suffix in ["rc2.c1.weight", "rc2.c2.weight"] {
            let name = format!("head.scratch.rn{i}.{suffix}");
            wino_cache.get_or_prepare(weights, &name, FUSION_C, FUSION_C);
        }
        if i != 4 {
            for suffix in ["rc1.c1.weight", "rc1.c2.weight"] {
                let name = format!("head.scratch.rn{i}.{suffix}");
                wino_cache.get_or_prepare(weights, &name, FUSION_C, FUSION_C);
            }
        }
    }
    let rn_weights = |i: usize, suffix: &str| -> &[f32] {
        get_weight(weights, &format!("head.scratch.rn{i}.{suffix}"))
    };
    let rn_filter = |i: usize, suffix: &str| -> &WinogradF2Filter {
        wino_cache
            .filters
            .get(&format!("head.scratch.rn{i}.{suffix}"))
            .expect("prepared RefineNet filter missing")
    };
    let get4 = |i: usize| -> (&WinogradF2Filter, &[f32], &WinogradF2Filter, &[f32]) {
        (
            rn_filter(i, "rc1.c1.weight"),
            rn_weights(i, "rc1.c1.bias"),
            rn_filter(i, "rc1.c2.weight"),
            rn_weights(i, "rc1.c2.bias"),
        )
    };
    let get4_rc2 = |i: usize| -> (&WinogradF2Filter, &[f32], &WinogradF2Filter, &[f32]) {
        (
            rn_filter(i, "rc2.c1.weight"),
            rn_weights(i, "rc2.c1.bias"),
            rn_filter(i, "rc2.c2.weight"),
            rn_weights(i, "rc2.c2.bias"),
        )
    };

    let fusion_started = std::time::Instant::now();
    let (h3, w3) = l_hw[3];
    let (h2, w2) = l_hw[2];
    let (h1, w1) = l_hw[1];
    let (h0, w0) = l_hw[0];

    // rn4: top=l4_rn (stage3's layer4_rn), no lateral, target = stage2's (identity) spatial size (h2,w2).
    let rn4_rc2 = get4_rc2(4);
    let rn4_out_w = rn_weights(4, "out.weight");
    let rn4_out_b = rn_weights(4, "out.bias");
    let rn4_started = std::time::Instant::now();
    let mut out = feature_fusion(
        &l_rn[3], h3, w3, None, None, rn4_rc2, rn4_out_w, rn4_out_b, h2, w2, &gemm, workspace,
    );
    recycle_buffer(workspace, std::mem::take(&mut l_rn[3]));
    let rn4_elapsed = rn4_started.elapsed();
    // rn3: top=rn4's output (h2,w2), lateral=layer3_rn (stage2), target = stage1's spatial size (h1,w1).
    let rn3_rc1 = get4(3);
    let rn3_rc2 = get4_rc2(3);
    let rn3_out_w = rn_weights(3, "out.weight");
    let rn3_out_b = rn_weights(3, "out.bias");
    let rn3_started = std::time::Instant::now();
    let next = feature_fusion(
        &out,
        h2,
        w2,
        Some(&l_rn[2]),
        Some(rn3_rc1),
        rn3_rc2,
        rn3_out_w,
        rn3_out_b,
        h1,
        w1,
        &gemm,
        workspace,
    );
    recycle_buffer(workspace, out);
    recycle_buffer(workspace, std::mem::take(&mut l_rn[2]));
    out = next;
    let rn3_elapsed = rn3_started.elapsed();

    // rn2: top=rn3's output (h1,w1), lateral=layer2_rn (stage1), target = stage0's spatial size (h0,w0).
    let rn2_rc1 = get4(2);
    let rn2_rc2 = get4_rc2(2);
    let rn2_out_w = rn_weights(2, "out.weight");
    let rn2_out_b = rn_weights(2, "out.bias");
    let rn2_started = std::time::Instant::now();
    let next = feature_fusion(
        &out,
        h1,
        w1,
        Some(&l_rn[1]),
        Some(rn2_rc1),
        rn2_rc2,
        rn2_out_w,
        rn2_out_b,
        h0,
        w0,
        &gemm,
        workspace,
    );
    recycle_buffer(workspace, out);
    recycle_buffer(workspace, std::mem::take(&mut l_rn[1]));
    out = next;
    let rn2_elapsed = rn2_started.elapsed();

    // rn1: top=rn2's output (h0,w0), lateral=layer1_rn (stage0), target = 2x rn2's output spatial size.
    let rn1_rc1 = get4(1);
    let rn1_rc2 = get4_rc2(1);
    let rn1_out_w = rn_weights(1, "out.weight");
    let rn1_out_b = rn_weights(1, "out.bias");
    let rn1_started = std::time::Instant::now();
    let next = feature_fusion(
        &out,
        h0,
        w0,
        Some(&l_rn[0]),
        Some(rn1_rc1),
        rn1_rc2,
        rn1_out_w,
        rn1_out_b,
        2 * h0,
        2 * w0,
        &gemm,
        workspace,
    );
    recycle_buffer(workspace, out);
    recycle_buffer(workspace, std::mem::take(&mut l_rn[0]));
    let out = next;
    let rn1_elapsed = rn1_started.elapsed();
    let (fh, fw) = (2 * h0, 2 * w0);
    if head_profile {
        eprintln!(
            "phase: head fusion_ops rn4={:.3}ms rn3={:.3}ms rn2={:.3}ms rn1={:.3}ms",
            rn4_elapsed.as_secs_f64() * 1e3,
            rn3_elapsed.as_secs_f64() * 1e3,
            rn2_elapsed.as_secs_f64() * 1e3,
            rn1_elapsed.as_secs_f64() * 1e3,
        );
        eprintln!(
            "phase: head fusion={:.3}ms",
            fusion_started.elapsed().as_secs_f64() * 1e3,
        );
    }

    // ---- output_conv1: 3x3 pad1, 128 -> 64 (the head_fused dump gate) ----
    let output_started = std::time::Instant::now();
    let out1_b = get_weight(weights, "head.scratch.out1.bias");
    let feat_half = if cfg.head_features != 0 {
        cfg.head_features as usize / 2
    } else {
        64
    };
    let mut fused = overwrite_buffer(workspace, feat_half * fh * fw);
    let out1_started = std::time::Instant::now();
    let out1_filter =
        wino_cache.get_or_prepare(weights, "head.scratch.out1.weight", FUSION_C, feat_half);
    conv3x3_winograd_f2_prepared(
        &out,
        FUSION_C,
        fh,
        fw,
        out1_filter,
        feat_half,
        Some(out1_b),
        &mut fused,
    );
    recycle_buffer(workspace, out);
    let out1_elapsed = out1_started.elapsed();

    // ---- output_conv2: conv(feat_half->32, 3x3 pad1) -> relu -> conv(32->output_dim, 1x1 pad0) ----
    let out2a_b = get_weight(weights, "head.scratch.out2a.bias");
    let mut mid = overwrite_buffer(workspace, 32 * h * w);
    let out2a_started = std::time::Instant::now();
    let final_f4_wino = std::env::var_os("DA3_WINO_F4_FINAL").is_some()
        && (feat_half, fh, fw, h, w) == (64, 288, 192, 504, 336);
    let fused_final_resize_wino = std::env::var_os("DA3_DISABLE_FUSE_FINAL_RESIZE_WINO").is_none()
        // This candidate is deliberately exact-shape gated: DA3-BASE F32's
        // 504x336 output is a 64x288x192 feature map resized to 64x504x336.
        // Other geometries keep the established materialized route.
        && (feat_half, fh, fw, h, w) == (64, 288, 192, 504, 336);
    let resize_started = std::time::Instant::now();
    if final_f4_wino {
        // F(4) is deliberately benchmark-only: it keeps the materialized
        // resize so this experiment changes exactly one algorithmic choice.
        let mut feat_map = vec![0f32; feat_half * h * w];
        bilinear_resize_align_corners(&fused, feat_half, fh, fw, h, w, &mut feat_map);
        if cfg.head_pos_embed {
            let uv = cache.get_or_build(h, w, feat_half);
            Kernels::detect().add(&mut feat_map, &uv);
        }
        let filter =
            wino_cache.get_or_prepare_f4(weights, "head.scratch.out2a.weight", feat_half, 32);
        conv3x3_winograd_f4_prepared(
            &feat_map,
            feat_half,
            h,
            w,
            filter,
            32,
            Some(out2a_b),
            &mut mid,
        );
    } else if fused_final_resize_wino {
        let uv = cfg
            .head_pos_embed
            .then(|| cache.get_or_build(h, w, feat_half));
        conv3x3_winograd_f2_prepared_resize_align_corners(
            &fused,
            feat_half,
            fh,
            fw,
            h,
            w,
            uv,
            wino_cache.get_or_prepare(weights, "head.scratch.out2a.weight", feat_half, 32),
            32,
            Some(out2a_b),
            &mut mid,
        );
    } else {
        let mut feat_map = vec![0f32; feat_half * h * w];
        bilinear_resize_align_corners(&fused, feat_half, fh, fw, h, w, &mut feat_map);
        if cfg.head_pos_embed {
            let uv = cache.get_or_build(h, w, feat_half);
            debug_assert_eq!(uv.len(), feat_map.len());
            Kernels::detect().add(&mut feat_map, &uv);
        }
        conv3x3_winograd_f2_prepared(
            &feat_map,
            feat_half,
            h,
            w,
            wino_cache.get_or_prepare(weights, "head.scratch.out2a.weight", feat_half, 32),
            32,
            Some(out2a_b),
            &mut mid,
        );
    }
    let debug_fused = if capture_debug {
        Some(fused)
    } else {
        recycle_buffer(workspace, fused);
        None
    };
    let resize_elapsed = resize_started.elapsed();
    let out2a_elapsed = out2a_started.elapsed();
    relu_inplace(&mut mid);

    let out2b_w = get_weight(weights, "head.scratch.out2b.weight");
    let out2b_b = get_weight(weights, "head.scratch.out2b.bias");
    let output_dim = out2b_b.len();
    assert_eq!(
        out2b_w.len(),
        output_dim * 32,
        "head.scratch.out2b.weight unexpected size"
    );
    let mut logits = overwrite_buffer(workspace, output_dim * h * w);
    let out2b_started = std::time::Instant::now();
    conv2d(
        &mid,
        32,
        h,
        w,
        out2b_w,
        output_dim,
        1,
        1,
        1,
        0,
        Some(out2b_b),
        &gemm,
        &mut logits,
    );
    recycle_buffer(workspace, mid);
    let out2b_elapsed = out2b_started.elapsed();

    // ---- Split + activate: depth = exp(logit0); conf = exp(logit1) + 1.0 ("expp1") ----
    let hw = h * w;
    let mut depth = vec![0f32; hw];
    for i in 0..hw {
        depth[i] = logits[i].exp();
    }
    let conf = if output_dim >= 2 {
        let mut conf = vec![0f32; hw];
        for i in 0..hw {
            conf[i] = logits[hw + i].exp() + 1.0;
        }
        conf
    } else {
        Vec::new()
    };
    recycle_buffer(workspace, logits);

    if head_profile {
        eprintln!(
            "phase: head output_ops out1={:.3}ms resize={:.3}ms out2a={:.3}ms out2b={:.3}ms",
            out1_elapsed.as_secs_f64() * 1e3,
            resize_elapsed.as_secs_f64() * 1e3,
            out2a_elapsed.as_secs_f64() * 1e3,
            out2b_elapsed.as_secs_f64() * 1e3,
        );
        eprintln!(
            "phase: head output={:.3}ms total={:.3}ms",
            output_started.elapsed().as_secs_f64() * 1e3,
            head_started.elapsed().as_secs_f64() * 1e3,
        );
    }

    (
        DepthOut { depth, conf, h, w },
        if capture_debug {
            Some(DptDebug {
                stages: debug_stages.expect("debug stages retained"),
                fused: debug_fused.expect("debug fused retained"),
            })
        } else {
            None
        },
    )
}

pub fn dpt_head_debug(
    feats: &[Vec<f32>],
    h: usize,
    w: usize,
    cfg: &ModelConfig,
    weights: &Weights,
    cache: &mut UvEmbedCache,
    wino_cache: &mut WinogradFilterCache,
) -> (DepthOut, DptDebug) {
    let (out, debug) = dpt_head_impl(feats, h, w, cfg, weights, cache, wino_cache, None, true);
    (out, debug.expect("debug capture requested"))
}

/// Thin wrapper over [`dpt_head_debug`] discarding the debug intermediates —
/// see that function's doc comment for the full contract.
pub fn dpt_head(
    feats: &[Vec<f32>],
    h: usize,
    w: usize,
    cfg: &ModelConfig,
    weights: &Weights,
    cache: &mut UvEmbedCache,
    wino_cache: &mut WinogradFilterCache,
) -> DepthOut {
    dpt_head_impl(feats, h, w, cfg, weights, cache, wino_cache, None, false).0
}

/// Production DPT route with an engine-owned activation workspace.  The
/// public debug route deliberately does not use this pool so captured
/// intermediates retain normal ownership.
pub fn dpt_head_with_workspace(
    feats: &[Vec<f32>],
    h: usize,
    w: usize,
    cfg: &ModelConfig,
    weights: &Weights,
    cache: &mut UvEmbedCache,
    wino_cache: &mut WinogradFilterCache,
    workspace: &HeadWorkspace,
) -> DepthOut {
    dpt_head_impl(
        feats,
        h,
        w,
        cfg,
        weights,
        cache,
        wino_cache,
        Some(workspace),
        false,
    )
    .0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fused_token_norm_chw_handoff_matches_materialized_path_bitwise() {
        let (n_tok, c, grid_h, grid_w) = (35usize, 16usize, 5usize, 7usize);
        let input: Vec<f32> = (0..n_tok * c)
            .map(|index| ((index * 37 % 251) as f32 - 125.0) * 0.0078125)
            .collect();
        let gamma: Vec<f32> = (0..c).map(|index| 0.5 + index as f32 * 0.03125).collect();
        let beta: Vec<f32> = (0..c)
            .map(|index| -0.25 + index as f32 * 0.015625)
            .collect();

        let mut materialized = input.clone();
        scalar::layernorm(&mut materialized, n_tok, c, &gamma, &beta, HEAD_NORM_EPS);
        let expected = tok_major_to_chw(&materialized, n_tok, c, grid_h, grid_w);
        let actual = token_major_to_chw_with_optional_layernorm(
            &input,
            n_tok,
            c,
            grid_h,
            grid_w,
            Some((&gamma, &beta)),
        );
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
        );
    }
    use da_graph::Weights;

    fn test_cfg() -> ModelConfig {
        ModelConfig {
            arch: "depthanything3".to_string(),
            patch_size: 8,
            image_size: 224,
            embed_dim: 4,
            depth: 1,
            num_heads: 1,
            head_dim: 4,
            mlp_hidden: 4,
            num_register: 0,
            rope_start: -1,
            qknorm_start: -1,
            rope_freq: 100.0,
            ln_eps: 1e-6,
            out_layers: vec![0, 1, 2, 3],
            ffn_type: "mlp".to_string(),
            head_features: 8, // feat_half = 4 (kept tiny for fast tests)
            head_max_depth: 1.0,
            img_mean: [0.0, 0.0, 0.0],
            img_std: [1.0, 1.0, 1.0],
            img_resize_mode: "bilinear".to_string(),
            alt_start: -1,
            cat_token: false, // c_in = embed_dim = 4 (keep tensors small)
            cam_dim_in: 1,
            head_pos_embed: true,
        }
    }

    struct Xorshift32(u32);
    impl Xorshift32 {
        fn next_f32(&mut self) -> f32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            self.0 = x;
            ((x as f32) / (u32::MAX as f32)) * 2.0 - 1.0
        }
    }
    fn random_vec(rng: &mut Xorshift32, n: usize) -> Vec<f32> {
        (0..n).map(|_| rng.next_f32()).collect()
    }

    /// Builds a full synthetic `head.*` weight set matching `test_cfg()`'s
    /// tiny dimensions (`c_in=4`, `oc=[96,192,384,768]` fixed regardless of
    /// `embed_dim` — the projections' *input* side is `c_in`, but their
    /// *output* side is always the real DA3-BASE `oc[]`, so this exercises
    /// the real channel-count pyramid even with a tiny `c_in`), grid=4x4
    /// (16 tokens).
    fn synthetic_weights(cfg: &ModelConfig, grid: usize, with_head_norm: bool) -> Weights {
        let c_in = if cfg.cat_token {
            2 * cfg.embed_dim as usize
        } else {
            cfg.embed_dim as usize
        };
        let oc = DEFAULT_OC;
        let mut rng = Xorshift32(0xD9_7E_A0);
        let mut w = Weights::new();
        // Scaled down from the usual [-1,1] test range: this synthetic
        // weight set feeds ~10 chained conv/residual-conv-unit layers, and
        // full-magnitude random weights compound into logits that overflow
        // f32 (exp() -> inf/0), which isn't a real numerical property of
        // the head (trained weights aren't adversarially chosen to blow up)
        // -- just an artifact of stacking that many random layers. Scaling
        // keeps the shape/gating tests below meaningful without pretending
        // to be closer to real weight statistics than they are.
        let mut put = |name: String, len: usize, w: &mut Weights| {
            let v: Vec<f32> = random_vec(&mut rng, len)
                .into_iter()
                .map(|x| x * 0.05)
                .collect();
            w.insert_f32(name, v);
        };

        if with_head_norm {
            put("head.norm.weight".to_string(), c_in, &mut w);
            put("head.norm.bias".to_string(), c_in, &mut w);
        }

        for s in 0..4 {
            put(format!("head.proj.{s}.weight"), oc[s] * c_in, &mut w);
            put(format!("head.proj.{s}.bias"), oc[s], &mut w);
        }
        put(
            "head.resize.0.weight".to_string(),
            oc[0] * oc[0] * 4 * 4,
            &mut w,
        );
        put("head.resize.0.bias".to_string(), oc[0], &mut w);
        put(
            "head.resize.1.weight".to_string(),
            oc[1] * oc[1] * 2 * 2,
            &mut w,
        );
        put("head.resize.1.bias".to_string(), oc[1], &mut w);
        put(
            "head.resize.3.weight".to_string(),
            oc[3] * oc[3] * 3 * 3,
            &mut w,
        );
        put("head.resize.3.bias".to_string(), oc[3], &mut w);

        for s in 0..4 {
            put(
                format!("head.scratch.layer{}_rn.weight", s + 1),
                FUSION_C * oc[s] * 3 * 3,
                &mut w,
            );
        }
        for i in 1..=4 {
            if i != 4 {
                for cn in ["c1", "c2"] {
                    put(
                        format!("head.scratch.rn{i}.rc1.{cn}.weight"),
                        FUSION_C * FUSION_C * 3 * 3,
                        &mut w,
                    );
                    put(
                        format!("head.scratch.rn{i}.rc1.{cn}.bias"),
                        FUSION_C,
                        &mut w,
                    );
                }
            }
            for cn in ["c1", "c2"] {
                put(
                    format!("head.scratch.rn{i}.rc2.{cn}.weight"),
                    FUSION_C * FUSION_C * 3 * 3,
                    &mut w,
                );
                put(
                    format!("head.scratch.rn{i}.rc2.{cn}.bias"),
                    FUSION_C,
                    &mut w,
                );
            }
            put(
                format!("head.scratch.rn{i}.out.weight"),
                FUSION_C * FUSION_C,
                &mut w,
            );
            put(format!("head.scratch.rn{i}.out.bias"), FUSION_C, &mut w);
        }

        let feat_half = cfg.head_features as usize / 2;
        put(
            "head.scratch.out1.weight".to_string(),
            feat_half * FUSION_C * 3 * 3,
            &mut w,
        );
        put("head.scratch.out1.bias".to_string(), feat_half, &mut w);
        put(
            "head.scratch.out2a.weight".to_string(),
            32 * feat_half * 3 * 3,
            &mut w,
        );
        put("head.scratch.out2a.bias".to_string(), 32, &mut w);
        put("head.scratch.out2b.weight".to_string(), 2 * 32, &mut w);
        put("head.scratch.out2b.bias".to_string(), 2, &mut w);

        let _ = grid;
        w
    }

    fn synthetic_feats(cfg: &ModelConfig, grid: usize) -> Vec<Vec<f32>> {
        let c_in = if cfg.cat_token {
            2 * cfg.embed_dim as usize
        } else {
            cfg.embed_dim as usize
        };
        let mut rng = Xorshift32(0xFEED_0001);
        (0..4)
            .map(|_| random_vec(&mut rng, grid * grid * c_in))
            .collect()
    }

    #[test]
    fn dpt_head_produces_expected_shapes_and_positive_depth() {
        let cfg = test_cfg();
        let grid = 4usize; // 16 tokens
        let weights = synthetic_weights(&cfg, grid, false);
        let feats = synthetic_feats(&cfg, grid);
        let mut cache = UvEmbedCache::new();
        let mut wino_cache = WinogradFilterCache::new();

        let (h, w) = (32usize, 32usize);
        let out = dpt_head(&feats, h, w, &cfg, &weights, &mut cache, &mut wino_cache);

        assert_eq!(out.h, h);
        assert_eq!(out.w, w);
        assert_eq!(out.depth.len(), h * w);
        assert_eq!(out.conf.len(), h * w);
        assert!(
            out.depth.iter().all(|&v| v > 0.0),
            "depth = exp(x) must be strictly positive"
        );
        assert!(
            out.conf.iter().all(|&v| v >= 1.0),
            "conf = exp(x)+1 must be >= 1.0"
        );
        assert!(out.depth.iter().all(|v| v.is_finite()));
        assert!(out.conf.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn workspace_route_matches_fresh_head_bitwise_across_reuse() {
        let cfg = test_cfg();
        let grid = 4usize;
        let weights = synthetic_weights(&cfg, grid, true);
        let feats = synthetic_feats(&cfg, grid);
        let (h, w) = (32usize, 32usize);

        let mut fresh_cache = UvEmbedCache::new();
        let mut fresh_wino = WinogradFilterCache::new();
        let fresh = dpt_head(
            &feats,
            h,
            w,
            &cfg,
            &weights,
            &mut fresh_cache,
            &mut fresh_wino,
        );

        let workspace = HeadWorkspace::new();
        let mut pooled_cache = UvEmbedCache::new();
        let mut pooled_wino = WinogradFilterCache::new();
        for _ in 0..2 {
            let pooled = dpt_head_with_workspace(
                &feats,
                h,
                w,
                &cfg,
                &weights,
                &mut pooled_cache,
                &mut pooled_wino,
                &workspace,
            );
            for (control, candidate) in fresh.depth.iter().zip(&pooled.depth) {
                assert_eq!(control.to_bits(), candidate.to_bits());
            }
            for (control, candidate) in fresh.conf.iter().zip(&pooled.conf) {
                assert_eq!(control.to_bits(), candidate.to_bits());
            }
        }
    }

    #[test]
    fn dpt_head_debug_stage_shapes_match_resize_layer_geometry() {
        let cfg = test_cfg();
        let grid = 4usize;
        let weights = synthetic_weights(&cfg, grid, true);
        let feats = synthetic_feats(&cfg, grid);
        let mut cache = UvEmbedCache::new();
        let mut wino_cache = WinogradFilterCache::new();

        let (h, w) = (32usize, 32usize);
        let (_, debug) = dpt_head_debug(&feats, h, w, &cfg, &weights, &mut cache, &mut wino_cache);

        let oc = DEFAULT_OC;
        // stage0: conv_transpose k4s4 -> 4*grid; stage1: k2s2 -> 2*grid;
        // stage2: identity -> grid; stage3: conv k3s2p1 -> (grid+2-3)/2+1.
        assert_eq!(debug.stages[0].len(), oc[0] * (4 * grid) * (4 * grid));
        assert_eq!(debug.stages[1].len(), oc[1] * (2 * grid) * (2 * grid));
        assert_eq!(debug.stages[2].len(), oc[2] * grid * grid);
        let g3 = (grid + 2 - 3) / 2 + 1;
        assert_eq!(debug.stages[3].len(), oc[3] * g3 * g3);

        let feat_half = cfg.head_features as usize / 2;
        let fh = 2 * (4 * grid);
        assert_eq!(
            debug.fused.len(),
            feat_half * fh * fh,
            "fused = output_conv1's [feat_half,fh,fh]"
        );
    }

    #[test]
    fn dpt_head_head_norm_presence_changes_output() {
        // Presence-gating (trap pattern shared with vit_block's ls1/ls2):
        // with vs. without head.norm must produce different output on the
        // SAME random tokens/other weights.
        let cfg = test_cfg();
        let grid = 4usize;
        let feats = synthetic_feats(&cfg, grid);
        let mut cache_a = UvEmbedCache::new();
        let mut cache_b = UvEmbedCache::new();
        let mut wino_cache_a = WinogradFilterCache::new();
        let mut wino_cache_b = WinogradFilterCache::new();

        let w_with = synthetic_weights(&cfg, grid, true);
        let w_without = synthetic_weights(&cfg, grid, false);

        let out_with = dpt_head(
            &feats,
            32,
            32,
            &cfg,
            &w_with,
            &mut cache_a,
            &mut wino_cache_a,
        );
        let out_without = dpt_head(
            &feats,
            32,
            32,
            &cfg,
            &w_without,
            &mut cache_b,
            &mut wino_cache_b,
        );

        assert_ne!(out_with.depth, out_without.depth);
    }

    #[test]
    fn dpt_head_pos_embed_gating_changes_output() {
        let mut cfg = test_cfg();
        let grid = 4usize;
        let weights = synthetic_weights(&cfg, grid, false);
        let feats = synthetic_feats(&cfg, grid);

        cfg.head_pos_embed = true;
        let mut cache = UvEmbedCache::new();
        let mut wino_cache = WinogradFilterCache::new();
        let out_with_pe = dpt_head(&feats, 32, 32, &cfg, &weights, &mut cache, &mut wino_cache);

        cfg.head_pos_embed = false;
        let mut cache2 = UvEmbedCache::new();
        let mut wino_cache2 = WinogradFilterCache::new();
        let out_without_pe = dpt_head(
            &feats,
            32,
            32,
            &cfg,
            &weights,
            &mut cache2,
            &mut wino_cache2,
        );

        assert_ne!(out_with_pe.depth, out_without_pe.depth);
    }

    #[test]
    #[should_panic(expected = "token count does not match")]
    fn dpt_head_panics_when_token_count_does_not_match_grid() {
        let cfg = test_cfg();
        let c_in = cfg.embed_dim as usize; // cat_token=false
        let mut rng = Xorshift32(0x1234);
        // 15 tokens cannot represent the 4x4 grid implied by 32/patch8.
        let feats: Vec<Vec<f32>> = (0..4).map(|_| random_vec(&mut rng, 15 * c_in)).collect();
        let weights = synthetic_weights(&cfg, 4, false);
        let mut cache = UvEmbedCache::new();
        let mut wino_cache = WinogradFilterCache::new();
        let _ = dpt_head(&feats, 32, 32, &cfg, &weights, &mut cache, &mut wino_cache);
    }
}
