//! Runs the full 12-layer (or however deep `cfg.depth` says) ViT block
//! stack, collecting the `feat_{5,7,9,11}`/`cam_token_{5,7,9,11}` outputs
//! (for DA3-BASE, whose `depthanything3.vit.out_layers` metadata is
//! `[5,7,9,11]` — see `../scripts/gguf_keys.py`) that `get_intermediate_layers`
//! actually produces.
//!
//! ## What this does (and why it isn't "just run `vit_block` `depth` times")
//!
//! The real C++ reference's `DinoBackbone::forward`
//! (`../src/dino_backbone.cpp`) does four things beyond a bare block loop,
//! all reproduced here:
//!
//! 1. **Camera-token injection**: right before block `i == cfg.alt_start`
//!    runs, token-0 (the CLS slot) of the current token buffer is
//!    overwritten with the `vit.camera_token` weight tensor's value.
//! 2. **Local/global attention alternation**: `global = cfg.alt_start >= 0
//!    && i >= cfg.alt_start && i % 2 == 1`. Global layers use a different
//!    RoPE position set ("nodiff": every patch position collapses to
//!    `(1,1)`) instead of the normal per-patch positions — see
//!    `vit_block`'s `global` parameter. A separate `local_x` buffer tracks
//!    the most recent LOCAL (non-global) block's output, updated only when
//!    a block is NOT global.
//! 3. **Final normalization + doubled-width concatenation**: at each
//!    captured out-layer, `feat = cat([local_x, vit_norm(x)])` over the
//!    channel dimension (patches 1..n, i.e. token-0/CLS is stripped),
//!    producing `[n_patch, 2*embed_dim]` when `cfg.cat_token == true`.
//! 4. **Cam-token output**: `cam = cat([local_x[token0], x[token0]])`, RAW
//!    (no normalization), producing `[2*embed_dim]` per out-layer — a
//!    separate gated output from `feat`.
//!
//! When `cfg.cat_token == false` (da2/mono models, not DA3-BASE), the
//! single-width form is used instead: `feat = vit_norm(x)` (patches 1..n,
//! `[n_patch, embed_dim]`), `cam = x[token0]` RAW (`[embed_dim]`, kept for
//! shape consistency even though it's unused on that branch per the C++
//! comment).
//!
//! ## Honesty note (still true after this fix)
//!
//! This was investigated and rewritten against `../src/dino_backbone.cpp`
//! read as ground truth, but is NOT hardware-verified: no
//! `../models/*.gguf` or `../dumps/` exist in this environment, so
//! `backbone_parity.rs`'s dump-gated test still SKIPS here. If a real
//! model/dumps ever become available and this doesn't match byte-for-byte,
//! the first things to re-check are the exact `alt_start`/`cat_token` GGUF
//! metadata values on the real file (this fix's `ModelConfig::from_gguf`
//! defaults — `alt_start: -1`, `cat_token: true` — were cross-referenced
//! against `include/da_gguf_keys.h` and `../src/model_loader.cpp`, not
//! observed on a real GGUF file directly).
#[cfg(test)]
use crate::vit_block::vit_block;
use crate::vit_block::{
    vit_block_with_residual, vit_block_with_views, AttentionExecutor, MlpExecutor,
    ResidualAddExecutor, TransformerTailExecutor,
};
use crate::ModelConfig;
use da_graph::{Backend, Weights};
use std::io::Write;
use std::path::PathBuf;

/// `vit.norm` LayerNorm weight/bias tensor names — the final normalization
/// applied to `x` (never to `local_x`) before it's concatenated into `feat`.
/// Confirmed against `../src/dino_backbone.cpp`
/// (`ml_.tensor("vit.norm.weight")` / `ml_.tensor("vit.norm.bias")`).
pub const VIT_NORM_WEIGHT: &str = "vit.norm.weight";
pub const VIT_NORM_BIAS: &str = "vit.norm.bias";

/// Camera-token weight tensor name. Confirmed against
/// `../src/dino_backbone.cpp` (`ml_.tensor("vit.camera_token")`, `ne0=embed,
/// ne1=2` — this single-view `forward()` path only ever reads slot 0, i.e.
/// the first `embed_dim` floats of the tensor's data).
pub const CAMERA_TOKEN_WEIGHT: &str = "vit.camera_token";

fn trace_multi_view_tensor(trace_dir: Option<&PathBuf>, name: &str, views: &[Vec<f32>]) {
    let Some(trace_dir) = trace_dir else {
        return;
    };
    let path = trace_dir.join(format!("rust-{name}.f32"));
    let mut file = std::fs::File::create(&path).unwrap_or_else(|error| {
        panic!("cannot create multi-view trace {}: {error}", path.display())
    });
    for value in views.iter().flatten() {
        file.write_all(&value.to_le_bytes())
            .unwrap_or_else(|error| {
                panic!("cannot write multi-view trace {}: {error}", path.display())
            });
    }
}

/// Per-out-layer captured outputs of `Backbone::forward`: `feat` (per-patch
/// features, token-0 stripped) and `cam_token` (the token-0/CLS-derived
/// "camera" summary), matching the real engine's `feat_*`/`cam_token_*`
/// dumps. Both are indexed in `out_layers` order (not layer-index order).
pub struct BackboneOutputs {
    /// `feats[o]` is `[n_patch * width]` row-major (`width = 2*embed_dim`
    /// when `cfg.cat_token`, else `embed_dim`), patches in the same
    /// row-major `(gh,gw)` order as `tokens` (token-0/CLS excluded).
    pub feats: Vec<Vec<f32>>,
    /// `cam_tokens[o]` is `[2*embed_dim]` when `cfg.cat_token`, else
    /// `[embed_dim]`.
    pub cam_tokens: Vec<Vec<f32>>,
}

/// Ordered multi-view captures, indexed as `[out_layer][view]`.
///
/// View zero is the reference view and receives camera-token slot zero;
/// every later view receives the source-camera slot. Reference-view
/// selection and restoration are intentionally a separate operation, as in
/// the pinned C++ oracle's `forward_mv` wrapper.
pub struct MultiViewBackboneOutputs {
    pub feats: Vec<Vec<Vec<f32>>>,
    pub cam_tokens: Vec<Vec<Vec<f32>>>,
}

/// Selects PR #2's saddle-balanced reference view from local CLS features.
///
/// Each row is one view's CLS vector. The selected view minimizes the sum of
/// distances to the normalized midpoints of mean cosine similarity, vector
/// norm, and unbiased normalized-feature variance. Ties retain the earliest
/// view, matching the C++ `<` comparison.
#[must_use]
pub fn select_reference_view_saddle(cls: &[Vec<f32>]) -> usize {
    if cls.len() <= 1 {
        return 0;
    }
    let embed = cls[0].len();
    assert!(
        embed > 1,
        "saddle selection needs at least two CLS channels"
    );
    assert!(
        cls.iter().all(|row| row.len() == embed),
        "all CLS feature rows must have the same width"
    );

    let mut norm = vec![0.0f64; cls.len()];
    let mut normalized = vec![vec![0.0f64; embed]; cls.len()];
    for (view, row) in cls.iter().enumerate() {
        let magnitude = row
            .iter()
            .map(|&value| f64::from(value) * f64::from(value))
            .sum::<f64>()
            .sqrt();
        norm[view] = magnitude;
        let inverse = if magnitude > 0.0 {
            1.0 / magnitude
        } else {
            0.0
        };
        for (dst, &value) in normalized[view].iter_mut().zip(row) {
            *dst = f64::from(value) * inverse;
        }
    }

    let mut similarity = vec![0.0f64; cls.len()];
    let mut variance = vec![0.0f64; cls.len()];
    for view in 0..cls.len() {
        for other in 0..cls.len() {
            if other != view {
                similarity[view] += normalized[view]
                    .iter()
                    .zip(&normalized[other])
                    .map(|(left, right)| left * right)
                    .sum::<f64>();
            }
        }
        similarity[view] /= (cls.len() - 1) as f64;
        let mean = normalized[view].iter().sum::<f64>() / embed as f64;
        variance[view] = normalized[view]
            .iter()
            .map(|value| {
                let delta = value - mean;
                delta * delta
            })
            .sum::<f64>()
            / (embed - 1) as f64;
    }

    normalize_zero_one(&mut similarity);
    normalize_zero_one(&mut norm);
    normalize_zero_one(&mut variance);

    let mut best = 0;
    let mut best_balance = f64::INFINITY;
    for view in 0..cls.len() {
        let balance = (similarity[view] - 0.5).abs()
            + (norm[view] - 0.5).abs()
            + (variance[view] - 0.5).abs();
        if balance < best_balance {
            best_balance = balance;
            best = view;
        }
    }
    best
}

fn normalize_zero_one(values: &mut [f64]) {
    let minimum = values.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let denominator = maximum - minimum + 1e-8;
    for value in values {
        *value = (*value - minimum) / denominator;
    }
}

#[must_use]
pub fn reference_first_order(view_count: usize, reference: usize) -> Vec<usize> {
    assert!(reference < view_count);
    let mut order = Vec::with_capacity(view_count);
    order.push(reference);
    order.extend((0..view_count).filter(|&view| view != reference));
    order
}

/// Owns nothing — just bundles the `cfg`/`weights`/`backend` a full
/// backbone forward pass needs, borrowed for the duration of `forward`.
pub struct Backbone<'a> {
    pub cfg: &'a ModelConfig,
    pub weights: &'a Weights,
    pub backend: &'a dyn Backend,
    residual_executor: Option<&'a dyn ResidualAddExecutor>,
    mlp_executor: Option<&'a dyn MlpExecutor>,
    attention_executor: Option<&'a dyn AttentionExecutor>,
    transformer_tail_executor: Option<&'a dyn TransformerTailExecutor>,
}

impl<'a> Backbone<'a> {
    pub fn new(cfg: &'a ModelConfig, weights: &'a Weights, backend: &'a dyn Backend) -> Self {
        Backbone {
            cfg,
            weights,
            backend,
            residual_executor: None,
            mlp_executor: None,
            attention_executor: None,
            transformer_tail_executor: None,
        }
    }

    #[cfg(feature = "cuda-residual-oracle")]
    pub(crate) fn new_with_residual(
        cfg: &'a ModelConfig,
        weights: &'a Weights,
        backend: &'a dyn Backend,
        residual_executor: &'a dyn ResidualAddExecutor,
    ) -> Self {
        Self {
            cfg,
            weights,
            backend,
            residual_executor: Some(residual_executor),
            mlp_executor: None,
            attention_executor: None,
            transformer_tail_executor: None,
        }
    }

    #[cfg(feature = "cuda-residual-oracle")]
    pub(crate) fn new_with_mlp(
        cfg: &'a ModelConfig,
        weights: &'a Weights,
        backend: &'a dyn Backend,
        mlp_executor: &'a dyn MlpExecutor,
    ) -> Self {
        Self {
            cfg,
            weights,
            backend,
            residual_executor: None,
            mlp_executor: Some(mlp_executor),
            attention_executor: None,
            transformer_tail_executor: None,
        }
    }

    #[cfg(feature = "cuda-residual-oracle")]
    pub(crate) fn new_with_attention(
        cfg: &'a ModelConfig,
        weights: &'a Weights,
        backend: &'a dyn Backend,
        attention_executor: &'a dyn AttentionExecutor,
    ) -> Self {
        Self {
            cfg,
            weights,
            backend,
            residual_executor: None,
            mlp_executor: None,
            attention_executor: Some(attention_executor),
            transformer_tail_executor: None,
        }
    }

    #[cfg(feature = "cuda-residual-oracle")]
    pub(crate) fn new_with_transformer_tail(
        cfg: &'a ModelConfig,
        weights: &'a Weights,
        backend: &'a dyn Backend,
        transformer_tail_executor: &'a dyn TransformerTailExecutor,
    ) -> Self {
        Self {
            cfg,
            weights,
            backend,
            residual_executor: None,
            mlp_executor: None,
            attention_executor: None,
            transformer_tail_executor: Some(transformer_tail_executor),
        }
    }

    /// Captures the CLS row after the first `upto` local transformer blocks.
    ///
    /// This is the preliminary pass used by PR #2 for `S >= 3` saddle-balanced
    /// reference selection. It deliberately runs before camera-token injection
    /// and before any global block, so no view can influence another here.
    pub fn capture_local_cls(
        &self,
        views: &[Vec<f32>],
        gh: usize,
        gw: usize,
        upto: usize,
    ) -> Vec<Vec<f32>> {
        assert!(
            !views.is_empty(),
            "local CLS capture needs at least one view"
        );
        let cfg = self.cfg;
        let embed = cfg.embed_dim as usize;
        assert!(
            upto <= cfg.depth as usize,
            "requested {upto} local blocks but model depth is {}",
            cfg.depth
        );
        let n = views[0].len() / embed;
        assert_eq!(n, 1 + cfg.num_register as usize + gh * gw);

        let mut captured_views = views.to_vec();
        for view in &mut captured_views {
            assert_eq!(view.len(), n * embed, "all views must share a token shape");
            for layer_idx in 0..upto {
                vit_block_with_residual(
                    view,
                    n,
                    gh,
                    gw,
                    false,
                    cfg,
                    layer_idx,
                    self.weights,
                    self.backend,
                    self.residual_executor,
                    self.mlp_executor,
                    self.attention_executor,
                    self.transformer_tail_executor,
                );
            }
        }
        captured_views
            .into_iter()
            .map(|view| view[..embed].to_vec())
            .collect()
    }

    /// Runs `cfg.depth` `vit_block` calls over `tokens` in place (applying
    /// camera-token injection at `cfg.alt_start` and local/global attention
    /// alternation per the module doc comment), then host-post-processes
    /// the raw per-out-layer captures into `feat`/`cam_token` pairs matching
    /// `get_intermediate_layers`'s output format.
    ///
    /// `tokens` must already be `prepare_tokens`'s output
    /// (`n = 1 + num_register + gh*gw` rows of `embed_dim`); `gh`/`gw` are
    /// the patch-grid resolution (needed for RoPE position derivation on any
    /// layer `>= cfg.rope_start`, and to know how many rows are patches vs.
    /// the leading CLS/register rows).
    ///
    /// `out_layers` order determines the order of `BackboneOutputs.feats`/
    /// `.cam_tokens` (matching callers' listed order, not execution order).
    pub fn forward(
        &self,
        tokens: &mut [f32],
        gh: usize,
        gw: usize,
        out_layers: &[i32],
    ) -> BackboneOutputs {
        #[cfg(feature = "cuda-residual-oracle")]
        if let Some(executor) = self.transformer_tail_executor {
            // The persistent route is intentionally single-view first. Global
            // multi-view needs distinct reference/source camera-token slots
            // and is kept on the independently validated host adapter until
            // that scheduler owns all view buffers on device as well.
            if std::env::var_os("VESTRA_TRACE_DIR").is_none()
                && self.cfg.qknorm_start >= 0
                && self.cfg.alt_start >= self.cfg.qknorm_start
            {
                if let Some(cuda_executor) = executor.persistent_cuda_tail() {
                    return self.forward_with_persistent_cuda_tail(
                        tokens,
                        gh,
                        gw,
                        out_layers,
                        cuda_executor,
                    );
                }
            }
        }
        let phase_profile = std::env::var_os("DA_PHASE_PROFILE").is_some();
        let cfg = self.cfg;
        let embed = cfg.embed_dim as usize;
        assert_eq!(
            tokens.len() % embed,
            0,
            "tokens length must be a multiple of embed_dim"
        );
        let n = tokens.len() / embed;
        let n_special = 1 + cfg.num_register as usize;
        assert!(
            n >= n_special,
            "token count {n} smaller than n_special (1 CLS + {} register)",
            cfg.num_register
        );

        // local_x: the most recent LOCAL (non-global) block's output,
        // cloned. Starts as the pre-block-0 token buffer (matching the C++
        // reference's `local_x = x` initialization before the loop).
        let mut local_x: Vec<f32> = tokens.to_vec();

        let mut feats: Vec<Option<Vec<f32>>> = vec![None; out_layers.len()];
        let mut cam_tokens: Vec<Option<Vec<f32>>> = vec![None; out_layers.len()];

        for layer_idx in 0..cfg.depth as usize {
            let block_started = std::time::Instant::now();
            // Camera-token overwrite BEFORE block i==alt_start: token-0
            // (CLS slot, row 0) <- vit.camera_token[0..embed].
            if cfg.alt_start >= 0 && layer_idx == cfg.alt_start as usize {
                let cam = self
                    .weights
                    .get_f32(CAMERA_TOKEN_WEIGHT)
                    .unwrap_or_else(|| panic!("missing weight tensor {CAMERA_TOKEN_WEIGHT:?} required by cfg.alt_start={}", cfg.alt_start));
                assert!(
                    cam.len() >= embed,
                    "{CAMERA_TOKEN_WEIGHT} too short: expected >= {embed} floats, got {}",
                    cam.len()
                );
                tokens[0..embed].copy_from_slice(&cam[0..embed]);
            }

            let global =
                cfg.alt_start >= 0 && (layer_idx as i32) >= cfg.alt_start && layer_idx % 2 == 1;

            vit_block_with_residual(
                tokens,
                n,
                gh,
                gw,
                global,
                cfg,
                layer_idx,
                self.weights,
                self.backend,
                self.residual_executor,
                self.mlp_executor,
                self.attention_executor,
                self.transformer_tail_executor,
            );

            if phase_profile {
                eprintln!(
                    "phase: transformer_block[{layer_idx}]={:.3}ms global={global}",
                    block_started.elapsed().as_secs_f64() * 1e3,
                );
            }

            for (slot, &wanted) in out_layers.iter().enumerate() {
                if wanted == layer_idx as i32 {
                    let local = if global { &local_x } else { &*tokens };
                    let (feat, cam) = self.post_process_capture(n, n_special, local, tokens);
                    feats[slot] = Some(feat);
                    cam_tokens[slot] = Some(cam);
                }
            }

            // `local_x` is only consumed by the immediately following global
            // block. Avoid a full token-buffer copy after local blocks that
            // cannot feed such a capture.
            let next_is_global = layer_idx + 1 < cfg.depth as usize
                && cfg.alt_start >= 0
                && (layer_idx + 1) as i32 >= cfg.alt_start
                && (layer_idx + 1) % 2 == 1;
            if !global && next_is_global {
                local_x.copy_from_slice(tokens);
            }
        }

        let feats: Vec<Vec<f32>> = feats
            .into_iter()
            .enumerate()
            .map(|(i, c)| {
                c.unwrap_or_else(|| {
                    panic!(
                        "out_layers[{i}]={} was never reached (depth={})",
                        out_layers[i], cfg.depth
                    )
                })
            })
            .collect();
        let cam_tokens: Vec<Vec<f32>> = cam_tokens
            .into_iter()
            .enumerate()
            .map(|(i, c)| {
                c.unwrap_or_else(|| {
                    panic!(
                        "out_layers[{i}]={} was never reached (depth={})",
                        out_layers[i], cfg.depth
                    )
                })
            })
            .collect();

        BackboneOutputs { feats, cam_tokens }
    }

    /// Single-view DA3-BASE backbone route which transfers tokens to CUDA at
    /// the first Q/K-normalized block and retains them there through the final
    /// block. Captures are the only intentional downloads in the tail: DPT is
    /// still CPU-owned, and the caller's mutable token slice receives the
    /// final device state for the ordinary `Backbone::forward` contract.
    #[cfg(feature = "cuda-residual-oracle")]
    fn forward_with_persistent_cuda_tail(
        &self,
        tokens: &mut [f32],
        gh: usize,
        gw: usize,
        out_layers: &[i32],
        executor: &crate::vit_block::CudaTransformerTailExecutor,
    ) -> BackboneOutputs {
        let cfg = self.cfg;
        let embed = cfg.embed_dim as usize;
        assert_eq!(
            tokens.len() % embed,
            0,
            "tokens length must be a multiple of embed"
        );
        let n = tokens.len() / embed;
        let n_special = 1 + cfg.num_register as usize;
        let first_device_layer = cfg.qknorm_start as usize;
        let phase_profile = std::env::var_os("DA_PHASE_PROFILE").is_some();
        let mut local_host = tokens.to_vec();
        let mut device_tokens = None;
        let mut device_local = None;
        let mut feats = vec![None; out_layers.len()];
        let mut cam_tokens = vec![None; out_layers.len()];

        for layer_idx in 0..cfg.depth as usize {
            let started = std::time::Instant::now();
            let global =
                cfg.alt_start >= 0 && layer_idx as i32 >= cfg.alt_start && layer_idx % 2 == 1;
            if layer_idx < first_device_layer {
                if cfg.alt_start >= 0 && layer_idx == cfg.alt_start as usize {
                    let camera = self
                        .weights
                        .get_f32(CAMERA_TOKEN_WEIGHT)
                        .unwrap_or_else(|| panic!("missing weight tensor {CAMERA_TOKEN_WEIGHT:?}"));
                    tokens[..embed].copy_from_slice(&camera[..embed]);
                }
                vit_block_with_residual(
                    tokens,
                    n,
                    gh,
                    gw,
                    global,
                    cfg,
                    layer_idx,
                    self.weights,
                    self.backend,
                    None,
                    None,
                    None,
                    None,
                );
                local_host.copy_from_slice(tokens);
            } else {
                if device_tokens.is_none() {
                    device_tokens = executor.upload_tokens(tokens);
                }
                let mut state = device_tokens
                    .take()
                    .expect("qualified CUDA tail must upload its first token state");
                if cfg.alt_start >= 0 && layer_idx == cfg.alt_start as usize {
                    executor
                        .inject_reference_camera_token(&mut state, embed)
                        .expect("qualified CUDA tail must inject reference camera token");
                }
                state = executor
                    .run_tail_device(layer_idx, state, n, gh, gw, global, 1, cfg)
                    .expect("qualified CUDA tail must execute its configured block");
                device_tokens = Some(state);
            }

            if phase_profile {
                eprintln!(
                    "phase: transformer_block[{layer_idx}]={:.3}ms global={global} persistent_cuda={}",
                    started.elapsed().as_secs_f64() * 1e3,
                    layer_idx >= first_device_layer,
                );
            }

            if layer_idx >= first_device_layer {
                let state = device_tokens.as_ref().expect("device state exists");
                for (slot, &wanted) in out_layers.iter().enumerate() {
                    if wanted == layer_idx as i32 {
                        let x = executor.download_tokens(state).expect("capture download");
                        let local = if global {
                            executor
                                .download_tokens(
                                    device_local
                                        .as_ref()
                                        .expect("global capture has local state"),
                                )
                                .expect("local capture download")
                        } else {
                            x.clone()
                        };
                        let (feat, cam) = self.post_process_capture(n, n_special, &local, &x);
                        feats[slot] = Some(feat);
                        cam_tokens[slot] = Some(cam);
                    }
                }
                let next_is_global = layer_idx + 1 < cfg.depth as usize
                    && cfg.alt_start >= 0
                    && (layer_idx + 1) as i32 >= cfg.alt_start
                    && (layer_idx + 1) % 2 == 1;
                if !global && next_is_global {
                    device_local =
                        Some(executor.copy_tokens(state).expect("device local snapshot"));
                }
            } else {
                for (slot, &wanted) in out_layers.iter().enumerate() {
                    if wanted == layer_idx as i32 {
                        let (feat, cam) =
                            self.post_process_capture(n, n_special, &local_host, tokens);
                        feats[slot] = Some(feat);
                        cam_tokens[slot] = Some(cam);
                    }
                }
            }
        }
        let final_tokens = executor
            .download_tokens(device_tokens.as_ref().expect("final device state"))
            .expect("final CUDA tail download");
        tokens.copy_from_slice(&final_tokens);
        BackboneOutputs {
            feats: feats
                .into_iter()
                .enumerate()
                .map(|(slot, value)| {
                    value.unwrap_or_else(|| panic!("out_layers[{slot}] was never reached"))
                })
                .collect(),
            cam_tokens: cam_tokens
                .into_iter()
                .enumerate()
                .map(|(slot, value)| {
                    value.unwrap_or_else(|| panic!("out_layers[{slot}] was never reached"))
                })
                .collect(),
        }
    }

    /// Runs the pinned PR #2 ordered multi-view transformer schedule.
    ///
    /// Local blocks execute independently for every view. Global blocks
    /// flatten all view-major token buffers into one attention sequence,
    /// then restore the per-view slices. This is the material distinction
    /// between real multi-view inference and a loop of single-image calls.
    /// View zero is assumed to have already been selected as the reference.
    pub fn forward_multi_view_ordered(
        &self,
        views: &mut [Vec<f32>],
        gh: usize,
        gw: usize,
        out_layers: &[i32],
    ) -> MultiViewBackboneOutputs {
        #[cfg(feature = "cuda-residual-oracle")]
        if let Some(executor) = self.transformer_tail_executor {
            if std::env::var_os("VESTRA_TRACE_DIR").is_none()
                && self.cfg.qknorm_start >= 0
                && self.cfg.alt_start >= self.cfg.qknorm_start
            {
                if let Some(cuda_executor) = executor.persistent_cuda_tail() {
                    return self.forward_multi_view_with_persistent_cuda_tail(
                        views,
                        gh,
                        gw,
                        out_layers,
                        cuda_executor,
                    );
                }
            }
        }
        assert!(
            !views.is_empty(),
            "multi-view forward needs at least one view"
        );
        let cfg = self.cfg;
        let embed = cfg.embed_dim as usize;
        let n = views[0].len() / embed;
        assert_eq!(views[0].len(), n * embed);
        for (index, view) in views.iter().enumerate() {
            assert_eq!(
                view.len(),
                n * embed,
                "view {index} has a different token shape"
            );
        }
        let view_count = views.len();
        let n_special = 1 + cfg.num_register as usize;
        assert_eq!(n, n_special + gh * gw);

        let trace_dir = std::env::var_os("VESTRA_TRACE_DIR").map(PathBuf::from);
        if let Some(trace_dir) = &trace_dir {
            std::fs::create_dir_all(trace_dir).unwrap_or_else(|error| {
                panic!(
                    "cannot create multi-view trace directory {}: {error}",
                    trace_dir.display()
                )
            });
        }
        trace_multi_view_tensor(trace_dir.as_ref(), "input", views);
        let mut local_x = views.to_vec();
        let mut feats = vec![vec![None; view_count]; out_layers.len()];
        let mut cam_tokens = vec![vec![None; view_count]; out_layers.len()];

        for layer_idx in 0..cfg.depth as usize {
            if cfg.alt_start >= 0 && layer_idx == cfg.alt_start as usize {
                let camera = self.weights.get_f32(CAMERA_TOKEN_WEIGHT).unwrap_or_else(|| {
                    panic!(
                        "missing weight tensor {CAMERA_TOKEN_WEIGHT:?} required by cfg.alt_start={}",
                        cfg.alt_start
                    )
                });
                assert!(
                    camera.len() >= embed,
                    "{CAMERA_TOKEN_WEIGHT} must contain a reference camera token"
                );
                if view_count > 1 {
                    assert!(
                        camera.len() >= 2 * embed,
                        "{CAMERA_TOKEN_WEIGHT} must contain reference and source camera tokens"
                    );
                }
                for (view_index, tokens) in views.iter_mut().enumerate() {
                    let slot = usize::from(view_index > 0);
                    let start = slot * embed;
                    tokens[..embed].copy_from_slice(&camera[start..start + embed]);
                }
            }

            let global =
                cfg.alt_start >= 0 && layer_idx as i32 >= cfg.alt_start && layer_idx % 2 == 1;
            if global {
                let mut flattened = Vec::with_capacity(view_count * n * embed);
                for view in views.iter() {
                    flattened.extend_from_slice(view);
                }
                vit_block_with_views(
                    &mut flattened,
                    n * view_count,
                    gh,
                    gw,
                    true,
                    view_count,
                    cfg,
                    layer_idx,
                    self.weights,
                    self.backend,
                    self.residual_executor,
                    self.mlp_executor,
                    self.attention_executor,
                    self.transformer_tail_executor,
                );
                for (view_index, view) in views.iter_mut().enumerate() {
                    let start = view_index * n * embed;
                    view.copy_from_slice(&flattened[start..start + n * embed]);
                }
            } else {
                for view in views.iter_mut() {
                    vit_block_with_residual(
                        view,
                        n,
                        gh,
                        gw,
                        false,
                        cfg,
                        layer_idx,
                        self.weights,
                        self.backend,
                        self.residual_executor,
                        self.mlp_executor,
                        self.attention_executor,
                        self.transformer_tail_executor,
                    );
                }
            }

            for (slot, &wanted) in out_layers.iter().enumerate() {
                if wanted == layer_idx as i32 {
                    for view_index in 0..view_count {
                        let local = if global {
                            &local_x[view_index]
                        } else {
                            &views[view_index]
                        };
                        let (feat, cam) =
                            self.post_process_capture(n, n_special, local, &views[view_index]);
                        feats[slot][view_index] = Some(feat);
                        cam_tokens[slot][view_index] = Some(cam);
                    }
                }
            }

            if !global {
                local_x.clone_from_slice(views);
            }
            trace_multi_view_tensor(trace_dir.as_ref(), &format!("block-{layer_idx}"), views);
        }

        let feats = feats
            .into_iter()
            .enumerate()
            .map(|(layer_slot, views)| {
                views
                    .into_iter()
                    .enumerate()
                    .map(|(view_index, value)| {
                        value.unwrap_or_else(|| {
                            panic!(
                                "out layer {} was not reached for view {view_index}",
                                out_layers[layer_slot]
                            )
                        })
                    })
                    .collect()
            })
            .collect();
        let cam_tokens = cam_tokens
            .into_iter()
            .enumerate()
            .map(|(layer_slot, views)| {
                views
                    .into_iter()
                    .enumerate()
                    .map(|(view_index, value)| {
                        value.unwrap_or_else(|| {
                            panic!(
                                "out layer {} was not reached for view {view_index}",
                                out_layers[layer_slot]
                            )
                        })
                    })
                    .collect()
            })
            .collect();

        MultiViewBackboneOutputs { feats, cam_tokens }
    }

    /// Device-resident counterpart of [`Self::forward_multi_view_ordered`].
    /// It retains one token tensor per view for local layers and constructs a
    /// temporary flattened tensor only for DA3's global layers. Both moves
    /// are CUDA device copies; observed activations cross to the host solely
    /// at DPT capture layers and at the final API boundary.
    #[cfg(feature = "cuda-residual-oracle")]
    fn forward_multi_view_with_persistent_cuda_tail(
        &self,
        views: &mut [Vec<f32>],
        gh: usize,
        gw: usize,
        out_layers: &[i32],
        executor: &crate::vit_block::CudaTransformerTailExecutor,
    ) -> MultiViewBackboneOutputs {
        assert!(
            !views.is_empty(),
            "multi-view forward needs at least one view"
        );
        let cfg = self.cfg;
        let embed = cfg.embed_dim as usize;
        let n = views[0].len() / embed;
        assert!(views.iter().all(|view| view.len() == n * embed));
        let view_count = views.len();
        let n_special = 1 + cfg.num_register as usize;
        assert_eq!(n, n_special + gh * gw);
        let first_device_layer = cfg.qknorm_start as usize;
        let view_values = n * embed;
        let phase_profile = std::env::var_os("DA_PHASE_PROFILE").is_some();
        let mut local_host = views.to_vec();
        let mut device_views: Option<Vec<vestra_kernels::cuda::CudaTensorF32>> = None;
        let mut device_local: Option<Vec<vestra_kernels::cuda::CudaTensorF32>> = None;
        let mut feats = vec![vec![None; view_count]; out_layers.len()];
        let mut cam_tokens = vec![vec![None; view_count]; out_layers.len()];

        for layer_idx in 0..cfg.depth as usize {
            let started = std::time::Instant::now();
            let global =
                cfg.alt_start >= 0 && layer_idx as i32 >= cfg.alt_start && layer_idx % 2 == 1;
            if layer_idx < first_device_layer {
                if cfg.alt_start >= 0 && layer_idx == cfg.alt_start as usize {
                    let camera = self
                        .weights
                        .get_f32(CAMERA_TOKEN_WEIGHT)
                        .unwrap_or_else(|| panic!("missing weight tensor {CAMERA_TOKEN_WEIGHT:?}"));
                    for (view_index, tokens) in views.iter_mut().enumerate() {
                        let start = usize::from(view_index > 0) * embed;
                        tokens[..embed].copy_from_slice(&camera[start..start + embed]);
                    }
                }
                if global {
                    let mut flattened = Vec::with_capacity(view_count * view_values);
                    for view in views.iter() {
                        flattened.extend_from_slice(view);
                    }
                    vit_block_with_views(
                        &mut flattened,
                        n * view_count,
                        gh,
                        gw,
                        true,
                        view_count,
                        cfg,
                        layer_idx,
                        self.weights,
                        self.backend,
                        None,
                        None,
                        None,
                        None,
                    );
                    for (view_index, view) in views.iter_mut().enumerate() {
                        let offset = view_index * view_values;
                        view.copy_from_slice(&flattened[offset..offset + view_values]);
                    }
                } else {
                    for view in views.iter_mut() {
                        vit_block_with_residual(
                            view,
                            n,
                            gh,
                            gw,
                            false,
                            cfg,
                            layer_idx,
                            self.weights,
                            self.backend,
                            None,
                            None,
                            None,
                            None,
                        );
                    }
                }
                local_host.clone_from_slice(views);
            } else {
                if device_views.is_none() {
                    device_views = Some(
                        views
                            .iter()
                            .map(|view| executor.upload_tokens(view).expect("CUDA tail upload"))
                            .collect(),
                    );
                }
                let states = device_views.as_mut().expect("device views exist");
                if cfg.alt_start >= 0 && layer_idx == cfg.alt_start as usize {
                    for (view_index, state) in states.iter_mut().enumerate() {
                        executor
                            .inject_camera_token_for_view(state, view_index, embed)
                            .expect("CUDA camera token injection");
                    }
                }
                if global {
                    let mut flattened = executor
                        .upload_tokens(&vec![0.0_f32; view_count * view_values])
                        .expect("CUDA global activation allocation");
                    for (view_index, state) in states.iter().enumerate() {
                        executor
                            .copy_token_segment_into(
                                &mut flattened,
                                view_index * view_values,
                                state,
                                0,
                                view_values,
                            )
                            .expect("CUDA global flatten");
                    }
                    let flattened = executor
                        .run_tail_device(
                            layer_idx,
                            flattened,
                            n * view_count,
                            gh,
                            gw,
                            true,
                            view_count,
                            cfg,
                        )
                        .expect("CUDA global tail");
                    *states = (0..view_count)
                        .map(|view_index| {
                            executor
                                .copy_token_segment(
                                    &flattened,
                                    view_index * view_values,
                                    view_values,
                                )
                                .expect("CUDA global split")
                        })
                        .collect();
                } else {
                    for state in states.iter_mut() {
                        let previous = executor.copy_tokens(state).expect("CUDA local input copy");
                        *state = executor
                            .run_tail_device(layer_idx, previous, n, gh, gw, false, 1, cfg)
                            .expect("CUDA local tail");
                    }
                }
            }

            if phase_profile {
                eprintln!(
                    "phase: multiview_transformer_block[{layer_idx}]={:.3}ms global={global} persistent_cuda={}",
                    started.elapsed().as_secs_f64() * 1e3,
                    layer_idx >= first_device_layer,
                );
            }
            if layer_idx >= first_device_layer {
                let states = device_views.as_ref().expect("device views exist");
                for (slot, &wanted) in out_layers.iter().enumerate() {
                    if wanted == layer_idx as i32 {
                        for view_index in 0..view_count {
                            let x = executor
                                .download_tokens(&states[view_index])
                                .expect("capture download");
                            let local = if global {
                                executor
                                    .download_tokens(
                                        &device_local.as_ref().expect("global capture local state")
                                            [view_index],
                                    )
                                    .expect("local capture download")
                            } else {
                                x.clone()
                            };
                            let (feat, cam) = self.post_process_capture(n, n_special, &local, &x);
                            feats[slot][view_index] = Some(feat);
                            cam_tokens[slot][view_index] = Some(cam);
                        }
                    }
                }
                let next_is_global = layer_idx + 1 < cfg.depth as usize
                    && cfg.alt_start >= 0
                    && (layer_idx + 1) as i32 >= cfg.alt_start
                    && (layer_idx + 1) % 2 == 1;
                if !global && next_is_global {
                    device_local = Some(
                        states
                            .iter()
                            .map(|state| executor.copy_tokens(state).expect("CUDA local snapshot"))
                            .collect(),
                    );
                }
            } else {
                for (slot, &wanted) in out_layers.iter().enumerate() {
                    if wanted == layer_idx as i32 {
                        for view_index in 0..view_count {
                            let local = if global {
                                &local_host[view_index]
                            } else {
                                &views[view_index]
                            };
                            let (feat, cam) =
                                self.post_process_capture(n, n_special, local, &views[view_index]);
                            feats[slot][view_index] = Some(feat);
                            cam_tokens[slot][view_index] = Some(cam);
                        }
                    }
                }
            }
        }
        for (view, state) in views
            .iter_mut()
            .zip(device_views.expect("final device views"))
        {
            *view = executor
                .download_tokens(&state)
                .expect("final CUDA download");
        }
        MultiViewBackboneOutputs {
            feats: feats
                .into_iter()
                .enumerate()
                .map(|(slot, outputs)| {
                    outputs
                        .into_iter()
                        .map(|value| {
                            value.unwrap_or_else(|| {
                                panic!("out layer {} missing capture", out_layers[slot])
                            })
                        })
                        .collect()
                })
                .collect(),
            cam_tokens: cam_tokens
                .into_iter()
                .enumerate()
                .map(|(slot, outputs)| {
                    outputs
                        .into_iter()
                        .map(|value| {
                            value.unwrap_or_else(|| {
                                panic!("out layer {} missing camera capture", out_layers[slot])
                            })
                        })
                        .collect()
                })
                .collect(),
        }
    }

    fn post_process_capture(
        &self,
        n: usize,
        n_special: usize,
        local_x: &[f32],
        x: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        let cfg = self.cfg;
        let embed = cfg.embed_dim as usize;
        let n_patch = n - n_special;
        let nw = self
            .weights
            .get_f32(VIT_NORM_WEIGHT)
            .unwrap_or_else(|| panic!("missing weight tensor {VIT_NORM_WEIGHT:?}"));
        let nb = self
            .weights
            .get_f32(VIT_NORM_BIAS)
            .unwrap_or_else(|| panic!("missing weight tensor {VIT_NORM_BIAS:?}"));

        if !cfg.cat_token {
            let cam = x[0..embed].to_vec();
            let mut feat = x[n_special * embed..n * embed].to_vec();
            vestra_kernels::scalar::layernorm(&mut feat, n_patch, embed, nw, nb, cfg.ln_eps);
            return (feat, cam);
        }

        let mut cam = Vec::with_capacity(2 * embed);
        cam.extend_from_slice(&local_x[0..embed]);
        cam.extend_from_slice(&x[0..embed]);
        let mut normed_x = x[n_special * embed..n * embed].to_vec();
        vestra_kernels::scalar::layernorm(&mut normed_x, n_patch, embed, nw, nb, cfg.ln_eps);
        let mut feat = vec![0f32; n_patch * 2 * embed];
        for t in 0..n_patch {
            let lrow = &local_x[(n_special + t) * embed..(n_special + t + 1) * embed];
            let nrow = &normed_x[t * embed..(t + 1) * embed];
            let dst = &mut feat[t * 2 * embed..(t + 1) * 2 * embed];
            dst[0..embed].copy_from_slice(lrow);
            dst[embed..2 * embed].copy_from_slice(nrow);
        }
        (feat, cam)
    }

    /// Host post-process matching `get_intermediate_layers` / the C++
    /// reference's post-loop code (see module doc comment items 3-4).
    /// `raw_local[o]`/`raw_x[o]` are each `[n * embed_dim]` token-major
    /// (row `t` = token `t`'s `embed_dim` channels).
    fn post_process(
        &self,
        n: usize,
        n_special: usize,
        raw_local: &[Vec<f32>],
        raw_x: &[Vec<f32>],
    ) -> BackboneOutputs {
        let cfg = self.cfg;
        let embed = cfg.embed_dim as usize;
        let n_patch = n - n_special;
        let nl = raw_local.len();

        let mut feats = Vec::with_capacity(nl);
        let mut cam_tokens = Vec::with_capacity(nl);

        let nw = self
            .weights
            .get_f32(VIT_NORM_WEIGHT)
            .unwrap_or_else(|| panic!("missing weight tensor {VIT_NORM_WEIGHT:?}"));
        let nb = self
            .weights
            .get_f32(VIT_NORM_BIAS)
            .unwrap_or_else(|| panic!("missing weight tensor {VIT_NORM_BIAS:?}"));

        if !cfg.cat_token {
            for o in 0..nl {
                let xx = &raw_x[o];
                // cam = x[token0] RAW (unused on this branch, kept for
                // shape consistency per the C++ reference's comment).
                cam_tokens.push(xx[0..embed].to_vec());
                // feat = vit_norm(x), patches n_special..n (token-0/CLS and
                // any register tokens stripped).
                let mut f = xx[n_special * embed..n * embed].to_vec();
                vestra_kernels::scalar::layernorm(&mut f, n_patch, embed, nw, nb, cfg.ln_eps);
                feats.push(f);
            }
            return BackboneOutputs { feats, cam_tokens };
        }

        // cat_token == true (DA3-BASE/giant): doubled-width feat/cam.
        for o in 0..nl {
            let lx = &raw_local[o];
            let xx = &raw_x[o];

            // cam = cat([local_x[token0], x[token0]]) RAW (no norm on
            // either half).
            let mut cam = Vec::with_capacity(2 * embed);
            cam.extend_from_slice(&lx[0..embed]);
            cam.extend_from_slice(&xx[0..embed]);
            cam_tokens.push(cam);

            // feat[patch] = cat([local_x_raw[patch], vit_norm(x)[patch]]),
            // for patches n_special..n (token-0/CLS and any register tokens
            // stripped). Normalize x's patch rows first (layernorm operates
            // row-major over n_patch rows of embed_dim), then interleave.
            let mut normed_x = xx[n_special * embed..n * embed].to_vec();
            vestra_kernels::scalar::layernorm(&mut normed_x, n_patch, embed, nw, nb, cfg.ln_eps);

            let mut f = vec![0f32; n_patch * 2 * embed];
            for t in 0..n_patch {
                let lrow = &lx[(n_special + t) * embed..(n_special + t + 1) * embed];
                let nrow = &normed_x[t * embed..(t + 1) * embed];
                let dst = &mut f[t * 2 * embed..(t + 1) * 2 * embed];
                dst[0..embed].copy_from_slice(lrow);
                dst[embed..2 * embed].copy_from_slice(nrow);
            }
            feats.push(f);
        }
        BackboneOutputs { feats, cam_tokens }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vit_block::QK_NORM_EPS;
    use da_graph::CpuBackend;

    fn test_cfg(depth: u32) -> ModelConfig {
        ModelConfig {
            arch: "depthanything3".to_string(),
            patch_size: 14,
            image_size: 28,
            embed_dim: 8,
            depth,
            num_heads: 2,
            head_dim: 4,
            mlp_hidden: 16,
            num_register: 0,
            rope_start: -1,
            qknorm_start: -1,
            rope_freq: 100.0,
            ln_eps: 1e-6,
            out_layers: vec![],
            ffn_type: "mlp".to_string(),
            alt_start: -1,
            cat_token: true,
            head_features: 1,
            head_max_depth: 1.0,
            img_mean: [0.0, 0.0, 0.0],
            img_std: [1.0, 1.0, 1.0],
            img_resize_mode: "bilinear".to_string(),
            cam_dim_in: 1,
            head_pos_embed: true,
        }
    }

    fn synthetic_weights(cfg: &ModelConfig) -> Weights {
        let embed = cfg.embed_dim as usize;
        let mlp_hidden = cfg.mlp_hidden as usize;
        let mut rng: u32 = 0x1357_9BDF;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            ((rng as f32) / (u32::MAX as f32)) * 2.0 - 1.0
        };
        let mut w = Weights::new();
        for layer_idx in 0..cfg.depth as usize {
            let mut put = |name: String, len: usize, w: &mut Weights| {
                w.insert_f32(name, (0..len).map(|_| next()).collect::<Vec<f32>>());
            };
            put(format!("vit.blk.{layer_idx}.norm1.weight"), embed, &mut w);
            put(format!("vit.blk.{layer_idx}.norm1.bias"), embed, &mut w);
            put(format!("vit.blk.{layer_idx}.norm2.weight"), embed, &mut w);
            put(format!("vit.blk.{layer_idx}.norm2.bias"), embed, &mut w);
            put(
                format!("vit.blk.{layer_idx}.attn_qkv.weight"),
                embed * 3 * embed,
                &mut w,
            );
            put(
                format!("vit.blk.{layer_idx}.attn_qkv.bias"),
                3 * embed,
                &mut w,
            );
            put(
                format!("vit.blk.{layer_idx}.attn_proj.weight"),
                embed * embed,
                &mut w,
            );
            put(format!("vit.blk.{layer_idx}.attn_proj.bias"), embed, &mut w);
            put(
                format!("vit.blk.{layer_idx}.mlp_fc1.weight"),
                embed * mlp_hidden,
                &mut w,
            );
            put(
                format!("vit.blk.{layer_idx}.mlp_fc1.bias"),
                mlp_hidden,
                &mut w,
            );
            put(
                format!("vit.blk.{layer_idx}.mlp_fc2.weight"),
                mlp_hidden * embed,
                &mut w,
            );
            put(format!("vit.blk.{layer_idx}.mlp_fc2.bias"), embed, &mut w);
        }
        // vit.norm + camera_token, always inserted (harmless when
        // cfg.alt_start == -1: camera_token simply goes unused).
        w.insert_f32("vit.norm.weight".to_string(), vec![1.0; embed]);
        w.insert_f32("vit.norm.bias".to_string(), vec![0.0; embed]);
        let mut rng2: u32 = 0xC0DE_1234;
        let mut next2 = move || {
            rng2 ^= rng2 << 13;
            rng2 ^= rng2 >> 17;
            rng2 ^= rng2 << 5;
            ((rng2 as f32) / (u32::MAX as f32)) * 2.0 - 1.0
        };
        w.insert_f32(
            "vit.camera_token".to_string(),
            (0..2 * embed).map(|_| next2()).collect::<Vec<f32>>(),
        );
        w
    }

    #[test]
    fn forward_collects_captures_at_out_layers_in_requested_order() {
        let cfg = test_cfg(6);
        let weights = synthetic_weights(&cfg);
        let backend = CpuBackend::new();
        let embed = cfg.embed_dim as usize;
        let n = 5usize; // 1 CLS + 2x2 patch grid
        let mut rng: u32 = 0xDEAD_BEEF;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            ((rng as f32) / (u32::MAX as f32)) * 2.0 - 1.0
        };
        let mut tokens: Vec<f32> = (0..n * embed).map(|_| next()).collect();

        let bb = Backbone::new(&cfg, &weights, &backend);
        // Deliberately out of increasing order to prove captures follow
        // `out_layers`'s order, not execution order.
        let out_layers = [3, 1, 5];
        let out = bb.forward(&mut tokens, 2, 2, &out_layers);

        assert_eq!(out.feats.len(), 3);
        assert_eq!(out.cam_tokens.len(), 3);
        for f in &out.feats {
            // cat_token == true and alt_start == -1: width = 2*embed, rows = n_patch (4).
            assert_eq!(f.len(), 4 * 2 * embed);
            assert!(f.iter().all(|v| v.is_finite()));
        }
        for c in &out.cam_tokens {
            assert_eq!(c.len(), 2 * embed);
            assert!(c.iter().all(|v| v.is_finite()));
        }
        // Different layer indices captured on a non-trivial forward pass
        // must generally produce different snapshots.
        assert_ne!(out.feats[0], out.feats[1]);
        assert_ne!(out.feats[1], out.feats[2]);
    }

    #[test]
    #[should_panic(expected = "was never reached")]
    fn forward_panics_if_out_layer_exceeds_depth() {
        let cfg = test_cfg(2);
        let weights = synthetic_weights(&cfg);
        let backend = CpuBackend::new();
        let embed = cfg.embed_dim as usize;
        let n = 5usize;
        let mut tokens = vec![0f32; n * embed];
        let bb = Backbone::new(&cfg, &weights, &backend);
        let _ = bb.forward(&mut tokens, 2, 2, &[10]);
    }

    #[test]
    fn qk_norm_eps_constant_is_reexported_and_matches_spec() {
        assert_eq!(QK_NORM_EPS, 1e-5);
    }

    #[test]
    fn cat_token_false_produces_single_width_outputs() {
        let mut cfg = test_cfg(3);
        cfg.cat_token = false;
        let weights = synthetic_weights(&cfg);
        let backend = CpuBackend::new();
        let embed = cfg.embed_dim as usize;
        let n = 5usize;
        let mut rng: u32 = 0xAAAA_5555;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            ((rng as f32) / (u32::MAX as f32)) * 2.0 - 1.0
        };
        let mut tokens: Vec<f32> = (0..n * embed).map(|_| next()).collect();
        let bb = Backbone::new(&cfg, &weights, &backend);
        let out = bb.forward(&mut tokens, 2, 2, &[2]);
        assert_eq!(out.feats[0].len(), 4 * embed);
        assert_eq!(out.cam_tokens[0].len(), embed);
    }

    #[test]
    fn alt_start_injects_camera_token_and_alternates_global_attention() {
        // alt_start=2, depth=4: layers 0,1 local; layer 2 (even, i>=alt_start) local;
        // layer 3 (odd, i>=alt_start) global. Camera token overwrites token-0 right
        // before layer 2 runs. Just check this runs without panicking and produces
        // finite output of the expected doubled shape — full numerical parity is
        // gated on real dumps (see module doc comment).
        let mut cfg = test_cfg(4);
        cfg.alt_start = 2;
        cfg.out_layers = vec![3];
        let weights = synthetic_weights(&cfg);
        let backend = CpuBackend::new();
        let embed = cfg.embed_dim as usize;
        let n = 5usize;
        let mut rng: u32 = 0x1111_2222;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            ((rng as f32) / (u32::MAX as f32)) * 2.0 - 1.0
        };
        let mut tokens: Vec<f32> = (0..n * embed).map(|_| next()).collect();
        let bb = Backbone::new(&cfg, &weights, &backend);
        let out = bb.forward(&mut tokens, 2, 2, &[3]);
        assert_eq!(out.feats[0].len(), 4 * 2 * embed);
        assert_eq!(out.cam_tokens[0].len(), 2 * embed);
        assert!(out.feats[0].iter().all(|v| v.is_finite()));
        assert!(out.cam_tokens[0].iter().all(|v| v.is_finite()));
    }

    #[test]
    fn ordered_multiview_s1_is_bitwise_equal_to_single_view() {
        let mut cfg = test_cfg(4);
        cfg.alt_start = 2;
        cfg.out_layers = vec![1, 3];
        let weights = synthetic_weights(&cfg);
        let backend = CpuBackend::new();
        let embed = cfg.embed_dim as usize;
        let mut tokens: Vec<f32> = (0..5 * embed)
            .map(|index| index as f32 * 0.013 - 0.4)
            .collect();
        let mut views = vec![tokens.clone()];
        let bb = Backbone::new(&cfg, &weights, &backend);

        let single = bb.forward(&mut tokens, 2, 2, &cfg.out_layers);
        let multi = bb.forward_multi_view_ordered(&mut views, 2, 2, &cfg.out_layers);

        assert_eq!(multi.feats.len(), single.feats.len());
        for layer in 0..single.feats.len() {
            assert_eq!(multi.feats[layer][0], single.feats[layer]);
            assert_eq!(multi.cam_tokens[layer][0], single.cam_tokens[layer]);
        }
        assert_eq!(views[0], tokens);
    }

    #[test]
    fn ordered_multiview_global_attention_couples_views() {
        let mut cfg = test_cfg(4);
        cfg.alt_start = 2;
        cfg.out_layers = vec![3];
        let weights = synthetic_weights(&cfg);
        let backend = CpuBackend::new();
        let embed = cfg.embed_dim as usize;
        let first: Vec<f32> = (0..5 * embed)
            .map(|index| index as f32 * 0.01 - 0.2)
            .collect();
        let near = first.iter().map(|value| value + 0.01).collect::<Vec<_>>();
        let far = first.iter().map(|value| 2.0 - value).collect::<Vec<_>>();
        let bb = Backbone::new(&cfg, &weights, &backend);

        let mut near_pair = vec![first.clone(), near];
        let mut far_pair = vec![first, far];
        let near_out = bb.forward_multi_view_ordered(&mut near_pair, 2, 2, &cfg.out_layers);
        let far_out = bb.forward_multi_view_ordered(&mut far_pair, 2, 2, &cfg.out_layers);

        assert_ne!(near_out.feats[0][0], far_out.feats[0][0]);
        assert_ne!(near_pair[0], far_pair[0]);
    }

    #[test]
    fn saddle_reference_matches_locked_cpp_formula() {
        let cls = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.7, 0.6, 0.1, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![-1.0, 0.0, 0.0, 0.0],
        ];
        assert_eq!(select_reference_view_saddle(&cls), 0);
    }

    #[test]
    fn reference_order_moves_reference_to_front_without_losing_views() {
        assert_eq!(reference_first_order(5, 3), vec![3, 0, 1, 2, 4]);
        assert_eq!(reference_first_order(3, 0), vec![0, 1, 2]);
    }

    #[test]
    fn preliminary_local_cls_pass_matches_manual_local_blocks() {
        let cfg = test_cfg(3);
        let weights = synthetic_weights(&cfg);
        let backend = CpuBackend::new();
        let embed = cfg.embed_dim as usize;
        let original = (0..5 * embed)
            .map(|index| index as f32 * 0.017 - 0.25)
            .collect::<Vec<_>>();
        let mut manual = original.clone();
        let bb = Backbone::new(&cfg, &weights, &backend);

        for layer_idx in 0..2 {
            vit_block(
                &mut manual,
                5,
                2,
                2,
                false,
                &cfg,
                layer_idx,
                &weights,
                &backend,
            );
        }
        let cls = bb.capture_local_cls(&[original], 2, 2, 2);
        assert_eq!(cls, vec![manual[..embed].to_vec()]);
    }
}
