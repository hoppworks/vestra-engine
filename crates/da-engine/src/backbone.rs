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
use crate::vit_block::vit_block;
use crate::ModelConfig;
use da_graph::{Backend, Weights};

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

/// Owns nothing — just bundles the `cfg`/`weights`/`backend` a full
/// backbone forward pass needs, borrowed for the duration of `forward`.
pub struct Backbone<'a> {
    pub cfg: &'a ModelConfig,
    pub weights: &'a Weights,
    pub backend: &'a dyn Backend,
}

impl<'a> Backbone<'a> {
    pub fn new(cfg: &'a ModelConfig, weights: &'a Weights, backend: &'a dyn Backend) -> Self {
        Backbone { cfg, weights, backend }
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
    pub fn forward(&self, tokens: &mut [f32], gh: usize, gw: usize, out_layers: &[i32]) -> BackboneOutputs {
        let cfg = self.cfg;
        let embed = cfg.embed_dim as usize;
        assert_eq!(tokens.len() % embed, 0, "tokens length must be a multiple of embed_dim");
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

        let mut raw_local: Vec<Option<Vec<f32>>> = vec![None; out_layers.len()];
        let mut raw_x: Vec<Option<Vec<f32>>> = vec![None; out_layers.len()];

        for layer_idx in 0..cfg.depth as usize {
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

            let global = cfg.alt_start >= 0 && (layer_idx as i32) >= cfg.alt_start && layer_idx % 2 == 1;

            vit_block(tokens, n, gh, gw, global, cfg, layer_idx, self.weights, self.backend);

            if !global {
                local_x.copy_from_slice(tokens);
            }

            for (slot, &wanted) in out_layers.iter().enumerate() {
                if wanted == layer_idx as i32 {
                    raw_local[slot] = Some(local_x.clone());
                    raw_x[slot] = Some(tokens.to_vec());
                }
            }
        }

        let raw_local: Vec<Vec<f32>> = raw_local
            .into_iter()
            .enumerate()
            .map(|(i, c)| c.unwrap_or_else(|| panic!("out_layers[{i}]={} was never reached (depth={})", out_layers[i], cfg.depth)))
            .collect();
        let raw_x: Vec<Vec<f32>> = raw_x
            .into_iter()
            .enumerate()
            .map(|(i, c)| c.unwrap_or_else(|| panic!("out_layers[{i}]={} was never reached (depth={})", out_layers[i], cfg.depth)))
            .collect();

        self.post_process(n, n_special, &raw_local, &raw_x)
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
                da_kernels::scalar::layernorm(&mut f, n_patch, embed, nw, nb, cfg.ln_eps);
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
            da_kernels::scalar::layernorm(&mut normed_x, n_patch, embed, nw, nb, cfg.ln_eps);

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
            put(format!("vit.blk.{layer_idx}.attn_qkv.weight"), embed * 3 * embed, &mut w);
            put(format!("vit.blk.{layer_idx}.attn_qkv.bias"), 3 * embed, &mut w);
            put(format!("vit.blk.{layer_idx}.attn_proj.weight"), embed * embed, &mut w);
            put(format!("vit.blk.{layer_idx}.attn_proj.bias"), embed, &mut w);
            put(format!("vit.blk.{layer_idx}.mlp_fc1.weight"), embed * mlp_hidden, &mut w);
            put(format!("vit.blk.{layer_idx}.mlp_fc1.bias"), mlp_hidden, &mut w);
            put(format!("vit.blk.{layer_idx}.mlp_fc2.weight"), mlp_hidden * embed, &mut w);
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
        w.insert_f32("vit.camera_token".to_string(), (0..2 * embed).map(|_| next2()).collect::<Vec<f32>>());
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
}
