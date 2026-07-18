//! Runs the full 12-layer (or however deep `cfg.depth` says) ViT block
//! stack, collecting a snapshot of the token buffer after each layer index
//! listed in `out_layers` (`feat_5`/`feat_7`/`feat_9`/`feat_11` for
//! DA3-BASE, whose `depthanything3.vit.out_layers` metadata is `[5,7,9,11]`
//! — see `../scripts/gguf_keys.py`).
//!
//! ## What this intentionally does NOT do (honest scope note)
//!
//! The real C++ reference's `DinoBackbone::forward`
//! (`../src/dino_backbone.cpp`) does more than "run `vit_block` `depth`
//! times": it also (a) tracks a separate `local_x`/`x` pair and swaps in a
//! learned camera token at `alt_start` for models with alternating
//! local/global attention, and (b) host-post-processes the raw per-layer
//! capture into `feat = cat([local_x, vit_norm(x)])` (doubling the channel
//! width) and `cam = cat([local_x[token0], x[token0]])` for the
//! `cam_token_*` dumps mentioned in this task's brief. `ModelConfig` (Task
//! 14) has no `alt_start` field, and DA3-BASE's converter output implies
//! `alt_start` is unused for this model (matching the brief's own
//! description of the block, which never mentions a camera-token swap) —
//! so `(a)` doesn't apply to the model this task's dumps target. `(b)` is
//! genuinely out of scope for the interface this task specifies
//! (`Backbone::forward(&self, tokens, out_layers) -> Vec<Vec<f32>>`, i.e.
//! *one* Vec per out-layer): `Backbone::forward` here returns the raw
//! per-layer token buffer (post-block, pre-final-norm), not the
//! doubled-width `vit_norm`-concatenated `feat` the real engine's host
//! post-processing produces. Whichever later task actually wires up
//! `backbone_parity.rs` against real `feat_{5,7,9,11}` dumps will need to
//! either add that post-processing here or confirm the dumps are the raw
//! per-layer capture instead — this is UNVERIFIED either way in this
//! environment (no dumps present).
use crate::vit_block::vit_block;
use crate::ModelConfig;
use da_graph::{Backend, Weights};

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

    /// Runs `cfg.depth` `vit_block` calls over `tokens` in place, collecting
    /// a clone of the token buffer after every layer index listed in
    /// `out_layers` (in `out_layers` order, not layer order, matching
    /// the order callers list them).
    ///
    /// `tokens` must already be `prepare_tokens`'s output
    /// (`n = 1 + num_register + gh*gw` rows of `embed_dim`); `gh`/`gw` are
    /// the patch-grid resolution (needed for RoPE position derivation on
    /// any layer `>= cfg.rope_start`).
    pub fn forward(&self, tokens: &mut [f32], gh: usize, gw: usize, out_layers: &[i32]) -> Vec<Vec<f32>> {
        let embed = self.cfg.embed_dim as usize;
        assert_eq!(tokens.len() % embed, 0, "tokens length must be a multiple of embed_dim");
        let n = tokens.len() / embed;

        let mut captures: Vec<Option<Vec<f32>>> = vec![None; out_layers.len()];
        for layer_idx in 0..self.cfg.depth as usize {
            vit_block(tokens, n, gh, gw, self.cfg, layer_idx, self.weights, self.backend);
            for (slot, &wanted) in out_layers.iter().enumerate() {
                if wanted == layer_idx as i32 {
                    captures[slot] = Some(tokens.to_vec());
                }
            }
        }

        captures
            .into_iter()
            .enumerate()
            .map(|(i, c)| c.unwrap_or_else(|| panic!("out_layers[{i}]={} was never reached (depth={})", out_layers[i], self.cfg.depth)))
            .collect()
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
            head_features: 1,
            head_max_depth: 1.0,
            img_mean: [0.0, 0.0, 0.0],
            img_std: [1.0, 1.0, 1.0],
            img_resize_mode: "bilinear".to_string(),
            cam_dim_in: 1,
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
        let feats = bb.forward(&mut tokens, 2, 2, &out_layers);

        assert_eq!(feats.len(), 3);
        for f in &feats {
            assert_eq!(f.len(), n * embed);
            assert!(f.iter().all(|v| v.is_finite()));
        }
        // Different layer indices captured on a non-trivial forward pass
        // must generally produce different snapshots (each is a further
        // 1-2 block applications away from the last).
        assert_ne!(feats[0], feats[1]);
        assert_ne!(feats[1], feats[2]);
        // The final full-depth run (depth=6) must match feats[2] (layer 5,
        // the last layer) exactly, since nothing runs after out_layers'
        // last requested index in this test's depth=6 setup.
        assert_eq!(tokens, feats[2]);
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
}
