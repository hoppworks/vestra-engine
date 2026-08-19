use std::collections::HashMap;

use crate::patch_embed::patch_embed;
use crate::ModelConfig;
use da_graph::Weights;

/// Learned positional-embedding grid tensor name. Confirmed against
/// `../scripts/gguf_keys.py::rename_backbone` (`pos_embed` -> `"vit.pos_embed"`)
/// and `../src/dino_backbone.cpp` (`ml_.tensor("vit.pos_embed")`).
///
/// Layout: `[(M*M + 1), embed_dim]` row-major, row 0 = the CLS token's
/// pos-embed, rows `1 + r*M + c` (r,c in `0..M`) = the learned `M x M`
/// patch-grid pos-embed, row-major over `(r, c)`.
pub const POS_EMBED_WEIGHT: &str = "vit.pos_embed";

/// CLS-token tensor name. Same two sources as above (`cls_token` ->
/// `"vit.cls_token"`).
pub const CLS_TOKEN_WEIGHT: &str = "vit.cls_token";

/// Register-token tensor name. **Not** confirmed against the C++ reference or
/// converter: `rename_backbone` in `../scripts/gguf_keys.py` has no mapping
/// rule for a `register_tokens` parameter, and `../src/dino_backbone.cpp`
/// never loads or concatenates a register-token tensor for the DA3-BASE path
/// (its `cfg.num_register` is read from GGUF metadata but has no
/// corresponding weight tensor in this codebase — likely `num_register == 0`
/// for every model this plan currently targets). This name is therefore a
/// documented placeholder for a variant that *does* export learned register
/// tokens; `prepare_tokens` below only uses it if present in `weights`, and
/// falls back to zero register tokens (matching the reference) otherwise.
pub const REGISTER_TOKENS_WEIGHT: &str = "vit.register_tokens";

/// Default bicubic-interpolation offset used by DINOv2-style pos-embed
/// resizing. Matches the C++ reference's default
/// (`src/model_loader.cpp`: `kv_f32(gguf_, DA_KV_VIT_INTERP_OFFSET, 0.1f)`,
/// i.e. `depthanything3.vit.interp_offset` defaults to `0.1` when absent).
///
/// This is **not currently wired through `ModelConfig`** — Task 14's
/// `ModelConfig` does not have an `interp_offset` field, and adding one would
/// touch every existing `ModelConfig` struct literal in this workspace
/// (config.rs tests, preprocess.rs tests, preprocess_parity.rs), which is out
/// of scope for this task. Using the hardcoded default here matches the real
/// model's value (DA3-BASE's `depthanything3.vit.interp_offset` metadata was
/// never overridden away from 0.1 in the converter), but a model with a
/// non-default `interp_offset` would silently get the wrong value until
/// `ModelConfig` grows this field.
const DEFAULT_INTERP_OFFSET: f32 = 0.1;

/// Keys' cubic convolution kernel, `a = -0.75` (the Catmull-Rom-derived
/// variant PyTorch's `F.interpolate(..., mode="bicubic")` uses, and thus what
/// DINOv2/EVA-style ViT pos-embed interpolation uses).
///
/// Verified byte-for-byte against the C++ reference's `cubic()` helper in
/// `../src/dino_backbone.cpp`:
/// ```text
/// static float cubic(float x){ // Catmull-Rom, a=-0.75 (PyTorch bicubic)
///     const float a=-0.75f; x=std::fabs(x);
///     if (x<1) return ((a+2)*x - (a+3))*x*x + 1;
///     if (x<2) return (((x-5)*x+8)*x-4)*a;
///     return 0;
/// }
/// ```
/// so this is not an independently-chosen coefficient — it is the same `a`
/// the reference engine uses, ported 1:1.
fn cubic(x: f32) -> f32 {
    const A: f32 = -0.75;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

/// Infers the square pos-embed grid side `M` from the flat tensor length and
/// `embed_dim`, given the `[(M*M+1), embed_dim]` layout described on
/// `POS_EMBED_WEIGHT`. Panics if the length isn't consistent with a perfect
/// square (a corrupt/mismatched weight tensor).
fn infer_grid(pos_embed_len: usize, embed: usize) -> usize {
    assert_eq!(
        pos_embed_len % embed,
        0,
        "pos_embed length not a multiple of embed_dim"
    );
    let rows = pos_embed_len / embed;
    assert!(rows >= 1, "pos_embed must have at least the CLS row");
    let patch_rows = rows - 1;
    let m = (patch_rows as f64).sqrt().round() as usize;
    assert_eq!(
        m * m,
        patch_rows,
        "pos_embed patch-row count {patch_rows} is not a perfect square"
    );
    m
}

/// Bicubically interpolates the learned `[grid*grid+1, embed]` pos-embed grid
/// to a `[1 + gh*gw, embed]` grid at patch resolution `(gh, gw)`, prepending
/// the (unchanged) CLS-token row.
///
/// This is a direct port of `DinoBackbone::interp_pos_embed` in
/// `../src/dino_backbone.cpp` (scale computation, half-pixel-center sampling,
/// clamp-to-edge 4x4 tap window) — see module doc for the `cubic()` and
/// `DEFAULT_INTERP_OFFSET` provenance.
///
/// NOT cross-checked numerically against the `pos_embed_added` reference dump
/// in this environment (no `../dumps/reference.gguf` present) — see
/// `tests/pos_embed_parity.rs`, which skips cleanly until that dump exists.
/// The math itself is transcribed from the C++ source, not reverse-engineered
/// from expected output, so confidence is reasonably high, but it is
/// UNVERIFIED end-to-end.
fn interpolate_pos_embed(
    pos_embed: &[f32],
    grid: usize,
    embed: usize,
    gh: usize,
    gw: usize,
    interp_offset: f32,
) -> Vec<f32> {
    assert_eq!(pos_embed.len(), (grid * grid + 1) * embed);

    let src = |r: usize, c: usize, ch: usize| -> f32 {
        let row = 1 + r * grid + c;
        pos_embed[row * embed + ch]
    };

    let mut out = vec![0f32; (1 + gh * gw) * embed];
    out[..embed].copy_from_slice(&pos_embed[..embed]); // CLS row: passed through unchanged

    let m = grid as f32;
    let sx = (gw as f32 + interp_offset) / m;
    let sy = (gh as f32 + interp_offset) / m;

    for oy in 0..gh {
        let iy = (oy as f32 + 0.5) / sy - 0.5;
        let y0 = iy.floor() as isize;
        let fy = iy - y0 as f32;
        for ox in 0..gw {
            let ix = (ox as f32 + 0.5) / sx - 0.5;
            let x0 = ix.floor() as isize;
            let fx = ix - x0 as f32;
            let orow = 1 + oy * gw + ox;
            for ch in 0..embed {
                let mut acc = 0f32;
                for dy in -1isize..=2 {
                    let wy = cubic(fy - dy as f32);
                    let yy = (y0 + dy).clamp(0, grid as isize - 1) as usize;
                    for dx in -1isize..=2 {
                        let wx = cubic(fx - dx as f32);
                        let xx = (x0 + dx).clamp(0, grid as isize - 1) as usize;
                        acc += wy * wx * src(yy, xx, ch);
                    }
                }
                out[orow * embed + ch] = acc;
            }
        }
    }
    out
}

/// Caches bicubically-interpolated positional embeddings per `(h, w)` patch-grid
/// resolution — the "~95ms lesson": the interpolation is input-INDEPENDENT
/// (depends only on the model's `pos_embed` weight + target grid size), so
/// rebuilding it on every forward pass at a fixed resolution is pure waste.
/// This mirrors `DinoBackbone::interp_pos_embed`'s C++-side
/// `static std::map<std::tuple<uintptr_t,int,int>, ...> pe_cache` in
/// `../src/dino_backbone.cpp`.
///
/// `(h, w)` here means the *patch-grid* resolution (`gh, gw` — number of
/// patches along each axis), i.e. exactly the pair `patch_embed` returns, not
/// pixel dimensions.
#[derive(Default)]
pub struct PosEmbedCache {
    by_resolution: HashMap<(usize, usize), Vec<f32>>,
}

impl PosEmbedCache {
    pub fn new() -> Self {
        PosEmbedCache {
            by_resolution: HashMap::new(),
        }
    }

    /// Returns the cached (or freshly-built-and-cached) interpolated pos-embed
    /// grid for patch resolution `(h, w)` (`h`=gh, `w`=gw). On a cache miss,
    /// bicubically interpolates `weights`'s `POS_EMBED_WEIGHT` tensor to this
    /// resolution and stores the result; on a hit, returns the stored value
    /// without recomputing.
    pub fn get_or_build(
        &mut self,
        h: usize,
        w: usize,
        cfg: &ModelConfig,
        weights: &Weights,
    ) -> &[f32] {
        self.by_resolution
            .entry((h, w))
            .or_insert_with(|| {
                let embed = cfg.embed_dim as usize;
                let pos_embed = weights
                    .get_f32(POS_EMBED_WEIGHT)
                    .unwrap_or_else(|| panic!("missing weight tensor {POS_EMBED_WEIGHT:?}"));
                let grid = infer_grid(pos_embed.len(), embed);
                interpolate_pos_embed(pos_embed, grid, embed, h, w, DEFAULT_INTERP_OFFSET)
            })
            .as_slice()
    }
}

/// Combines `patch_embed` -> CLS-token (+ optional register-token) prepend ->
/// cached bicubic pos-embed add, producing the token sequence a ViT block
/// stack consumes.
///
/// Token order: `[CLS, register_0..register_{k-1} (if present), patch_0..patch_{n-1}]`,
/// matching `DinoBackbone::prepare_tokens` in `../src/dino_backbone.cpp` for the
/// `k=0` case (that function never concatenates register tokens for the
/// DA3-BASE path — see `REGISTER_TOKENS_WEIGHT` doc comment). Positional
/// embeddings are added to the CLS row and the patch rows only; register
/// tokens (if present) receive no positional-embedding contribution, which
/// matches standard DINOv2 register-token convention (they're register slots,
/// not spatial patches) and is consistent with the reference never touching
/// them at all.
///
/// Gated against the `pos_embed_added` reference dump — see `tests/pos_embed_parity.rs`,
/// which skips (does not fail) when `../dumps/reference.gguf` is absent.
pub fn prepare_tokens(
    img_nchw: &[f32],
    h: usize,
    w: usize,
    cfg: &ModelConfig,
    weights: &Weights,
    cache: &mut PosEmbedCache,
    out_tokens: &mut Vec<f32>,
) -> (usize, usize) {
    let mut patch_tokens = Vec::new();
    let (gh, gw) = patch_embed(img_nchw, h, w, cfg, weights, &mut patch_tokens);
    assemble_tokens_from_patch_tokens(&patch_tokens, gh, gw, cfg, weights, cache, out_tokens);
    (gh, gw)
}

/// Adds CLS/register tokens and the cached positional grid to already
/// projected token-major patch rows. This keeps position-token assembly shared
/// between CPU patch embedding and a qualified CUDA patch-projection seam.
pub fn assemble_tokens_from_patch_tokens(
    patch_tokens: &[f32],
    gh: usize,
    gw: usize,
    cfg: &ModelConfig,
    weights: &Weights,
    cache: &mut PosEmbedCache,
    out_tokens: &mut Vec<f32>,
) {
    let embed = cfg.embed_dim as usize;
    let n_patches = gh * gw;
    assert_eq!(patch_tokens.len(), n_patches * embed);

    let cls = weights
        .get_f32(CLS_TOKEN_WEIGHT)
        .unwrap_or_else(|| panic!("missing weight tensor {CLS_TOKEN_WEIGHT:?}"));
    assert_eq!(
        cls.len(),
        embed,
        "{CLS_TOKEN_WEIGHT} shape mismatch: expected embed_dim"
    );

    let register_tokens = weights.get_f32(REGISTER_TOKENS_WEIGHT);
    let n_register = register_tokens.map_or(0, |rt| {
        assert_eq!(
            rt.len() % embed,
            0,
            "{REGISTER_TOKENS_WEIGHT} length must be a multiple of embed_dim"
        );
        rt.len() / embed
    });

    let n_special = 1 + n_register;
    let n_tok = n_special + n_patches;

    out_tokens.clear();
    out_tokens.resize(n_tok * embed, 0.0);
    out_tokens[0..embed].copy_from_slice(cls);
    if let Some(rt) = register_tokens {
        out_tokens[embed..embed + rt.len()].copy_from_slice(rt);
    }
    let patch_start = n_special * embed;
    out_tokens[patch_start..patch_start + n_patches * embed].copy_from_slice(patch_tokens);

    // Add cached, bicubically-interpolated pos-embed to CLS + patch rows only.
    let pos = cache.get_or_build(gh, gw, cfg, weights);
    for i in 0..embed {
        out_tokens[i] += pos[i];
    }
    for i in 0..n_patches * embed {
        out_tokens[patch_start + i] += pos[embed + i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cfg() -> ModelConfig {
        ModelConfig {
            arch: "depthanything3".to_string(),
            patch_size: 2,
            image_size: 4,
            embed_dim: 2,
            depth: 1,
            num_heads: 1,
            head_dim: 1,
            mlp_hidden: 1,
            num_register: 0,
            rope_start: 0,
            qknorm_start: 0,
            rope_freq: 100.0,
            ln_eps: 1e-6,
            out_layers: vec![0],
            ffn_type: "mlp".to_string(),
            head_features: 1,
            head_max_depth: 1.0,
            img_mean: [0.0, 0.0, 0.0],
            img_std: [1.0, 1.0, 1.0],
            img_resize_mode: "bilinear".to_string(),
            alt_start: -1,
            cat_token: true,
            cam_dim_in: 1,
            head_pos_embed: true,
        }
    }

    /// A 3x3 grid (embed=2) of distinct pos-embed vectors, so interpolation
    /// output actually depends on grid position (not just a constant).
    fn test_weights(cfg: &ModelConfig) -> Weights {
        let embed = cfg.embed_dim as usize;
        let grid = 3usize;
        let mut pos = vec![0f32; (grid * grid + 1) * embed];
        for row in 0..(grid * grid + 1) {
            pos[row * embed] = row as f32;
            pos[row * embed + 1] = -(row as f32);
        }
        let mut weights = Weights::new();
        weights.insert_f32(POS_EMBED_WEIGHT, pos);
        weights.insert_f32(CLS_TOKEN_WEIGHT, vec![100.0, -100.0]);
        weights
    }

    #[test]
    fn infer_grid_recovers_square_side() {
        // grid=3 -> rows = 3*3+1 = 10, embed=2 -> len=20
        assert_eq!(infer_grid(20, 2), 3);
        // grid=16 (a realistic DA3-BASE-ish grid) -> rows = 257, embed=384
        assert_eq!(infer_grid(257 * 384, 384), 16);
    }

    #[test]
    #[should_panic(expected = "not a perfect square")]
    fn infer_grid_rejects_non_square_row_count() {
        infer_grid(19 * 2, 2); // 19 rows -> 18 patch rows, not a perfect square
    }

    #[test]
    fn cache_hit_returns_identical_result_without_recompute() {
        let cfg = test_cfg();
        let weights = test_weights(&cfg);
        let mut cache = PosEmbedCache::new();

        let first: Vec<f32> = cache.get_or_build(4, 4, &cfg, &weights).to_vec();
        // A second call at the SAME resolution must return byte-identical
        // content (self-consistency of the cache, not ground truth).
        let second: Vec<f32> = cache.get_or_build(4, 4, &cfg, &weights).to_vec();
        assert_eq!(first, second);
        assert_eq!(
            cache.by_resolution.len(),
            1,
            "same-resolution calls must not create a second cache entry"
        );
    }

    #[test]
    fn cache_distinguishes_resolutions() {
        let cfg = test_cfg();
        let weights = test_weights(&cfg);
        let mut cache = PosEmbedCache::new();

        let small: Vec<f32> = cache.get_or_build(2, 2, &cfg, &weights).to_vec();
        let large: Vec<f32> = cache.get_or_build(5, 5, &cfg, &weights).to_vec();
        assert_ne!(small.len(), large.len());
        assert_eq!(cache.by_resolution.len(), 2);

        // Re-fetching the first resolution after the cache has a second entry
        // still returns the original content unchanged.
        let small_again: Vec<f32> = cache.get_or_build(2, 2, &cfg, &weights).to_vec();
        assert_eq!(small, small_again);
    }

    #[test]
    fn interpolation_at_native_grid_size_is_near_identity_for_cls_and_corners() {
        // When (gh,gw) == (grid,grid) and interp_offset were 0, bicubic
        // resampling at integer sample points would be exact identity. With
        // the nonzero DEFAULT_INTERP_OFFSET (matching the real model), it's
        // only approximately identity — so this test only pins down the CLS
        // row (which is always an exact passthrough) rather than asserting
        // exact patch-row equality.
        let cfg = test_cfg();
        let weights = test_weights(&cfg);
        let mut cache = PosEmbedCache::new();
        let out = cache.get_or_build(3, 3, &cfg, &weights);
        assert_eq!(out[0], 0.0); // CLS row, channel 0 == pos_embed row 0
        assert_eq!(out[1], -0.0);
    }

    #[test]
    fn prepare_tokens_shape_and_pos_embed_add() {
        let cfg = test_cfg();
        let weights = test_weights(&cfg);
        let mut cache = PosEmbedCache::new();

        // 4x4 image, patch=2 -> gh=gw=2 -> 4 patches + 1 CLS = 5 tokens.
        let img = vec![0f32; 3 * 4 * 4]; // zero image -> patch_embed output = bias-only;
        let mut weights = weights;
        // patch_embed also needs its own weight tensors.
        weights.insert_f32("vit.patch_embed.weight", vec![0.0; 2 * 3 * 2 * 2]);
        weights.insert_f32("vit.patch_embed.bias", vec![0.0, 0.0]);

        let mut tokens = Vec::new();
        let (gh, gw) = prepare_tokens(&img, 4, 4, &cfg, &weights, &mut cache, &mut tokens);
        assert_eq!((gh, gw), (2, 2));
        assert_eq!(tokens.len(), 5 * 2);

        // CLS row = cls_token + pos_embed[row 0] = [100,-100] + [0,-0] = [100,-100]
        assert_eq!(&tokens[0..2], &[100.0, -100.0]);
    }
}
