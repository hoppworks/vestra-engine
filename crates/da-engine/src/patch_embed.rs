use crate::ModelConfig;
use da_graph::Weights;
use da_kernels::conv::conv2d;
use da_kernels::gemm::ScalarGemm;

/// Weight-tensor names for the conv-patchify projection.
///
/// These are the *real* GGUF converter names, confirmed from two independent
/// sources in the C++ project (not guessed placeholders):
/// - `../scripts/gguf_keys.py::rename_backbone`: `patch_embed.proj.weight` ->
///   `"vit.patch_embed.weight"`, `patch_embed.proj.bias` -> `"vit.patch_embed.bias"`.
/// - `../src/dino_backbone.cpp` (`prepare_tokens`/`forward`): loads
///   `ml_.tensor("vit.patch_embed.weight")` / `ml_.tensor("vit.patch_embed.bias")`
///   and feeds them straight into `ggml_conv_2d(ctx, pw, img, patch, patch, 0, 0, 1, 1)`.
///
/// So Task 20's real weight-loading must populate `Weights` under exactly these
/// two names for this function (and `pos_embed.rs`'s tensor names below) to work
/// against a real model.
pub const PATCH_EMBED_WEIGHT: &str = "vit.patch_embed.weight";
pub const PATCH_EMBED_BIAS: &str = "vit.patch_embed.bias";

const CHANNELS: usize = 3;

/// Conv-patchify: a non-overlapping `conv2d` (`kernel = stride = patch_size`,
/// `pad = 0`) that turns an NCHW (batch=1) image into a `[n_patches, embed_dim]`
/// token grid.
///
/// Patch order is row-major over the `(gh, gw)` patch grid (`gh = h/patch_size`,
/// `gw = w/patch_size`), i.e. patch index `p = row*gw + col`, matching the C++
/// reference's `ggml_conv_2d` -> `reshape(gw*gh, embed)` -> `transpose` pipeline
/// (`src/dino_backbone.cpp`, `prepare_tokens`/`forward`), which produces tokens
/// in exactly this order before the CLS-token concat.
///
/// `weight` is expected in GGUF/PyTorch `OIHW` layout: `[embed_dim, 3, patch, patch]`
/// (matches `da_kernels::conv::conv2d`'s `weight: out_c*in_c*kh*kw` layout).
///
/// Returns `(gh, gw)`, the patch-grid resolution — the same `(h, w)` pair that
/// should be used as the `PosEmbedCache` key for this call.
pub fn patch_embed(
    img_nchw: &[f32],
    h: usize,
    w: usize,
    cfg: &ModelConfig,
    weights: &Weights,
    out_tokens: &mut Vec<f32>,
) -> (usize, usize) {
    let patch = cfg.patch_size as usize;
    let embed = cfg.embed_dim as usize;
    assert!(patch > 0, "patch_size must be > 0");
    assert_eq!(
        img_nchw.len(),
        CHANNELS * h * w,
        "img_nchw length must be 3*h*w (NCHW, batch=1)"
    );
    assert_eq!(h % patch, 0, "h ({h}) must be a multiple of patch_size ({patch})");
    assert_eq!(w % patch, 0, "w ({w}) must be a multiple of patch_size ({patch})");

    let weight = weights
        .get_f32(PATCH_EMBED_WEIGHT)
        .unwrap_or_else(|| panic!("missing weight tensor {PATCH_EMBED_WEIGHT:?}"));
    let bias = weights
        .get_f32(PATCH_EMBED_BIAS)
        .unwrap_or_else(|| panic!("missing weight tensor {PATCH_EMBED_BIAS:?}"));
    assert_eq!(
        weight.len(),
        embed * CHANNELS * patch * patch,
        "{PATCH_EMBED_WEIGHT} shape mismatch: expected embed_dim*3*patch*patch"
    );
    assert_eq!(bias.len(), embed, "{PATCH_EMBED_BIAS} shape mismatch: expected embed_dim");

    let gh = h / patch;
    let gw = w / patch;
    let n_patches = gh * gw;

    // conv2d writes NCHW-style [embed, gh, gw] (out_c major, then spatial).
    let mut conv_out = vec![0f32; embed * gh * gw];
    conv2d(
        img_nchw,
        CHANNELS,
        h,
        w,
        weight,
        embed,
        patch,
        patch,
        patch, // stride == kernel size: non-overlapping patchify
        0,     // pad
        Some(bias),
        &ScalarGemm,
        &mut conv_out,
    );

    // Transpose [embed, n_patches] -> [n_patches, embed] (token-major, matching
    // the C++ reference's `ggml_transpose` after the conv+reshape).
    out_tokens.clear();
    out_tokens.resize(n_patches * embed, 0.0);
    for p in 0..n_patches {
        for e in 0..embed {
            out_tokens[p * embed + e] = conv_out[e * n_patches + p];
        }
    }

    (gh, gw)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cfg() -> ModelConfig {
        ModelConfig {
            arch: "depthanything3".to_string(),
            patch_size: 2,
            image_size: 4,
            embed_dim: 3,
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
            cam_dim_in: 1,
        }
    }

    #[test]
    fn produces_expected_token_count_and_shape() {
        let cfg = test_cfg();
        let h = 4usize;
        let w = 4usize;
        let img = vec![1.0f32; CHANNELS * h * w];

        let mut weights = Weights::new();
        // embed=3, in_c=3, patch=2 -> weight len = 3*3*2*2 = 36
        weights.insert_f32(PATCH_EMBED_WEIGHT, vec![0.0; 3 * 3 * 2 * 2]);
        weights.insert_f32(PATCH_EMBED_BIAS, vec![1.0, 2.0, 3.0]);

        let mut tokens = Vec::new();
        let (gh, gw) = patch_embed(&img, h, w, &cfg, &weights, &mut tokens);
        assert_eq!((gh, gw), (2, 2));
        assert_eq!(tokens.len(), 4 * 3); // n_patches=4, embed=3

        // Zero weight -> conv output is just the bias, broadcast to every patch.
        for p in 0..4 {
            assert_eq!(&tokens[p * 3..(p + 1) * 3], &[1.0, 2.0, 3.0][..]);
        }
    }

    #[test]
    fn token_order_is_row_major_over_patch_grid() {
        // Use a 1-channel-equivalent (embed=1) identity-ish weight so the output
        // for each patch equals the sum of that patch's input pixels (weight=1)
        // plus bias=0, letting us check *which* patch maps to which output row.
        let mut cfg = test_cfg();
        cfg.embed_dim = 1;
        cfg.patch_size = 2;
        let h = 4usize;
        let w = 4usize;
        // Distinct per-pixel values so each patch sums to a distinct value.
        let mut img = vec![0f32; CHANNELS * h * w];
        for y in 0..h {
            for x in 0..w {
                // Only channel 0 nonzero, value = row-major pixel index.
                img[(0 * h + y) * w + x] = (y * w + x) as f32;
            }
        }

        let mut weights = Weights::new();
        // embed=1, in_c=3, patch=2 -> weight len = 1*3*2*2=12; only channel-0 taps = 1.
        let mut w_data = vec![0f32; 1 * 3 * 2 * 2];
        for i in 0..4 {
            w_data[i] = 1.0; // in_c=0 taps
        }
        weights.insert_f32(PATCH_EMBED_WEIGHT, w_data);
        weights.insert_f32(PATCH_EMBED_BIAS, vec![0.0]);

        let mut tokens = Vec::new();
        let (gh, gw) = patch_embed(&img, h, w, &cfg, &weights, &mut tokens);
        assert_eq!((gh, gw), (2, 2));
        // Patch (0,0): pixels (0,0),(0,1),(1,0),(1,1) = 0,1,4,5 -> sum=10
        // Patch (0,1): pixels (0,2),(0,3),(1,2),(1,3) = 2,3,6,7 -> sum=18
        // Patch (1,0): pixels (2,0),(2,1),(3,0),(3,1) = 8,9,12,13 -> sum=42
        // Patch (1,1): pixels (2,2),(2,3),(3,2),(3,3) = 10,11,14,15 -> sum=50
        assert_eq!(tokens, vec![10.0, 18.0, 42.0, 50.0]);
    }

    #[test]
    #[should_panic(expected = "must be a multiple of patch_size")]
    fn panics_on_non_patch_aligned_size() {
        let cfg = test_cfg();
        let img = vec![0f32; CHANNELS * 3 * 4];
        let weights = Weights::new();
        let mut tokens = Vec::new();
        patch_embed(&img, 3, 4, &cfg, &weights, &mut tokens);
    }
}
