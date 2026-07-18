use crate::ModelConfig;
use da_kernels::bilinear_resize;

/// Resizes a raw HWC `u8` image to `cfg.image_size x cfg.image_size` and
/// normalizes it into CHW (channel-planar) `f32`, ready to feed the ViT.
///
/// Normalization convention: `(pixel/255.0 - mean[c]) / std[c]` per channel,
/// applied *after* scaling to `[0,1]`. This mirrors the C++ reference engine's
/// `preprocess`/`preprocess_real` in `src/preprocess.cpp` (both do
/// `v = rgb[...] / 255.f; chw[...] = (v - mean[c]) / std[c];`, iterating `c`
/// as the outermost loop, i.e. HWC input -> CHW output), so this convention
/// is cross-checked against the existing C++ source, not just assumed from
/// generic ImageNet convention. It has NOT been cross-checked against the
/// `reference.gguf` dump tensors (`raw_image`/`input_image`) because that
/// file does not exist in this environment — see `tests/preprocess_parity.rs`,
/// which will skip until dumps are available.
///
/// Resize: only `da_kernels::resample::bilinear_resize` exists in da-kernels
/// (Task 11), so for `cfg.img_resize_mode == "bilinear"` we use it directly.
/// For any other mode (notably `"bicubic"`, which is what real DA3 GGUF
/// configs use per `crates/da-engine/src/config.rs` test fixtures) we fall
/// back to bilinear as a **documented approximation** — this is a known gap:
/// true bicubic resize (matching cv2 `INTER_CUBIC`/`resize_cubic` in
/// `src/preprocess.cpp`) is not implemented here and would need a dedicated
/// kernel to reach parity for bicubic-configured models. The C++ engine's
/// real preprocessing pipeline (`preprocess_real`) is also more elaborate
/// than a single resize-to-`image_size` step (boundary resize to a target
/// side length, then a second resize to snap to a patch-size multiple); this
/// function instead does the simpler single resize to `image_size x
/// image_size` implied by the Task 15 brief's interface (which only takes a
/// `ModelConfig` with a scalar `image_size`, not the target/patch-aware
/// policy). Reconciling with `preprocess_real`'s two-step policy is out of
/// scope for this task.
///
/// Returns `(H, W)` after resize — currently always
/// `(cfg.image_size as usize, cfg.image_size as usize)`, but computed rather
/// than hardcoded so a future aspect-ratio-preserving resize mode can change
/// it without changing the signature's contract.
pub fn preprocess(
    raw_hwc_u8: &[u8],
    h: usize,
    w: usize,
    cfg: &ModelConfig,
    out_nchw: &mut Vec<f32>,
) -> (usize, usize) {
    const CHANNELS: usize = 3;
    assert_eq!(
        raw_hwc_u8.len(),
        h * w * CHANNELS,
        "raw_hwc_u8 length must be h*w*3 (HWC, u8)"
    );

    let out_h = cfg.image_size as usize;
    let out_w = cfg.image_size as usize;

    // HWC u8 -> CHW f32, scaled to [0,1]. This is the layout bilinear_resize
    // expects (channel-planar).
    let mut in_chw = vec![0f32; CHANNELS * h * w];
    for y in 0..h {
        for x in 0..w {
            let px = (y * w + x) * CHANNELS;
            for c in 0..CHANNELS {
                in_chw[(c * h + y) * w + x] = raw_hwc_u8[px + c] as f32 / 255.0;
            }
        }
    }

    // Resize (channel-planar in, channel-planar out).
    let mut resized_chw = vec![0f32; CHANNELS * out_h * out_w];
    if cfg.img_resize_mode == "bilinear" {
        bilinear_resize(&in_chw, CHANNELS, h, w, out_h, out_w, &mut resized_chw);
    } else {
        // Documented approximation for "bicubic" (and any other unrecognized
        // mode): use bilinear. See module-level doc comment for the gap this
        // leaves for bicubic-configured models.
        bilinear_resize(&in_chw, CHANNELS, h, w, out_h, out_w, &mut resized_chw);
    }

    // Normalize per channel: (v - mean[c]) / std[c]. Already CHW, so this is
    // just a per-plane affine transform; also serves as our NCHW output
    // (batch=1, so NCHW == CHW).
    out_nchw.clear();
    out_nchw.resize(CHANNELS * out_h * out_w, 0.0);
    for c in 0..CHANNELS {
        let mean = cfg.img_mean[c];
        let std = cfg.img_std[c];
        let plane_len = out_h * out_w;
        let src = &resized_chw[c * plane_len..(c + 1) * plane_len];
        let dst = &mut out_nchw[c * plane_len..(c + 1) * plane_len];
        for i in 0..plane_len {
            dst[i] = (src[i] - mean) / std;
        }
    }

    (out_h, out_w)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cfg(image_size: u32, resize_mode: &str) -> ModelConfig {
        ModelConfig {
            arch: "depthanything3".to_string(),
            patch_size: 14,
            image_size,
            embed_dim: 384,
            depth: 12,
            num_heads: 6,
            head_dim: 64,
            mlp_hidden: 1536,
            num_register: 4,
            rope_start: 0,
            qknorm_start: 0,
            rope_freq: 100.0,
            ln_eps: 1e-6,
            out_layers: vec![2, 5, 8, 11],
            head_features: 256,
            head_max_depth: 20.0,
            img_mean: [0.485, 0.456, 0.406],
            img_std: [0.229, 0.224, 0.225],
            img_resize_mode: resize_mode.to_string(),
            cam_dim_in: 8,
        }
    }

    #[test]
    fn returns_target_size() {
        let cfg = test_cfg(4, "bilinear");
        let raw = vec![128u8; 2 * 2 * 3];
        let mut out = Vec::new();
        let (oh, ow) = preprocess(&raw, 2, 2, &cfg, &mut out);
        assert_eq!((oh, ow), (4, 4));
        assert_eq!(out.len(), 3 * 4 * 4);
    }

    #[test]
    fn identity_resize_matches_hand_computed_normalization() {
        // No resize (input already at target size) isolates the
        // normalization math from the resize kernel.
        let cfg = test_cfg(2, "bilinear");
        // 2x2 HWC image, channel values chosen so the math is easy to verify.
        // Pixel (0,0): R=255,G=0,B=0 ; Pixel (0,1): R=0,G=255,B=0
        // Pixel (1,0): R=0,G=0,B=255 ; Pixel (1,1): R=128,G=128,B=128
        let raw: Vec<u8> = vec![
            255, 0, 0, 0, 255, 0, //
            0, 0, 255, 128, 128, 128, //
        ];
        let mut out = Vec::new();
        let (oh, ow) = preprocess(&raw, 2, 2, &cfg, &mut out);
        assert_eq!((oh, ow), (2, 2));

        let expect = |v: u8, c: usize| -> f32 {
            (v as f32 / 255.0 - cfg.img_mean[c]) / cfg.img_std[c]
        };
        // NCHW layout: channel-major, then row-major within each plane.
        // R plane: [255, 0, 0, 128]
        assert!((out[0] - expect(255, 0)).abs() < 1e-6);
        assert!((out[1] - expect(0, 0)).abs() < 1e-6);
        assert!((out[2] - expect(0, 0)).abs() < 1e-6);
        assert!((out[3] - expect(128, 0)).abs() < 1e-6);
        // G plane: [0, 255, 0, 128]
        assert!((out[4] - expect(0, 1)).abs() < 1e-6);
        assert!((out[5] - expect(255, 1)).abs() < 1e-6);
        assert!((out[6] - expect(0, 1)).abs() < 1e-6);
        assert!((out[7] - expect(128, 1)).abs() < 1e-6);
        // B plane: [0, 0, 255, 128]
        assert!((out[8] - expect(0, 2)).abs() < 1e-6);
        assert!((out[9] - expect(0, 2)).abs() < 1e-6);
        assert!((out[10] - expect(255, 2)).abs() < 1e-6);
        assert!((out[11] - expect(128, 2)).abs() < 1e-6);
    }

    #[test]
    fn unrecognized_resize_mode_falls_back_to_bilinear() {
        let cfg = test_cfg(4, "bicubic");
        let raw = vec![200u8; 3 * 3 * 3];
        let mut out = Vec::new();
        let (oh, ow) = preprocess(&raw, 3, 3, &cfg, &mut out);
        assert_eq!((oh, ow), (4, 4));
        assert_eq!(out.len(), 3 * 4 * 4);
    }
}
