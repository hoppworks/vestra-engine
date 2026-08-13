use crate::ModelConfig;
use da_kernels::bilinear_resize;

/// The exact pixel-coordinate mapping produced by [`preprocess_letterbox`].
///
/// Coordinates use the usual calibrated-camera convention: integer values
/// denote **pixel centres**.  This matters for resizing: with the
/// half-pixel-centre interpolation convention used by [`bilinear_resize`], a
/// source coordinate `x` maps to
/// `pad_left + (x + 0.5) * scale_x - 0.5`, rather than simply
/// `pad_left + x * scale_x`.  Keeping that half-pixel term explicit makes
/// this transform suitable for moving camera intrinsics and reconstructed
/// depth pixels between the source video and model image without a hidden
/// half-pixel drift.
///
/// The resize dimensions are rounded to integral raster dimensions.  For
/// source aspect ratios that cannot be represented exactly at the model
/// resolution, `scale_x` and `scale_y` therefore differ by at most that
/// unavoidable rounding error.  They describe the actual resampling, not an
/// idealised scale, and are the values that geometry consumers must use.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LetterboxTransform {
    /// Source video frame dimensions, in pixels.
    pub source_width: usize,
    pub source_height: usize,
    /// Model canvas dimensions, in pixels.
    pub model_width: usize,
    pub model_height: usize,
    /// Dimensions of the resized source image placed on the model canvas.
    pub resized_width: usize,
    pub resized_height: usize,
    /// Integral black-padding amounts on each canvas edge.
    pub pad_left: usize,
    pub pad_top: usize,
    pub pad_right: usize,
    pub pad_bottom: usize,
    /// Actual source-to-resized-raster scale factors.
    pub scale_x: f32,
    pub scale_y: f32,
}

impl LetterboxTransform {
    /// Converts a source-frame pixel-centre coordinate into a model-canvas
    /// pixel-centre coordinate.
    #[must_use]
    pub fn source_to_model_pixel(&self, x: f32, y: f32) -> (f32, f32) {
        (
            self.pad_left as f32 + (x + 0.5) * self.scale_x - 0.5,
            self.pad_top as f32 + (y + 0.5) * self.scale_y - 0.5,
        )
    }

    /// Converts a model-canvas pixel-centre coordinate back into the source
    /// frame. This is algebraically the inverse of
    /// [`Self::source_to_model_pixel`], including for coordinates in padding;
    /// use [`Self::model_pixel_is_content`] to reject padded pixels first.
    #[must_use]
    pub fn model_to_source_pixel(&self, x: f32, y: f32) -> (f32, f32) {
        (
            ((x - self.pad_left as f32) + 0.5) / self.scale_x - 0.5,
            ((y - self.pad_top as f32) + 0.5) / self.scale_y - 0.5,
        )
    }

    /// Returns whether a model-canvas coordinate lies inside the resized
    /// source rectangle rather than letterbox padding. The rectangle uses
    /// pixel-area bounds, so an integer model pixel coordinate can be passed
    /// directly.
    #[must_use]
    pub fn model_pixel_is_content(&self, x: f32, y: f32) -> bool {
        let left = self.pad_left as f32;
        let top = self.pad_top as f32;
        x >= left - 0.5
            && x < left + self.resized_width as f32 - 0.5
            && y >= top - 0.5
            && y < top + self.resized_height as f32 - 0.5
    }

    /// Moves a row-major `3x3` source camera-intrinsics matrix onto the
    /// letterboxed model canvas. This applies the same pixel-centre affine
    /// transform as [`Self::source_to_model_pixel`], including non-uniform
    /// scale caused by integer resize rounding.
    #[must_use]
    pub fn source_to_model_intrinsics(&self, source: [f32; 9]) -> [f32; 9] {
        let offset_x = self.pad_left as f32 + 0.5 * self.scale_x - 0.5;
        let offset_y = self.pad_top as f32 + 0.5 * self.scale_y - 0.5;
        [
            self.scale_x * source[0] + offset_x * source[6],
            self.scale_x * source[1] + offset_x * source[7],
            self.scale_x * source[2] + offset_x * source[8],
            self.scale_y * source[3] + offset_y * source[6],
            self.scale_y * source[4] + offset_y * source[7],
            self.scale_y * source[5] + offset_y * source[8],
            source[6],
            source[7],
            source[8],
        ]
    }

    /// Moves a row-major `3x3` camera-intrinsics matrix from the letterboxed
    /// model canvas back to the source video frame.
    #[must_use]
    pub fn model_to_source_intrinsics(&self, model: [f32; 9]) -> [f32; 9] {
        let offset_x = self.pad_left as f32 + 0.5 * self.scale_x - 0.5;
        let offset_y = self.pad_top as f32 + 0.5 * self.scale_y - 0.5;
        [
            (model[0] - offset_x * model[6]) / self.scale_x,
            (model[1] - offset_x * model[7]) / self.scale_x,
            (model[2] - offset_x * model[8]) / self.scale_x,
            (model[3] - offset_y * model[6]) / self.scale_y,
            (model[4] - offset_y * model[7]) / self.scale_y,
            (model[5] - offset_y * model[8]) / self.scale_y,
            model[6],
            model[7],
            model[8],
        ]
    }
}

/// Aspect-ratio-preserving letterbox preprocessing for calibrated video
/// frames.
///
/// The source is resized into `cfg.image_size x cfg.image_size` without
/// cropping, centered in black padding, then normalized into CHW `f32` like
/// [`preprocess`]. The returned transform is the authoritative reversible
/// source-frame <-> model-canvas pixel mapping. It deliberately does not
/// replace [`preprocess`], whose stretch-to-square behavior remains the
/// compatibility path for existing inference callers.
///
/// # Panics
/// Panics when the image dimensions are zero, `cfg.image_size` is zero, or
/// the HWC input byte length is not exactly `h * w * 3`.
pub fn preprocess_letterbox(
    raw_hwc_u8: &[u8],
    h: usize,
    w: usize,
    cfg: &ModelConfig,
    out_nchw: &mut Vec<f32>,
) -> LetterboxTransform {
    const CHANNELS: usize = 3;
    assert!(
        h > 0 && w > 0,
        "letterbox source dimensions must be non-zero"
    );
    assert!(
        cfg.image_size > 0,
        "letterbox model image_size must be non-zero"
    );
    assert_eq!(
        raw_hwc_u8.len(),
        h * w * CHANNELS,
        "raw_hwc_u8 length must be h*w*3 (HWC, u8)"
    );

    let model_h = cfg.image_size as usize;
    let model_w = cfg.image_size as usize;
    let ideal_scale = (model_w as f64 / w as f64).min(model_h as f64 / h as f64);
    let resized_w = ((w as f64 * ideal_scale).round() as usize).clamp(1, model_w);
    let resized_h = ((h as f64 * ideal_scale).round() as usize).clamp(1, model_h);
    let pad_left = (model_w - resized_w) / 2;
    let pad_top = (model_h - resized_h) / 2;

    let transform = LetterboxTransform {
        source_width: w,
        source_height: h,
        model_width: model_w,
        model_height: model_h,
        resized_width: resized_w,
        resized_height: resized_h,
        pad_left,
        pad_top,
        pad_right: model_w - resized_w - pad_left,
        pad_bottom: model_h - resized_h - pad_top,
        scale_x: resized_w as f32 / w as f32,
        scale_y: resized_h as f32 / h as f32,
    };

    // HWC u8 -> CHW f32 in [0,1], matching the existing preprocess path.
    let mut in_chw = vec![0f32; CHANNELS * h * w];
    for y in 0..h {
        for x in 0..w {
            let px = (y * w + x) * CHANNELS;
            for c in 0..CHANNELS {
                in_chw[(c * h + y) * w + x] = raw_hwc_u8[px + c] as f32 / 255.0;
            }
        }
    }

    let mut resized_chw = vec![0f32; CHANNELS * resized_h * resized_w];
    bilinear_resize(
        &in_chw,
        CHANNELS,
        h,
        w,
        resized_h,
        resized_w,
        &mut resized_chw,
    );

    // Build the black letterbox canvas before normalization. Consequently a
    // black pad is represented by (0 - mean[c]) / std[c], exactly as if it
    // were an ordinary black source pixel.
    out_nchw.clear();
    out_nchw.resize(CHANNELS * model_h * model_w, 0.0);
    for c in 0..CHANNELS {
        let mean = cfg.img_mean[c];
        let std = cfg.img_std[c];
        for y in 0..model_h {
            for x in 0..model_w {
                let value = if transform.model_pixel_is_content(x as f32, y as f32) {
                    let src_x = x - pad_left;
                    let src_y = y - pad_top;
                    resized_chw[(c * resized_h + src_y) * resized_w + src_x]
                } else {
                    0.0
                };
                out_nchw[(c * model_h + y) * model_w + x] = (value - mean) / std;
            }
        }
    }

    transform
}

#[inline]
fn saturating_round_u8(value: f32) -> u8 {
    value.round_ties_even().clamp(0.0, 255.0) as u8
}

#[inline]
fn cubic_weight(mut x: f32) -> f32 {
    const A: f32 = -0.75;
    x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

fn resize_bilinear_u8(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    let mut dst = vec![0; dw * dh * 3];
    let sx = sw as f32 / dw as f32;
    let sy = sh as f32 / dh as f32;
    for y in 0..dh {
        let fy = (y as f32 + 0.5) * sy - 0.5;
        let y0 = fy.floor() as isize;
        let wy = fy - y0 as f32;
        let y0c = y0.clamp(0, sh as isize - 1) as usize;
        let y1c = (y0 + 1).clamp(0, sh as isize - 1) as usize;
        for x in 0..dw {
            let fx = (x as f32 + 0.5) * sx - 0.5;
            let x0 = fx.floor() as isize;
            let wx = fx - x0 as f32;
            let x0c = x0.clamp(0, sw as isize - 1) as usize;
            let x1c = (x0 + 1).clamp(0, sw as isize - 1) as usize;
            for c in 0..3 {
                let top = src[(y0c * sw + x0c) * 3 + c] as f32 * (1.0 - wx)
                    + src[(y0c * sw + x1c) * 3 + c] as f32 * wx;
                let bottom = src[(y1c * sw + x0c) * 3 + c] as f32 * (1.0 - wx)
                    + src[(y1c * sw + x1c) * 3 + c] as f32 * wx;
                dst[(y * dw + x) * 3 + c] = saturating_round_u8(top * (1.0 - wy) + bottom * wy);
            }
        }
    }
    dst
}

fn resize_cubic_u8(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    let sx = sw as f64 / dw as f64;
    let sy = sh as f64 / dh as f64;
    let mut x_indices = vec![[0usize; 4]; dw];
    let mut x_weights = vec![[0.0f32; 4]; dw];
    for x in 0..dw {
        let fx = (x as f64 + 0.5) * sx - 0.5;
        let ix = fx.floor() as isize;
        let t = (fx - ix as f64) as f32;
        let weights = [
            cubic_weight(t + 1.0),
            cubic_weight(t),
            cubic_weight(t - 1.0),
            cubic_weight(t - 2.0),
        ];
        for k in 0..4 {
            x_indices[x][k] = (ix - 1 + k as isize).clamp(0, sw as isize - 1) as usize;
            x_weights[x][k] = weights[k];
        }
    }

    let mut tmp = vec![0.0f32; sh * dw * 3];
    for y in 0..sh {
        for x in 0..dw {
            for c in 0..3 {
                let mut sum = 0.0;
                for k in 0..4 {
                    sum += x_weights[x][k] * src[(y * sw + x_indices[x][k]) * 3 + c] as f32;
                }
                tmp[(y * dw + x) * 3 + c] = sum;
            }
        }
    }

    let mut dst = vec![0u8; dh * dw * 3];
    for y in 0..dh {
        let fy = (y as f64 + 0.5) * sy - 0.5;
        let iy = fy.floor() as isize;
        let t = (fy - iy as f64) as f32;
        let weights = [
            cubic_weight(t + 1.0),
            cubic_weight(t),
            cubic_weight(t - 1.0),
            cubic_weight(t - 2.0),
        ];
        let indices = [
            (iy - 1).clamp(0, sh as isize - 1) as usize,
            iy.clamp(0, sh as isize - 1) as usize,
            (iy + 1).clamp(0, sh as isize - 1) as usize,
            (iy + 2).clamp(0, sh as isize - 1) as usize,
        ];
        for x in 0..dw {
            for c in 0..3 {
                let mut sum = 0.0;
                for k in 0..4 {
                    sum += weights[k] * tmp[(indices[k] * dw + x) * 3 + c];
                }
                dst[(y * dw + x) * 3 + c] = saturating_round_u8(sum);
            }
        }
    }
    dst
}

#[derive(Clone, Copy)]
struct AreaTap {
    destination: usize,
    source: usize,
    weight: f32,
}

fn area_table(source_size: usize, destination_size: usize) -> Vec<AreaTap> {
    let scale = source_size as f64 / destination_size as f64;
    let mut taps = Vec::new();
    for destination in 0..destination_size {
        let source_start_f = destination as f64 * scale;
        let source_end_f = source_start_f + scale;
        let cell_width = scale.min(source_size as f64 - source_start_f);
        let mut source_start = source_start_f.ceil() as usize;
        let mut source_end = source_end_f.floor() as usize;
        source_end = source_end.min(source_size - 1);
        source_start = source_start.min(source_end);
        if source_start as f64 - source_start_f > 1e-3 {
            taps.push(AreaTap {
                destination,
                source: source_start - 1,
                weight: ((source_start as f64 - source_start_f) / cell_width) as f32,
            });
        }
        for source in source_start..source_end {
            taps.push(AreaTap {
                destination,
                source,
                weight: (1.0 / cell_width) as f32,
            });
        }
        if source_end_f - source_end as f64 > 1e-3 {
            taps.push(AreaTap {
                destination,
                source: source_end,
                weight: ((source_end_f - source_end as f64).min(1.0).min(cell_width) / cell_width)
                    as f32,
            });
        }
    }
    taps
}

fn resize_area_u8(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    if dw >= sw && dh >= sh {
        return resize_bilinear_u8(src, sw, sh, dw, dh);
    }
    let x_taps = area_table(sw, dw);
    let y_taps = area_table(sh, dh);
    let mut tmp = vec![0.0f32; sh * dw * 3];
    for y in 0..sh {
        for tap in &x_taps {
            for c in 0..3 {
                tmp[(y * dw + tap.destination) * 3 + c] +=
                    tap.weight * src[(y * sw + tap.source) * 3 + c] as f32;
            }
        }
    }
    let mut accumulated = vec![0.0f32; dh * dw * 3];
    for tap in &y_taps {
        for x in 0..dw {
            for c in 0..3 {
                accumulated[(tap.destination * dw + x) * 3 + c] +=
                    tap.weight * tmp[(tap.source * dw + x) * 3 + c];
            }
        }
    }
    accumulated.into_iter().map(saturating_round_u8).collect()
}

#[inline]
fn nearest_multiple(value: usize, multiple: usize) -> usize {
    let down = value / multiple * multiple;
    let up = down + multiple;
    if up.abs_diff(value) <= value.abs_diff(down) {
        up
    } else {
        down
    }
}

/// Applies the production C++ `preprocess_real` contract: preserve aspect
/// ratio, resize the configured boundary to `image_size`, snap both axes to
/// the nearest patch multiple, round back to RGB `u8`, then normalize to CHW.
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
/// Returns the patch-aligned `(H, W)` actually fed to the model.
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

    let target = cfg.image_size.max(1) as usize;
    let patch = cfg.patch_size.max(1) as usize;
    let upper_bound = !cfg.img_resize_mode.starts_with("lower");
    let bound = if upper_bound { w.max(h) } else { w.min(h) };
    let scale = target as f64 / bound as f64;
    let boundary_w = ((w as f64 * scale).round_ties_even() as usize).max(1);
    let boundary_h = ((h as f64 * scale).round_ties_even() as usize).max(1);
    let mut resized = if (scale - 1.0).abs() < f64::EPSILON {
        raw_hwc_u8.to_vec()
    } else if scale > 1.0 {
        resize_cubic_u8(raw_hwc_u8, w, h, boundary_w, boundary_h)
    } else {
        resize_area_u8(raw_hwc_u8, w, h, boundary_w, boundary_h)
    };

    let out_w = nearest_multiple(boundary_w, patch).max(1);
    let out_h = nearest_multiple(boundary_h, patch).max(1);
    if out_w != boundary_w || out_h != boundary_h {
        let upscale = out_w > boundary_w || out_h > boundary_h;
        resized = if upscale {
            resize_cubic_u8(&resized, boundary_w, boundary_h, out_w, out_h)
        } else {
            resize_area_u8(&resized, boundary_w, boundary_h, out_w, out_h)
        };
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
        let dst = &mut out_nchw[c * plane_len..(c + 1) * plane_len];
        for y in 0..out_h {
            for x in 0..out_w {
                let value = resized[(y * out_w + x) * CHANNELS + c] as f32 / 255.0;
                dst[y * out_w + x] = (value - mean) / std;
            }
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
            patch_size: 2,
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
            ffn_type: "mlp".to_string(),
            head_features: 256,
            head_max_depth: 20.0,
            img_mean: [0.485, 0.456, 0.406],
            img_std: [0.229, 0.224, 0.225],
            img_resize_mode: resize_mode.to_string(),
            alt_start: -1,
            cat_token: true,
            cam_dim_in: 8,
            head_pos_embed: true,
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

        let expect =
            |v: u8, c: usize| -> f32 { (v as f32 / 255.0 - cfg.img_mean[c]) / cfg.img_std[c] };
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

    #[test]
    fn production_upper_bound_preserves_aspect_and_snaps_to_patch() {
        let mut cfg = test_cfg(504, "upper_bound");
        cfg.patch_size = 14;
        let raw = vec![128u8; 427 * 640 * 3];
        let mut out = Vec::new();
        let (oh, ow) = preprocess(&raw, 427, 640, &cfg, &mut out);
        assert_eq!((oh, ow), (336, 504));
        assert_eq!(out.len(), 3 * 336 * 504);
    }

    #[test]
    fn letterbox_keeps_content_aspect_and_normalizes_padding_as_black() {
        let mut cfg = test_cfg(8, "bilinear");
        cfg.img_mean = [0.0, 0.0, 0.0];
        cfg.img_std = [1.0, 1.0, 1.0];
        // 2x4 image: resize to 4x8 then place two black rows above and below.
        let raw = vec![255u8; 2 * 4 * 3];
        let mut out = Vec::new();
        let transform = preprocess_letterbox(&raw, 2, 4, &cfg, &mut out);

        assert_eq!(transform.resized_width, 8);
        assert_eq!(transform.resized_height, 4);
        assert_eq!((transform.pad_left, transform.pad_right), (0, 0));
        assert_eq!((transform.pad_top, transform.pad_bottom), (2, 2));
        assert_eq!(out.len(), 3 * 8 * 8);

        // R channel; the other channels use the same fixture values.
        assert_eq!(out[0 * 8 + 3], 0.0, "top padding must be black");
        assert_eq!(out[2 * 8 + 3], 1.0, "first content row must preserve white");
        assert_eq!(out[6 * 8 + 3], 0.0, "bottom padding must be black");
    }

    #[test]
    fn letterbox_pixel_transform_round_trips_with_half_pixel_centres() {
        let cfg = test_cfg(8, "bilinear");
        let raw = vec![0u8; 3 * 5 * 3];
        let mut out = Vec::new();
        let transform = preprocess_letterbox(&raw, 3, 5, &cfg, &mut out);

        // 5x3 -> 8x5. The rounded height makes scale_y differ from scale_x;
        // the transform must still describe the actual raster exactly.
        assert_eq!((transform.resized_width, transform.resized_height), (8, 5));
        assert!((transform.scale_x - 1.6).abs() < 1e-6);
        assert!((transform.scale_y - 5.0 / 3.0).abs() < 1e-6);
        assert_eq!((transform.pad_top, transform.pad_bottom), (1, 2));

        let source = (3.25, 1.5);
        let model = transform.source_to_model_pixel(source.0, source.1);
        let restored = transform.model_to_source_pixel(model.0, model.1);
        assert!((restored.0 - source.0).abs() < 1e-6);
        assert!((restored.1 - source.1).abs() < 1e-6);

        // The top padding is invalid geometry, but inversion remains defined
        // so callers can inspect/debug it without a special code path.
        assert!(!transform.model_pixel_is_content(4.0, 0.0));
        assert!(transform.model_pixel_is_content(4.0, 1.0));

        // During downsampling the first source centre projects between the
        // canvas's -0.5 and 0.0 pixel-centre coordinates. It is still valid
        // content geometry, not padding.
        let downscaled = preprocess_letterbox(&vec![0u8; 8 * 16 * 3], 8, 16, &cfg, &mut out);
        let origin = downscaled.source_to_model_pixel(0.0, 0.0);
        assert!(downscaled.model_pixel_is_content(origin.0, origin.1));
    }

    #[test]
    fn letterbox_moves_calibrated_intrinsics_and_restores_them() {
        let cfg = test_cfg(8, "bilinear");
        let raw = vec![0u8; 2 * 4 * 3];
        let mut out = Vec::new();
        let transform = preprocess_letterbox(&raw, 2, 4, &cfg, &mut out);
        let source = [100.0, 0.5, 2.0, 0.0, 120.0, 1.0, 0.0, 0.0, 1.0];

        let model = transform.source_to_model_intrinsics(source);
        // 2x uniform resize and two rows of top padding. With pixel centres,
        // cx' = 2*cx + 0.5 = 4.5 and cy' = 2*cy + 2.5 = 4.5.
        assert!((model[0] - 200.0).abs() < 1e-6);
        assert!((model[1] - 1.0).abs() < 1e-6);
        assert!((model[2] - 4.5).abs() < 1e-6);
        assert!((model[4] - 240.0).abs() < 1e-6);
        assert!((model[5] - 4.5).abs() < 1e-6);

        let restored = transform.model_to_source_intrinsics(model);
        for (actual, expected) in restored.into_iter().zip(source) {
            assert!(
                (actual - expected).abs() < 1e-5,
                "got {actual}, want {expected}"
            );
        }
    }
}
