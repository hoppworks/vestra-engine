//! UV (2D sinusoidal) positional embedding used by the DPT head — added at
//! every `projects[s]` stage and once more after the final upsample (Task
//! 18's `dpt_head`), gated by `cfg.head_pos_embed`.
//!
//! Ported byte-for-byte from the real C++ reference's `uv_pos_embed`
//! (`../src/uv_posembed.cpp`) and its caller `add_uv_input`
//! (`../src/dpt_head.cpp`), which additionally:
//! 1. Reorders `uv_pos_embed`'s raw `[ph, pw, c]` (`(y*pw+x)*c+ch`) output
//!    into `[c, ph, pw]` CHW (`ch*ph*pw + y*pw + x`), matching the feature
//!    maps this gets added onto.
//! 2. Scales every value by a fixed `ratio = 0.1` **baked into the cached
//!    buffer itself** (not applied again at each add site) — see
//!    `uv_pos_embed_c_h_w`'s doc comment.
//! 3. Caches the finished buffer per `(w, h, c)` (the embedding is
//!    input-independent — depends only on feature-map geometry — so
//!    recomputing it on every forward pass at a fixed resolution is pure
//!    waste; this mirrors `../src/dino_backbone.cpp`'s/`pos_embed.rs`'s own
//!    `PosEmbedCache` pattern).
//!
//! NOT numerically cross-checked against the `uv_embed_64` reference dump in
//! this environment (no `../dumps/reference.gguf` present) — the formula is
//! transcribed directly from the C++ source (see module + function doc
//! comments for the line-by-line correspondence), not reverse-engineered
//! from expected output, so confidence is high, but it is UNVERIFIED
//! end-to-end.

use std::collections::HashMap;

/// The `omega0` base used by `uv_pos_embed`'s per-frequency exponential
/// decay. Matches the C++ reference's default argument
/// (`uv_pos_embed(int pw, int ph, int C, float aspect, float omega0=100.0f)`,
/// `../src/uv_posembed.hpp`) — DA3-BASE never overrides it.
const DEFAULT_OMEGA0: f32 = 100.0;

/// The fixed ratio scale baked into the cached UV embedding buffer by the
/// C++ reference's `add_uv_input` (`../src/dpt_head.cpp`: `ratio = 0.1f`,
/// applied while filling the `[C,H,W]` cache buffer, not re-applied at each
/// `ggml_add` call site). `uv_embed` below applies this same `*0.1` before
/// returning/caching, matching that "baked into the cached buffer itself"
/// pattern described in the task brief.
const RATIO_SCALE: f32 = 0.1;

/// Raw (un-scaled) 2D sinusoidal positional embedding at patch-grid
/// resolution `(pw, ph)` with `c` channels (must be a multiple of 4: `D =
/// c/2` channels per axis, `F = D/2` frequencies per axis/sin-cos pair).
///
/// Output layout: `[ph, pw, c]` row-major, i.e. flat index
/// `(y*pw+x)*c+channel` — the same layout `uv_embed` below permutes into
/// `[c, ph, pw]` CHW before caching.
///
/// Ported line-for-line from `../src/uv_posembed.cpp::uv_pos_embed`
/// (see that file for the `create_uv_grid`-derived span/linspace math and
/// the emb_x/emb_y sin/cos channel-block layout).
pub fn uv_pos_embed(pw: usize, ph: usize, c: usize, aspect: f32, omega0: f32) -> Vec<f32> {
    let mut out = vec![0f32; ph * pw * c];
    if pw == 0 || ph == 0 || c == 0 {
        return out;
    }
    assert_eq!(c % 4, 0, "uv_pos_embed: c={c} must be a multiple of 4 (D=c/2, F=D/2)");

    let diag = ((aspect * aspect + 1.0) as f64).sqrt();
    let span_x = aspect as f64 / diag;
    let span_y = 1.0 / diag;
    let left_x = -span_x * (pw - 1) as f64 / pw as f64;
    let right_x = span_x * (pw - 1) as f64 / pw as f64;
    let top_y = -span_y * (ph - 1) as f64 / ph as f64;
    let bottom_y = span_y * (ph - 1) as f64 / ph as f64;

    let x_coords: Vec<f64> = if pw == 1 {
        vec![left_x]
    } else {
        let step = (right_x - left_x) / (pw - 1) as f64;
        (0..pw).map(|i| left_x + i as f64 * step).collect()
    };
    let y_coords: Vec<f64> = if ph == 1 {
        vec![top_y]
    } else {
        let step = (bottom_y - top_y) / (ph - 1) as f64;
        (0..ph).map(|i| top_y + i as f64 * step).collect()
    };

    let d = c / 2;
    let f = d / 2;
    let omega0 = omega0 as f64;
    let omega: Vec<f64> = (0..f).map(|j| 1.0 / omega0.powf(j as f64 / f as f64)).collect();

    for y in 0..ph {
        for x in 0..pw {
            let xc = x_coords[x];
            let yc = y_coords[y];
            let base = (y * pw + x) * c;
            for j in 0..f {
                let o = xc * omega[j];
                out[base + j] = o.sin() as f32;
                out[base + f + j] = o.cos() as f32;
            }
            for j in 0..f {
                let o = yc * omega[j];
                out[base + d + j] = o.sin() as f32;
                out[base + d + f + j] = o.cos() as f32;
            }
        }
    }
    out
}

/// Builds the finished, `*0.1`-scaled, `[c, h, w]` CHW UV positional
/// embedding at resolution `(h, w)` with `dim` channels and an explicit
/// `aspect`, ready to be added directly onto a `[c, h, w]` feature map.
/// `omega0 = 100.0`, matching `add_uv_input`'s call to `uv_pos_embed` in the
/// C++ reference.
///
/// **`aspect` is a separate parameter from `(h, w)`, not derived from them**
/// — this matters: the C++ reference's `DptHead::build_depth_graph` always
/// uses the *full preprocessed image's* pixel aspect ratio (`(float)W /
/// (float)H`, the target depth-map resolution), even when building the UV
/// embedding at a much smaller *patch-grid* resolution `(pw, ph)` for the
/// per-stage `projects[s]` additions (`../src/dpt_head.cpp`:
/// `add_uv_input(ctx, be_, pool, pw, ph, oc[s], aspect, ratio)` where
/// `aspect` was computed once from `W,H` at the top of the function, not
/// from `pw,ph`). Deriving `aspect` from whatever `(h, w)` this function
/// happens to be called with — e.g. `aspect = w/h` at a square 16x16 patch
/// grid, always `1.0` regardless of the real image's aspect ratio — would
/// silently produce the wrong UV embedding for any non-square image.
fn build_uv_embed_chw(h: usize, w: usize, dim: usize, aspect: f32) -> Vec<f32> {
    let raw = uv_pos_embed(w, h, dim, aspect, DEFAULT_OMEGA0); // [h, w, dim] i.e. (y*w+x)*dim+ch
    let mut out = vec![0f32; dim * h * w];
    for c in 0..dim {
        for y in 0..h {
            for x in 0..w {
                out[c * h * w + y * w + x] = RATIO_SCALE * raw[(y * w + x) * dim + c];
            }
        }
    }
    out
}

/// Caches finished (`*0.1`-scaled, `[c,h,w]` CHW) UV positional embeddings
/// per `(w, h, dim, aspect)` — the embedding is input-INDEPENDENT (depends
/// only on feature-map geometry + the target aspect ratio), so rebuilding it
/// on every `dpt_head` call at a fixed resolution is pure waste. Mirrors
/// `pos_embed.rs`'s `PosEmbedCache` pattern and the C++ reference's own
/// `add_uv_input` static cache (`../src/dpt_head.cpp`, keyed on
/// `(W,H,C,aspect_bits,ratio_bits)`).
#[derive(Default)]
pub struct UvEmbedCache {
    by_key: HashMap<(usize, usize, usize, u32), Vec<f32>>,
}

impl UvEmbedCache {
    pub fn new() -> Self {
        UvEmbedCache { by_key: HashMap::new() }
    }

    /// Returns the cached (or freshly-built-and-cached) `[dim, h, w]` CHW UV
    /// positional embedding for this `(h, w, dim)`, with `aspect` derived
    /// as `w/h` (correct when `(h, w)` IS the real target aspect ratio,
    /// e.g. the final full-resolution UV add after upsampling — see
    /// [`Self::get_or_build_with_aspect`] for the per-stage case where the
    /// UV field's own `(h, w)` and the target aspect ratio differ).
    pub fn get_or_build(&mut self, h: usize, w: usize, dim: usize) -> &[f32] {
        self.get_or_build_with_aspect(h, w, dim, w as f32 / h as f32)
    }

    /// Like [`Self::get_or_build`], but with an explicit `aspect` decoupled
    /// from `(h, w)` — needed by `dpt_head`'s per-stage `projects[s]`
    /// additions, which build the UV field at patch-grid resolution `(pw,
    /// ph)` but must still use the full target image's pixel aspect ratio
    /// (see [`build_uv_embed_chw`]'s doc comment for why this distinction
    /// matters).
    pub fn get_or_build_with_aspect(&mut self, h: usize, w: usize, dim: usize, aspect: f32) -> &[f32] {
        let key = (h, w, dim, aspect.to_bits());
        self.by_key.entry(key).or_insert_with(|| build_uv_embed_chw(h, w, dim, aspect)).as_slice()
    }
}

/// Convenience one-shot wrapper (no caching) matching the task interface's
/// `uv_embed(h, w, dim, out)` signature: computes the `*0.1`-scaled `[dim,
/// h, w]` CHW UV positional embedding directly into `out`. Prefer
/// [`UvEmbedCache::get_or_build`] on any hot path that calls this at a fixed
/// resolution repeatedly (e.g. `dpt_head` across multiple forward passes) —
/// this function recomputes from scratch every call.
pub fn uv_embed(h: usize, w: usize, dim: usize, out: &mut Vec<f32>) {
    let aspect = w as f32 / h as f32;
    *out = build_uv_embed_chw(h, w, dim, aspect);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uv_pos_embed_shape() {
        let out = uv_pos_embed(4, 3, 8, 4.0 / 3.0, 100.0);
        assert_eq!(out.len(), 3 * 4 * 8);
    }

    #[test]
    fn uv_pos_embed_center_of_square_grid_is_near_zero() {
        // Odd square grid: the middle sample's x_coord/y_coord should be
        // (near) zero by symmetry of the linspace around 0, so sin(0)=0,
        // cos(0)=1 for every frequency at that position.
        let pw = 5;
        let ph = 5;
        let c = 8;
        let out = uv_pos_embed(pw, ph, c, 1.0, 100.0);
        let mid = (2 * pw + 2) * c; // (y=2,x=2)
        let d = c / 2;
        let f = d / 2;
        for j in 0..f {
            assert!(out[mid + j].abs() < 1e-6, "sin(x) at center should be ~0, got {}", out[mid + j]);
            assert!((out[mid + f + j] - 1.0).abs() < 1e-6, "cos(x) at center should be ~1, got {}", out[mid + f + j]);
            assert!(out[mid + d + j].abs() < 1e-6, "sin(y) at center should be ~0, got {}", out[mid + d + j]);
            assert!(
                (out[mid + d + f + j] - 1.0).abs() < 1e-6,
                "cos(y) at center should be ~1, got {}",
                out[mid + d + f + j]
            );
        }
    }

    #[test]
    fn uv_pos_embed_single_pixel_grid_uses_left_top_coord() {
        // pw==1/ph==1 degenerate case: x_coords=[left_x], y_coords=[top_y]
        // (not a division-by-zero in the linspace step).
        let out = uv_pos_embed(1, 1, 4, 1.0, 100.0);
        assert_eq!(out.len(), 4);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    #[should_panic(expected = "must be a multiple of 4")]
    fn uv_pos_embed_rejects_non_multiple_of_4_channels() {
        uv_pos_embed(2, 2, 6, 1.0, 100.0);
    }

    #[test]
    fn build_uv_embed_chw_applies_ratio_scale_and_chw_layout() {
        let h = 3;
        let w = 4;
        let dim = 8;
        let aspect = w as f32 / h as f32;
        let raw = uv_pos_embed(w, h, dim, aspect, DEFAULT_OMEGA0);
        let chw = build_uv_embed_chw(h, w, dim, aspect);
        assert_eq!(chw.len(), dim * h * w);
        // Spot-check one element: CHW[c=1,y=2,x=3] == 0.1 * raw[(y*w+x)*dim + c]
        let c = 1;
        let y = 2;
        let x = 3;
        let want = RATIO_SCALE * raw[(y * w + x) * dim + c];
        let got = chw[c * h * w + y * w + x];
        assert!((got - want).abs() < 1e-7, "got={got} want={want}");
    }

    #[test]
    fn uv_embed_matches_build_uv_embed_chw() {
        let mut out = Vec::new();
        uv_embed(5, 6, 8, &mut out);
        let expected = build_uv_embed_chw(5, 6, 8, 6.0 / 5.0);
        assert_eq!(out, expected);
    }

    #[test]
    fn cache_hit_returns_identical_result_without_recompute() {
        let mut cache = UvEmbedCache::new();
        let first: Vec<f32> = cache.get_or_build(4, 4, 8).to_vec();
        let second: Vec<f32> = cache.get_or_build(4, 4, 8).to_vec();
        assert_eq!(first, second);
        assert_eq!(cache.by_key.len(), 1, "same-key calls must not create a second cache entry");
    }

    #[test]
    fn cache_distinguishes_keys() {
        let mut cache = UvEmbedCache::new();
        let a: Vec<f32> = cache.get_or_build(4, 4, 8).to_vec();
        let b: Vec<f32> = cache.get_or_build(8, 8, 8).to_vec();
        let c_: Vec<f32> = cache.get_or_build(4, 4, 16).to_vec();
        assert_ne!(a.len(), b.len());
        assert_ne!(a.len(), c_.len());
        assert_eq!(cache.by_key.len(), 3);
    }

    #[test]
    fn aspect_is_decoupled_from_geometry_not_derived_from_h_w() {
        // Same (h, w, dim) = same-shaped output, but a different explicit
        // `aspect` must produce numerically different content — this is the
        // exact bug `get_or_build` (aspect=w/h) would silently reproduce if
        // `dpt_head`'s per-stage UV adds used it instead of
        // `get_or_build_with_aspect` (see this module's doc comment on
        // `build_uv_embed_chw`: the C++ reference always uses the full
        // target image's aspect ratio, not the UV field's own grid aspect).
        let mut cache = UvEmbedCache::new();
        let square_grid_aspect = cache.get_or_build_with_aspect(16, 16, 8, 1.0).to_vec();
        let wide_image_aspect = cache.get_or_build_with_aspect(16, 16, 8, 2.0).to_vec();
        assert_eq!(square_grid_aspect.len(), wide_image_aspect.len());
        assert_ne!(square_grid_aspect, wide_image_aspect);
        assert_eq!(cache.by_key.len(), 2, "different aspect at the same (h,w,dim) must be a distinct cache entry");
    }
}
