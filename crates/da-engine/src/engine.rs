//! The `Engine` facade: wires together everything built in Tasks 14-19
//! (`ModelConfig`, `preprocess`, `patch_embed`/`prepare_tokens`, `Backbone`,
//! `dpt_head`, `cam_pose`) into a single `Engine::load` / `Engine::infer`
//! API, loading real weights from a GGUF file into `da_graph::Weights`.
//!
//! ## Where the GGUF -> `Weights` bulk loader lives (and why)
//!
//! `da_graph::Weights` (Task 13) is defined in the `da-graph` crate, whose
//! own doc comment already says "actually loading it from a real GGUF file
//! is da-engine's job (M5)" — i.e. this was always meant to live here, not
//! in `da-graph`. There's also a hard technical reason it *can't* live in
//! `da-graph` as `impl Weights { pub fn from_gguf(...) }`: Rust's orphan
//! rule would require either `Weights` or `GgufFile`/`EngineError` to be
//! defined in the same crate as the `impl` block, and neither crate wants
//! the other as a forward dependency (`da-graph` has no `da-gguf` or
//! `da-engine` dependency, and shouldn't grow one just for this). So
//! [`weights_from_gguf`] is a free function here in `da-engine`, which
//! already depends on both `da-gguf` (for `GgufFile`) and `da-graph` (for
//! `Weights`) — this matches the task brief's explicitly-sanctioned
//! alternative ("adding a loader function in da-engine that calls
//! `Weights::insert_f32`/`insert_q8_0`").
//!
//! ## `QuantPref`: a scope-limited plumbing hook, not a real dispatch yet
//!
//! [`QuantPref`] exists so this API's *shape* is future-proof for
//! quantized-inference dispatch, but it does **nothing** yet: every tensor
//! is dequantized to `f32` via `GgufFile::tensor_f32` (which transparently
//! handles F32/F16/Q8_0 source dtypes) and stored with `Weights::insert_f32`
//! regardless of which `QuantPref` variant is passed in. This is
//! deliberate, not a silent bug: every kernel built in Tasks 16-19
//! (`patch_embed`, `vit_block`, `dpt_head`, `cam_pose`) only ever calls
//! `Weights::get_f32` — none of them call `Weights::get_q8_0`, even though
//! `da_kernels::gemm_q8_0` (Task 9) exists as a standalone kernel. Wiring a
//! real q8_0-compute path through `vit_block`/`dpt_head`'s linear/conv ops
//! is future work; until then, honoring `QuantPref::PreferQ8_0` by storing
//! q8_0 blocks would just mean every consumer's `get_f32` call panics on a
//! present-but-wrong-variant lookup, which is strictly worse than "ignored
//! today, honestly documented".
use std::path::Path;

use da_gguf::GgufFile;
use da_graph::{CpuBackend, Weights};

use crate::backbone::Backbone;
use crate::config::EngineError;
use crate::pos_embed::{prepare_tokens, PosEmbedCache};
use crate::pose::cam_pose;
use crate::preprocess::preprocess;
use crate::uv_embed::UvEmbedCache;
use crate::{dpt_head, ModelConfig};

/// Quantization preference for [`weights_from_gguf`] / [`Engine::load`].
///
/// See this module's doc comment: as of Task 20, this is a plumbing hook
/// for future quantized-inference work, not a real dispatch — both
/// variants currently produce identical (all-`f32`) `Weights` output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantPref {
    PreferF32,
    PreferQ8_0,
}

/// Bulk-loads every tensor in `f` into a fresh [`Weights`] map, keyed by its
/// GGUF tensor name, dequantized to `f32` via [`GgufFile::tensor_f32`]
/// regardless of `_prefer` — see this module's doc comment for why.
pub fn weights_from_gguf(f: &GgufFile, _prefer: QuantPref) -> Result<Weights, EngineError> {
    let mut weights = Weights::new();
    // Collect names first: `tensor_f32` borrows `f` immutably (fine to
    // re-borrow), but this avoids holding the `tensor_names()` iterator
    // borrow across the loop body for no reason.
    let names: Vec<String> = f.tensor_names().map(|n| n.to_string()).collect();
    for name in names {
        let t = f.tensor_f32(&name)?;
        weights.insert_f32(name, t.data);
    }
    Ok(weights)
}

/// Final output of [`Engine::infer`]: dense depth + confidence maps at the
/// preprocessed pixel resolution `(h, w)`, plus the decoded camera pose
/// (extrinsics/intrinsics) for the same frame.
pub struct InferOut {
    pub depth: Vec<f32>,
    pub conf: Vec<f32>,
    pub h: usize,
    pub w: usize,
    pub extrinsics: [f32; 12],
    pub intrinsics: [f32; 9],
}

/// The end-to-end facade: owns a loaded model's config + weights, plus the
/// two input-independent caches ([`PosEmbedCache`], [`UvEmbedCache`]) that
/// are safe (and worthwhile) to reuse across repeated [`Engine::infer`]
/// calls at the same input resolution — both caches key on `(h, w)`, so a
/// changing input resolution just grows the cache rather than invalidating
/// it.
///
/// `backbone` is deliberately NOT a stored field: `Backbone<'a>` (Task 17)
/// only ever borrows `cfg`/`weights`/`backend` for the duration of a single
/// `forward` call and owns no state of its own (no field of `Backbone`
/// needs to persist across calls the way `PosEmbedCache`/`UvEmbedCache`
/// do), so `Engine::infer` constructs a fresh (zero-cost, borrow-only)
/// `Backbone::new(&self.cfg, &self.weights, &self.backend)` each time
/// instead of fighting Rust's self-referential-struct restrictions to store
/// one.
pub struct Engine {
    cfg: ModelConfig,
    weights: Weights,
    backend: CpuBackend,
    pos_cache: PosEmbedCache,
    uv_cache: UvEmbedCache,
}

impl Engine {
    /// Opens `path` as a GGUF file, parses `depthanything3.*` metadata into
    /// a [`ModelConfig`], bulk-loads every tensor into a [`Weights`] map
    /// (see [`weights_from_gguf`]), and returns a ready-to-`infer` `Engine`.
    pub fn load(path: &Path, quant_prefer: QuantPref) -> Result<Engine, EngineError> {
        let f = GgufFile::open(path)?;
        let cfg = ModelConfig::from_gguf(&f)?;
        let weights = weights_from_gguf(&f, quant_prefer)?;
        Ok(Engine {
            cfg,
            weights,
            backend: CpuBackend::new(),
            pos_cache: PosEmbedCache::new(),
            uv_cache: UvEmbedCache::new(),
        })
    }

    /// Runs the full depth+pose pipeline on one raw HWC `u8` image:
    /// `preprocess` -> `prepare_tokens` (patch_embed + CLS/register prepend
    /// + pos-embed add) -> `Backbone::forward` (the `cfg.depth`-layer ViT
    /// stack, capturing `cfg.out_layers`' `feat`/`cam_token` outputs) ->
    /// `dpt_head` (dense depth/conf) + `cam_pose` (camera extrinsics/
    /// intrinsics, from the LAST out-layer's `cam_token` — for DA3-BASE's
    /// `out_layers = [5,7,9,11]`, that's layer 11, the final/deepest
    /// captured layer, matching `../src/depth_anything3.cpp`'s use of the
    /// backbone's last `cam_token` output for pose regression).
    ///
    /// Returns `Result<InferOut, EngineError>` rather than the task brief's
    /// literal bare `InferOut` return type: `cam_pose` already returns
    /// `Result<PoseOut, EngineError>` (Task 19's established convention —
    /// `EngineError::CamTokenDimMismatch` is a genuine runtime/input error,
    /// not a panic-worthy structural bug), so silently unwrapping that here
    /// would either panic on a legitimate bad-input case or require
    /// swallowing the error information. Propagating `Result` all the way
    /// out is consistent with every other fallible entry point in this
    /// crate (`ModelConfig::from_gguf`, `weights_from_gguf`, `cam_pose`).
    ///
    /// # Panics
    /// Individual stages (`patch_embed`, `prepare_tokens`, `Backbone::forward`,
    /// `dpt_head`) panic on missing/malformed weight tensors rather than
    /// returning `Result` — this matches those modules' own established
    /// convention (documented on their `get_weight` helpers: a missing
    /// weight tensor is a structural model-loading bug, not a recoverable
    /// runtime/input error) and is NOT changed by this task.
    pub fn infer(&mut self, raw_hwc_u8: &[u8], h: usize, w: usize) -> Result<InferOut, EngineError> {
        let mut chw = Vec::new();
        let (ph, pw) = preprocess(raw_hwc_u8, h, w, &self.cfg, &mut chw);

        let mut tokens = Vec::new();
        let (gh, gw) = prepare_tokens(&chw, ph, pw, &self.cfg, &self.weights, &mut self.pos_cache, &mut tokens);

        let backbone = Backbone::new(&self.cfg, &self.weights, &self.backend);
        let bb_out = backbone.forward(&mut tokens, gh, gw, &self.cfg.out_layers);

        let depth_out = dpt_head::dpt_head(&bb_out.feats, ph, pw, &self.cfg, &self.weights, &mut self.uv_cache);

        let last_cam = bb_out
            .cam_tokens
            .last()
            .ok_or(EngineError::EmptyOutLayers)?;
        let pose_out = cam_pose(last_cam, ph, pw, &self.cfg, &self.weights)?;

        Ok(InferOut {
            depth: depth_out.depth,
            conf: depth_out.conf,
            h: depth_out.h,
            w: depth_out.w,
            extrinsics: pose_out.extrinsics,
            intrinsics: pose_out.intrinsics,
        })
    }
}
