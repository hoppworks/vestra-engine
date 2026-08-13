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

use crate::backbone::{Backbone, BackboneOutputs};
use crate::config::EngineError;
use crate::dpt_head::{HeadWorkspace, WinogradFilterCache};
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
///
/// ## Rank-2 tensors are transposed on load (the Critical fix this doc block
/// documents)
///
/// `vit_block.rs`'s and `pose.rs`'s "Linear-weight orientation" doc comments
/// both require every linear-layer weight (`vit.blk.{i}.attn_qkv.weight`,
/// `attn_proj.weight`, `mlp_fc1.weight`, `mlp_fc2.weight`, and
/// `cam.bb0/bb2/fc_t/fc_q/fc_fov.weight`) to be stored `[in_features,
/// out_features]` — the transpose of GGUF/PyTorch's native `nn.Linear.weight`
/// layout, `[out_features, in_features]`. This function transposes every
/// **rank-2** tensor (`t.shape.len() == 2`) unconditionally on load.
///
/// This was checked to be safe (not just convenient) for every tensor name
/// that actually appears in this codebase's GGUFs, per
/// `../scripts/convert_da3_to_gguf.py`: every tensor is written via
/// `np.ascontiguousarray(param.numpy())` with no reshape/squeeze, so each
/// GGUF tensor's rank is exactly its source `nn.Parameter`'s rank.
/// - The genuine 2-D `nn.Linear.weight` params above are the only rank-2
///   tensors in this model: verified by grepping every `run_linear`/
///   `linear_vec` call site in `vit_block.rs`/`pose.rs`.
/// - Conv weights (`vit.patch_embed.weight`, `head.proj.*.weight`,
///   `head.scratch.*.weight`, including the 1x1 `rn{i}.out.weight`) come from
///   `nn.Conv2d` params, which PyTorch always shapes `[out_c, in_c, kh, kw]`
///   (rank 4 even for a 1x1 kernel — no squeeze anywhere in the converter),
///   so they're untouched by the rank-2 check and keep GGUF's native
///   `[out_c, in_c, kh, kw]` order, which is exactly what
///   `da_kernels::conv::conv2d`'s `weight` parameter expects.
/// - `vit.pos_embed`, `vit.cls_token`, `vit.register_tokens`,
///   `vit.camera_token` are DINOv2-style `nn.Parameter`s with a leading
///   batch/singleton dim (e.g. `pos_embed = nn.Parameter(torch.zeros(1, rows,
///   embed_dim))` — confirmed by `convert_da3_to_gguf.py`'s
///   `bb.pos_embed.shape[1]` access, which only makes sense if dim 0 is a
///   size-1 batch axis), so their GGUF rank is 3, not 2 — also untouched.
///   These are embedding-table-style lookups (indexed by row), NOT
///   matrix-multiply weights, so transposing them would be wrong; it's
///   fortunate but verified, not assumed, that their rank already excludes
///   them from this function's rank-2 transpose.
pub fn weights_from_gguf(f: &GgufFile, _prefer: QuantPref) -> Result<Weights, EngineError> {
    let mut weights = Weights::new();
    // Collect names first: `tensor_f32` borrows `f` immutably (fine to
    // re-borrow), but this avoids holding the `tensor_names()` iterator
    // borrow across the loop body for no reason.
    let names: Vec<String> = f.tensor_names().map(|n| n.to_string()).collect();
    for name in names {
        let t = f.tensor_f32(&name)?;
        let data = if t.shape.len() == 2 {
            transpose_2d(&t.data, t.shape[0] as usize, t.shape[1] as usize)
        } else {
            t.data
        };
        weights.insert_f32(name, data);
    }
    Ok(weights)
}

/// Transposes a `[rows, cols]` row-major buffer into a `[cols, rows]`
/// row-major buffer: `out[c*rows + r] = in[r*cols + c]`. Used by
/// [`weights_from_gguf`] to convert GGUF's native `[out_features,
/// in_features]` linear-weight layout into the `[in_features,
/// out_features]` layout `da_graph::Op::Gemm` (and therefore
/// `vit_block::run_linear`/`pose::linear_vec`) require.
fn transpose_2d(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    debug_assert_eq!(
        data.len(),
        rows * cols,
        "transpose_2d: data length must equal rows*cols"
    );
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
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

/// Depth-only output used by runtime benchmarks that compare the same work
/// as the C++ and PyTorch depth paths (no camera-pose head).
pub struct DepthInferOut {
    pub depth: Vec<f32>,
    pub conf: Vec<f32>,
    pub h: usize,
    pub w: usize,
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
    wino_cache: WinogradFilterCache,
    head_workspace: HeadWorkspace,
}

impl Engine {
    /// Opens `path` as a GGUF file, parses `depthanything3.*` metadata into
    /// a [`ModelConfig`], bulk-loads every tensor into a [`Weights`] map
    /// (see [`weights_from_gguf`]), and returns a ready-to-`infer` `Engine`.
    pub fn load(path: &Path, quant_prefer: QuantPref) -> Result<Engine, EngineError> {
        let f = GgufFile::open(path)?;
        let cfg = ModelConfig::from_gguf(&f)?;
        // Verify that out_layers is strictly ascending — the invariant that
        // `Engine::infer`'s `.last()` call relies on to select the deepest
        // (final) transformer layer's cam_token for pose regression.
        debug_assert!(
            cfg.out_layers.windows(2).all(|w| w[0] < w[1]),
            "out_layers must be strictly ascending, got {:?}",
            cfg.out_layers
        );
        let weights = weights_from_gguf(&f, quant_prefer)?;
        Ok(Engine {
            cfg,
            weights,
            backend: CpuBackend::new(),
            pos_cache: PosEmbedCache::new(),
            uv_cache: UvEmbedCache::new(),
            wino_cache: WinogradFilterCache::new(),
            head_workspace: HeadWorkspace::new(),
        })
    }

    fn forward_depth(
        &mut self,
        raw_hwc_u8: &[u8],
        h: usize,
        w: usize,
    ) -> (dpt_head::DepthOut, BackboneOutputs, usize, usize) {
        let profile = std::env::var_os("DA_PROFILE").is_some();
        let started = std::time::Instant::now();
        let mut chw = Vec::new();
        let (ph, pw) = preprocess(raw_hwc_u8, h, w, &self.cfg, &mut chw);
        let preprocessed = std::time::Instant::now();

        let mut tokens = Vec::new();
        let (gh, gw) = prepare_tokens(
            &chw,
            ph,
            pw,
            &self.cfg,
            &self.weights,
            &mut self.pos_cache,
            &mut tokens,
        );
        let tokens_prepared = std::time::Instant::now();

        let backbone = Backbone::new(&self.cfg, &self.weights, &self.backend);
        let bb_out = backbone.forward(&mut tokens, gh, gw, &self.cfg.out_layers);
        let backbone_done = std::time::Instant::now();
        let depth_out = if std::env::var_os("DA3_DISABLE_HEAD_WORKSPACE").is_some() {
            dpt_head::dpt_head(
                &bb_out.feats,
                ph,
                pw,
                &self.cfg,
                &self.weights,
                &mut self.uv_cache,
                &mut self.wino_cache,
            )
        } else {
            dpt_head::dpt_head_with_workspace(
                &bb_out.feats,
                ph,
                pw,
                &self.cfg,
                &self.weights,
                &mut self.uv_cache,
                &mut self.wino_cache,
                &self.head_workspace,
            )
        };
        if profile {
            let head_done = std::time::Instant::now();
            eprintln!(
                "profile: preprocess={:.1}ms tokens={:.1}ms backbone={:.1}ms head={:.1}ms",
                (preprocessed - started).as_secs_f64() * 1e3,
                (tokens_prepared - preprocessed).as_secs_f64() * 1e3,
                (backbone_done - tokens_prepared).as_secs_f64() * 1e3,
                (head_done - backbone_done).as_secs_f64() * 1e3,
            );
        }
        (depth_out, bb_out, ph, pw)
    }

    /// Runs only preprocessing, backbone and the depth/confidence head.
    /// This is the fair timing path against reference depth-only runners.
    pub fn infer_depth(
        &mut self,
        raw_hwc_u8: &[u8],
        h: usize,
        w: usize,
    ) -> Result<DepthInferOut, EngineError> {
        let (depth_out, _, _, _) = self.forward_depth(raw_hwc_u8, h, w);
        Ok(DepthInferOut {
            depth: depth_out.depth,
            conf: depth_out.conf,
            h: depth_out.h,
            w: depth_out.w,
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
    pub fn infer(
        &mut self,
        raw_hwc_u8: &[u8],
        h: usize,
        w: usize,
    ) -> Result<InferOut, EngineError> {
        let (depth_out, bb_out, ph, pw) = self.forward_depth(raw_hwc_u8, h, w);

        // Select the camera token from the LAST (deepest) out-layer for pose regression.
        // This relies on the invariant that `cfg.out_layers` is strictly ascending
        // (so the last entry is the deepest/final layer) — defended by the `debug_assert!`
        // in `Engine::load`.
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

#[cfg(test)]
mod weights_from_gguf_tests {
    use super::*;

    /// Minimal binary GGUF writer, just enough to round-trip through
    /// `GgufFile::open` (magic, version, KV section (empty here), one
    /// tensor-info entry with a real multi-dim shape, alignment padding,
    /// then the tensor's `f32` data block) — mirrors the pattern already
    /// used by `tests/e2e_native.rs`'s `GgufBuilder`, but supports an
    /// arbitrary-rank shape (that test helper hardcodes rank 1) since this
    /// regression test's whole point is exercising the rank-2 transpose
    /// path in `weights_from_gguf`.
    fn write_minimal_gguf(path: &std::path::Path, tensor_name: &str, shape: &[u64], data: &[f32]) {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&2u32.to_le_bytes()); // version
        buf.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&0u64.to_le_bytes()); // kv_count

        // Tensor-info section: name, n_dims, dims (inner->outer per
        // `GgufFile::tensor_f32`'s "dims sind inner→outer gespeichert, also
        // umdrehen" comment, so we write `shape` reversed), dtype (0 = F32),
        // offset (0, the only tensor).
        buf.extend_from_slice(&(tensor_name.len() as u64).to_le_bytes());
        buf.extend_from_slice(tensor_name.as_bytes());
        buf.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for &d in shape.iter().rev() {
            buf.extend_from_slice(&d.to_le_bytes());
        }
        buf.extend_from_slice(&0u32.to_le_bytes()); // dtype = F32
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset

        let pad = (32 - (buf.len() % 32)) % 32;
        buf.extend_from_slice(&vec![0u8; pad]);
        for v in data {
            buf.extend_from_slice(&v.to_le_bytes());
        }

        std::fs::write(path, &buf).expect("write temp gguf");
    }

    /// Regression test for Critical finding C1: `weights_from_gguf` must
    /// transpose every rank-2 tensor from GGUF's native `[rows, cols]`
    /// (`[out_features, in_features]` for a real linear weight) into
    /// `[cols, rows]` on load. Constructs a minimal synthetic GGUF holding
    /// one known 2x3 tensor (`[[1,2,3],[4,5,6]]`) and asserts the loaded
    /// `Weights` buffer equals its 3x2 transpose, `[1,4,2,5,3,6]`
    /// flattened row-major — exactly the cheap, no-real-model-required
    /// regression guard the final reviewer asked for.
    #[test]
    fn weights_from_gguf_transposes_rank2_tensors() {
        let counter = std::sync::atomic::AtomicU64::new(0);
        let n = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "da_engine_weights_from_gguf_transpose_test_{}_{}.gguf",
            std::process::id(),
            n
        ));

        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [[1,2,3],[4,5,6]], row-major [2,3]
        write_minimal_gguf(&path, "some.linear.weight", &[2, 3], &data);

        let f = GgufFile::open(&path).expect("open synthetic gguf");
        let weights = weights_from_gguf(&f, QuantPref::PreferF32).expect("weights_from_gguf");
        let got = weights
            .get_f32("some.linear.weight")
            .expect("tensor present");

        assert_eq!(
            got,
            &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0][..],
            "rank-2 tensor must be transposed [2,3] -> [3,2] on load"
        );

        let _ = std::fs::remove_file(&path);
    }

    /// Companion assertion: a rank-1 tensor (e.g. a bias or LayerNorm gamma)
    /// must pass through `weights_from_gguf` unmodified — only rank-2
    /// tensors are transposed.
    #[test]
    fn weights_from_gguf_leaves_rank1_tensors_unmodified() {
        let counter = std::sync::atomic::AtomicU64::new(0);
        let n = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "da_engine_weights_from_gguf_rank1_test_{}_{}.gguf",
            std::process::id(),
            n
        ));

        let data = vec![10.0f32, 20.0, 30.0];
        write_minimal_gguf(&path, "some.bias", &[3], &data);

        let f = GgufFile::open(&path).expect("open synthetic gguf");
        let weights = weights_from_gguf(&f, QuantPref::PreferF32).expect("weights_from_gguf");
        let got = weights.get_f32("some.bias").expect("tensor present");

        assert_eq!(got, &data[..], "rank-1 tensor must be loaded unmodified");

        let _ = std::fs::remove_file(&path);
    }
}
