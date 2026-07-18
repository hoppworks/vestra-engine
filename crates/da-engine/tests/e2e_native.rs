//! End-to-end tests for `da_engine::Engine` — Task 20's facade wiring
//! `preprocess -> prepare_tokens -> Backbone -> dpt_head + cam_pose`.
//!
//! Two independent tests here, deliberately covering different things:
//!
//! 1. [`engine_matches_reference_depth_and_pose`] — the REAL parity gate
//!    against `raw_image` -> `head_depth`/`extrinsics`/`intrinsics` from
//!    `../dumps/reference.gguf`, loading a real `../models/da3-base-f16.gguf`
//!    via `Engine::load`. SKIPS (does not fail) in this environment: no
//!    dumps and no real model GGUF exist here. This is the honest, still-
//!    open numerical-correctness gate — nothing in this task can close it
//!    without a real model + dumps.
//!
//! 2. [`engine_load_and_infer_run_to_completion_on_synthetic_gguf`] — builds
//!    a tiny, fully self-contained synthetic GGUF file on disk (real binary
//!    GGUF bytes, not a mocked `Weights` map) with every tensor name+shape
//!    `Engine::load`/`Engine::infer` need, and proves the *facade's wiring*
//!    (weight loading, data flow between `preprocess`/`prepare_tokens`/
//!    `Backbone`/`dpt_head`/`cam_pose`, shape composition end-to-end) works
//!    without panicking, independent of any dump/model file. This does NOT
//!    prove numerical correctness against the C++ reference — the weights
//!    are small pseudo-random noise, so the output numbers are meaningless
//!    — but it is real, dump-independent coverage of the one thing this
//!    task actually adds: composing Tasks 14-19 into a single working
//!    `Engine::load`/`Engine::infer` call.

use da_engine::{Engine, QuantPref};
use da_gguf::GgufFile;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------
// Real-model/dump-gated parity test (Step 1 of the task brief).
// ---------------------------------------------------------------------

fn model_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../models/da3-base-f16.gguf")
}

fn dumps_gguf_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../dumps/reference.gguf")
}

/// Gates `Engine::load` + `Engine::infer` against the `raw_image` ->
/// `head_depth`/`extrinsics`/`intrinsics` reference dump — the "culmination"
/// gate the task brief describes. SKIPS (does not fail) when either the
/// real model or the dumps are absent, matching every other dump-gated test
/// in this crate (`backbone_parity.rs`, `dpt_parity.rs`, `pose_parity.rs`,
/// `preprocess_parity.rs`, `pos_embed_parity.rs`, `config_from_model.rs`).
#[test]
fn engine_matches_reference_depth_and_pose() {
    let model = model_path();
    if !model.exists() {
        eprintln!("[skip] no model at {}", model.display());
        return;
    }
    let dumps = dumps_gguf_path();
    if !dumps.exists() {
        eprintln!("[skip] no dumps at {}", dumps.display());
        return;
    }

    // Both files are present in some environment this test might run in —
    // wire the real gate. `da_parity::Dumps`/`assert_parity` mirror the
    // exact pattern `backbone_parity.rs`/`dpt_parity.rs`/`pose_parity.rs`
    // already use.
    use da_parity::{assert_parity, Dumps};

    let mut engine = Engine::load(&model, QuantPref::PreferF32).expect("Engine::load should succeed against a real model");

    let d = Dumps::open(&dumps, &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../dumps/manifest.json")).unwrap();
    let raw = d.reference("raw_image").expect("dumps must contain raw_image");
    // raw_image is expected HWC u8-equivalent per the dump convention used
    // elsewhere in this crate's parity tests (f32 in [0,255] or [0,1] —
    // Task 20b's honesty note about `preprocess` vs `preprocess_real` scope
    // applies here too: this gate only exercises the identity-resize
    // regime, same as every other parity test in this plan).
    let (h, w) = match raw.shape.as_slice() {
        [hh, ww, 3] => (*hh as usize, *ww as usize),
        [3, hh, ww] => (*hh as usize, *ww as usize),
        other => panic!("unexpected raw_image shape: {other:?}"),
    };
    let raw_u8: Vec<u8> = raw.data.iter().map(|&v| v.round().clamp(0.0, 255.0) as u8).collect();

    let out = engine.infer(&raw_u8, h, w).expect("Engine::infer should succeed against a real model");

    let expected_depth = d.reference("head_depth").expect("dumps must contain head_depth");
    assert_parity(&out.depth, &expected_depth.data, d.atol(), d.rtol(), "head_depth");

    let expected_extrinsics = d.reference("extrinsics").expect("dumps must contain extrinsics");
    assert_parity(&out.extrinsics, &expected_extrinsics.data, d.atol(), d.rtol(), "extrinsics");

    let expected_intrinsics = d.reference("intrinsics").expect("dumps must contain intrinsics");
    assert_parity(&out.intrinsics, &expected_intrinsics.data, d.atol(), d.rtol(), "intrinsics");
}

// ---------------------------------------------------------------------
// Synthetic-GGUF plumbing test (dump-independent).
// ---------------------------------------------------------------------

/// A tiny binary GGUF writer, independent of (and much simpler than) a real
/// model converter — just enough of the format (see `da-gguf/src/reader.rs`)
/// to round-trip through `GgufFile::open`: magic, version, KV section,
/// tensor-info section, alignment padding, then contiguous per-tensor `f32`
/// data blocks at the offsets the tensor-info section declares.
struct GgufBuilder {
    kv: Vec<u8>,
    kv_count: u64,
    tensor_info: Vec<u8>,
    tensor_count: u64,
    data: Vec<u8>,
}

impl GgufBuilder {
    fn new() -> Self {
        GgufBuilder { kv: Vec::new(), kv_count: 0, tensor_info: Vec::new(), tensor_count: 0, data: Vec::new() }
    }

    fn write_gguf_string(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    fn kv_str(&mut self, key: &str, val: &str) {
        Self::write_gguf_string(&mut self.kv, key);
        self.kv.extend_from_slice(&8u32.to_le_bytes()); // vtype 8 = string
        Self::write_gguf_string(&mut self.kv, val);
        self.kv_count += 1;
    }

    fn kv_u32(&mut self, key: &str, val: u32) {
        Self::write_gguf_string(&mut self.kv, key);
        self.kv.extend_from_slice(&4u32.to_le_bytes()); // vtype 4 = uint32
        self.kv.extend_from_slice(&val.to_le_bytes());
        self.kv_count += 1;
    }

    fn kv_i32(&mut self, key: &str, val: i32) {
        Self::write_gguf_string(&mut self.kv, key);
        self.kv.extend_from_slice(&5u32.to_le_bytes()); // vtype 5 = int32
        self.kv.extend_from_slice(&val.to_le_bytes());
        self.kv_count += 1;
    }

    fn kv_f32(&mut self, key: &str, val: f32) {
        Self::write_gguf_string(&mut self.kv, key);
        self.kv.extend_from_slice(&6u32.to_le_bytes()); // vtype 6 = float32
        self.kv.extend_from_slice(&val.to_le_bytes());
        self.kv_count += 1;
    }

    fn kv_arr_f32(&mut self, key: &str, vals: &[f32]) {
        Self::write_gguf_string(&mut self.kv, key);
        self.kv.extend_from_slice(&9u32.to_le_bytes()); // vtype 9 = array
        self.kv.extend_from_slice(&6u32.to_le_bytes()); // elem type 6 = float32
        self.kv.extend_from_slice(&(vals.len() as u64).to_le_bytes());
        for v in vals {
            self.kv.extend_from_slice(&v.to_le_bytes());
        }
        self.kv_count += 1;
    }

    fn kv_arr_i32(&mut self, key: &str, vals: &[i32]) {
        Self::write_gguf_string(&mut self.kv, key);
        self.kv.extend_from_slice(&9u32.to_le_bytes()); // vtype 9 = array
        self.kv.extend_from_slice(&5u32.to_le_bytes()); // elem type 5 = int32
        self.kv.extend_from_slice(&(vals.len() as u64).to_le_bytes());
        for v in vals {
            self.kv.extend_from_slice(&v.to_le_bytes());
        }
        self.kv_count += 1;
    }

    /// Adds a 1-D `f32` tensor (shape doesn't matter for `Weights::insert_f32`
    /// consumers — only the flat element count does — so every tensor here
    /// is declared as a single dimension `[len]`).
    fn tensor_f32(&mut self, name: &str, values: &[f32]) {
        Self::write_gguf_string(&mut self.tensor_info, name);
        self.tensor_info.extend_from_slice(&1u32.to_le_bytes()); // n_dims = 1
        self.tensor_info.extend_from_slice(&(values.len() as u64).to_le_bytes());
        self.tensor_info.extend_from_slice(&0u32.to_le_bytes()); // dtype 0 = F32
        // offset relative to data_start = current cumulative data length.
        self.tensor_info.extend_from_slice(&(self.data.len() as u64).to_le_bytes());
        for v in values {
            self.data.extend_from_slice(&v.to_le_bytes());
        }
        self.tensor_count += 1;
    }

    fn build(self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&2u32.to_le_bytes()); // version
        buf.extend_from_slice(&self.tensor_count.to_le_bytes());
        buf.extend_from_slice(&self.kv_count.to_le_bytes());
        buf.extend_from_slice(&self.kv);
        buf.extend_from_slice(&self.tensor_info);
        // Pad to the default alignment (32) before the data block, matching
        // `GgufFile::open`'s `data_start` computation.
        let pad = (32 - (buf.len() % 32)) % 32;
        buf.extend_from_slice(&vec![0u8; pad]);
        buf.extend_from_slice(&self.data);
        buf
    }
}

/// Deterministic small-magnitude pseudo-random generator (xorshift32),
/// mirroring the pattern already used by `backbone.rs`'s
/// `synthetic_weights` test helper — scaled down (`*0.02`) so that summing
/// across the DPT head's wide fixed channel counts (96/192/384/768,
/// hardcoded independent of this test's tiny `embed_dim` — see
/// `dpt_head.rs`'s `DEFAULT_OC`) doesn't risk `exp()` overflow in the final
/// depth/conf activation. This test only asserts "ran to completion with
/// correctly-shaped output", not numerical values, so exact magnitude
/// doesn't matter beyond staying finite-ish.
struct Xorshift32(u32);
impl Xorshift32 {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        (((self.0 as f32) / (u32::MAX as f32)) * 2.0 - 1.0) * 0.02
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32()).collect()
    }
}

/// Builds a complete synthetic GGUF file (KV metadata + every weight tensor
/// `Engine::load`/`Engine::infer` touch) at a tiny scale:
/// - `patch_size=2`, `image_size=4` -> a 2x2 patch grid (4 patches + 1 CLS
///   = 5 tokens), so `preprocess`'s identity-resize regime (input already
///   `image_size x image_size`) applies — same scope limitation documented
///   on `preprocess.rs`.
/// - `embed_dim=4`, `depth=4`, `num_heads=2`, `head_dim=2`, `mlp_hidden=8`,
///   `out_layers=[0,1,2,3]` (`dpt_head` hard-requires exactly 4 out-layer
///   feats — see `dpt_head_debug`'s `assert_eq!(feats.len(), 4, ...)`).
/// - `rope_start=-1`, `qknorm_start=-1`, `alt_start=-1`: disables RoPE/
///   QK-norm/camera-token-injection so no `attn_qnorm`/`attn_knorm`/RoPE-only
///   tensors are needed (all presence-gated per `vit_block.rs`'s doc
///   comment) — this test is about wiring, not exercising every optional
///   code path.
/// - `cat_token=true` (DA3-BASE default): `feat`/`cam_token` are
///   doubled-width (`2*embed_dim=8`), so `cam.bb0.weight`'s input dim
///   (`cam_dim_in`) must be `8` to satisfy `cam_pose`'s dimension check.
/// - The DPT head's per-stage channel counts (`oc = [96,192,384,768]`,
///   `FUSION_C=128`) are hardcoded in `dpt_head.rs` (`DEFAULT_OC`),
///   independent of this test's `embed_dim` — so those tensors are
///   necessarily NOT tiny (the largest, `head.resize.3.weight`, is
///   `768*768*3*3` floats). This is a structural property of the current
///   `dpt_head` module, not something this test can shrink further while
///   still exercising the real function.
fn build_synthetic_gguf() -> Vec<u8> {
    const PATCH: usize = 2;
    const IMAGE_SIZE: usize = 4;
    const EMBED: usize = 4;
    const DEPTH: usize = 4;
    const MLP_HIDDEN: usize = 8;
    const GRID: usize = IMAGE_SIZE / PATCH; // 2
    const C_IN: usize = 2 * EMBED; // cat_token=true -> 8
    const OC: [usize; 4] = [96, 192, 384, 768];
    const FUSION_C: usize = 128;
    const FEAT_HALF: usize = 4; // head_features/2
    const OUTPUT_DIM: usize = 2; // depth + conf

    let mut g = GgufBuilder::new();
    let mut rng = Xorshift32(0xC0FF_EE42);

    // ---- ModelConfig KV metadata ----
    g.kv_str("depthanything3.arch", "depthanything3");
    g.kv_u32("depthanything3.patch_size", PATCH as u32);
    g.kv_u32("depthanything3.image_size", IMAGE_SIZE as u32);
    g.kv_u32("depthanything3.vit.embed_dim", EMBED as u32);
    g.kv_u32("depthanything3.vit.depth", DEPTH as u32);
    g.kv_u32("depthanything3.vit.num_heads", 2);
    g.kv_u32("depthanything3.vit.head_dim", 2);
    g.kv_u32("depthanything3.vit.mlp_hidden", MLP_HIDDEN as u32);
    g.kv_u32("depthanything3.vit.num_register_tokens", 0);
    g.kv_i32("depthanything3.vit.rope_start", -1);
    g.kv_i32("depthanything3.vit.qknorm_start", -1);
    g.kv_f32("depthanything3.vit.rope_freq", 100.0);
    g.kv_f32("depthanything3.vit.ln_eps", 1e-6);
    g.kv_arr_i32("depthanything3.vit.out_layers", &[0, 1, 2, 3]);
    g.kv_str("depthanything3.vit.ffn_type", "mlp");
    g.kv_i32("depthanything3.vit.alt_start", -1);
    // cat_token has no Kv::Bool helper here (matches config.rs's test
    // pattern); true is the default when absent, so simply omit the key.
    g.kv_u32("depthanything3.head.features", 8);
    g.kv_f32("depthanything3.head.max_depth", 20.0);
    g.kv_arr_f32("depthanything3.img.mean", &[0.0, 0.0, 0.0]);
    g.kv_arr_f32("depthanything3.img.std", &[1.0, 1.0, 1.0]);
    g.kv_str("depthanything3.img.resize_mode", "bilinear");
    g.kv_u32("depthanything3.cam.dim_in", C_IN as u32);

    // ---- ViT backbone weights ----
    g.tensor_f32("vit.patch_embed.weight", &rng.vec(EMBED * 3 * PATCH * PATCH));
    g.tensor_f32("vit.patch_embed.bias", &rng.vec(EMBED));
    g.tensor_f32("vit.pos_embed", &rng.vec((GRID * GRID + 1) * EMBED));
    g.tensor_f32("vit.cls_token", &rng.vec(EMBED));
    g.tensor_f32("vit.norm.weight", &vec![1.0; EMBED]);
    g.tensor_f32("vit.norm.bias", &vec![0.0; EMBED]);
    g.tensor_f32("vit.camera_token", &rng.vec(2 * EMBED));

    for i in 0..DEPTH {
        let p = |suffix: &str| format!("vit.blk.{i}.{suffix}");
        g.tensor_f32(&p("norm1.weight"), &vec![1.0; EMBED]);
        g.tensor_f32(&p("norm1.bias"), &vec![0.0; EMBED]);
        g.tensor_f32(&p("norm2.weight"), &vec![1.0; EMBED]);
        g.tensor_f32(&p("norm2.bias"), &vec![0.0; EMBED]);
        g.tensor_f32(&p("attn_qkv.weight"), &rng.vec(EMBED * 3 * EMBED));
        g.tensor_f32(&p("attn_qkv.bias"), &rng.vec(3 * EMBED));
        g.tensor_f32(&p("attn_proj.weight"), &rng.vec(EMBED * EMBED));
        g.tensor_f32(&p("attn_proj.bias"), &rng.vec(EMBED));
        g.tensor_f32(&p("mlp_fc1.weight"), &rng.vec(EMBED * MLP_HIDDEN));
        g.tensor_f32(&p("mlp_fc1.bias"), &rng.vec(MLP_HIDDEN));
        g.tensor_f32(&p("mlp_fc2.weight"), &rng.vec(MLP_HIDDEN * EMBED));
        g.tensor_f32(&p("mlp_fc2.bias"), &rng.vec(EMBED));
    }

    // ---- DPT head weights ----
    for s in 0..4 {
        g.tensor_f32(&format!("head.proj.{s}.weight"), &rng.vec(OC[s] * C_IN));
        g.tensor_f32(&format!("head.proj.{s}.bias"), &rng.vec(OC[s]));
    }
    g.tensor_f32("head.resize.0.weight", &rng.vec(OC[0] * OC[0] * 4 * 4));
    g.tensor_f32("head.resize.0.bias", &rng.vec(OC[0]));
    g.tensor_f32("head.resize.1.weight", &rng.vec(OC[1] * OC[1] * 2 * 2));
    g.tensor_f32("head.resize.1.bias", &rng.vec(OC[1]));
    // stage 2 is Identity (no resize weight needed).
    g.tensor_f32("head.resize.3.weight", &rng.vec(OC[3] * OC[3] * 3 * 3));
    g.tensor_f32("head.resize.3.bias", &rng.vec(OC[3]));

    for s in 0..4 {
        g.tensor_f32(&format!("head.scratch.layer{}_rn.weight", s + 1), &rng.vec(FUSION_C * OC[s] * 3 * 3));
    }

    for i in 1..=4 {
        if i != 4 {
            // rn4 has no lateral, so no rc1.
            g.tensor_f32(&format!("head.scratch.rn{i}.rc1.c1.weight"), &rng.vec(FUSION_C * FUSION_C * 3 * 3));
            g.tensor_f32(&format!("head.scratch.rn{i}.rc1.c1.bias"), &rng.vec(FUSION_C));
            g.tensor_f32(&format!("head.scratch.rn{i}.rc1.c2.weight"), &rng.vec(FUSION_C * FUSION_C * 3 * 3));
            g.tensor_f32(&format!("head.scratch.rn{i}.rc1.c2.bias"), &rng.vec(FUSION_C));
        }
        g.tensor_f32(&format!("head.scratch.rn{i}.rc2.c1.weight"), &rng.vec(FUSION_C * FUSION_C * 3 * 3));
        g.tensor_f32(&format!("head.scratch.rn{i}.rc2.c1.bias"), &rng.vec(FUSION_C));
        g.tensor_f32(&format!("head.scratch.rn{i}.rc2.c2.weight"), &rng.vec(FUSION_C * FUSION_C * 3 * 3));
        g.tensor_f32(&format!("head.scratch.rn{i}.rc2.c2.bias"), &rng.vec(FUSION_C));
        g.tensor_f32(&format!("head.scratch.rn{i}.out.weight"), &rng.vec(FUSION_C * FUSION_C * 1 * 1));
        g.tensor_f32(&format!("head.scratch.rn{i}.out.bias"), &rng.vec(FUSION_C));
    }

    g.tensor_f32("head.scratch.out1.weight", &rng.vec(FEAT_HALF * FUSION_C * 3 * 3));
    g.tensor_f32("head.scratch.out1.bias", &rng.vec(FEAT_HALF));
    g.tensor_f32("head.scratch.out2a.weight", &rng.vec(32 * FEAT_HALF * 3 * 3));
    g.tensor_f32("head.scratch.out2a.bias", &rng.vec(32));
    g.tensor_f32("head.scratch.out2b.weight", &rng.vec(OUTPUT_DIM * 32));
    g.tensor_f32("head.scratch.out2b.bias", &rng.vec(OUTPUT_DIM));

    // ---- Camera pose head weights ----
    const HIDDEN0: usize = 6;
    const HIDDEN1: usize = 6;
    g.tensor_f32("cam.bb0.weight", &rng.vec(C_IN * HIDDEN0));
    g.tensor_f32("cam.bb0.bias", &rng.vec(HIDDEN0));
    g.tensor_f32("cam.bb2.weight", &rng.vec(HIDDEN0 * HIDDEN1));
    g.tensor_f32("cam.bb2.bias", &rng.vec(HIDDEN1));
    g.tensor_f32("cam.fc_t.weight", &rng.vec(HIDDEN1 * 3));
    g.tensor_f32("cam.fc_t.bias", &rng.vec(3));
    g.tensor_f32("cam.fc_q.weight", &rng.vec(HIDDEN1 * 4));
    // Nonzero-`qr` bias so the decoded quaternion isn't the degenerate
    // all-zero case (`s = 2/(qi^2+qj^2+qk^2+qr^2)` in `pose.rs::decode`
    // would divide by zero) even if upstream noise happens to net to ~0.
    g.tensor_f32("cam.fc_q.bias", &[0.0, 0.0, 0.0, 1.0]);
    g.tensor_f32("cam.fc_fov.weight", &rng.vec(HIDDEN1 * 2));
    g.tensor_f32("cam.fc_fov.bias", &[0.8, 0.8]);

    g.build()
}

/// Writes `bytes` to a unique temp file and returns its path — same
/// unique-temp-filename convention as `config.rs`'s test helpers (atomic
/// counter + PID + nanos, safe under parallel test execution).
fn write_temp_gguf(bytes: &[u8]) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let counter = COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let path = std::env::temp_dir().join(format!("da_engine_e2e_synth_{pid}_{nanos}_{counter}.gguf"));
    let mut f = std::fs::File::create(&path).expect("create temp synthetic gguf");
    f.write_all(bytes).expect("write temp synthetic gguf");
    path
}

/// The real dump-independent plumbing gate: `Engine::load` a from-scratch
/// synthetic GGUF, then `Engine::infer` a tiny synthetic image, asserting
/// only shape/completion properties (NOT numerical values — see this
/// module's doc comment for why). Proves the facade's data flow — GGUF ->
/// `ModelConfig`/`Weights` -> `preprocess` -> `prepare_tokens` -> `Backbone`
/// -> `dpt_head` + `cam_pose` -> `InferOut` — actually composes without
/// panicking, independent of any external dump/model file.
#[test]
fn engine_load_and_infer_run_to_completion_on_synthetic_gguf() {
    let bytes = build_synthetic_gguf();
    let path = write_temp_gguf(&bytes);

    // Sanity: the file we just wrote actually round-trips through
    // `GgufFile::open` before handing it to `Engine::load` (isolates a
    // builder bug from an `Engine`-side bug if this ever fails).
    {
        let f = GgufFile::open(&path).expect("synthetic gguf should parse as a valid GgufFile");
        assert!(f.tensor_names().count() > 0, "synthetic gguf should have tensors");
    }

    let mut engine = Engine::load(&path, QuantPref::PreferF32).expect("Engine::load should succeed on a well-formed synthetic gguf");

    // 4x4 RGB image (matches the synthetic model's image_size=4), simple
    // deterministic gradient content (not all-zero, so patch_embed actually
    // sees varying input).
    const IMAGE_SIZE: usize = 4;
    let mut raw = vec![0u8; IMAGE_SIZE * IMAGE_SIZE * 3];
    for y in 0..IMAGE_SIZE {
        for x in 0..IMAGE_SIZE {
            let px = (y * IMAGE_SIZE + x) * 3;
            raw[px] = ((x * 40 + y * 10) % 256) as u8;
            raw[px + 1] = ((y * 40 + x * 10) % 256) as u8;
            raw[px + 2] = 128;
        }
    }

    let out = engine.infer(&raw, IMAGE_SIZE, IMAGE_SIZE).expect("Engine::infer should run to completion on synthetic weights");

    // Shape assertions: this is what this test actually proves (see module
    // doc comment) — the facade produced output at the right resolution
    // with the right array shapes, not that the values are meaningful.
    assert_eq!(out.h, IMAGE_SIZE, "InferOut.h should equal the preprocessed pixel height");
    assert_eq!(out.w, IMAGE_SIZE, "InferOut.w should equal the preprocessed pixel width");
    assert_eq!(out.depth.len(), IMAGE_SIZE * IMAGE_SIZE, "depth map should be h*w");
    assert_eq!(out.conf.len(), IMAGE_SIZE * IMAGE_SIZE, "conf map should be h*w (output_dim=2 in this synthetic model)");
    assert_eq!(out.extrinsics.len(), 12);
    assert_eq!(out.intrinsics.len(), 9);
    // intrinsics[8] (bottom-right of the 3x3 row-major K matrix) is always
    // exactly 1.0 by construction in `pose.rs::decode` — a cheap, precise
    // sanity check that we actually reached the pose-decode step and it
    // populated real data, not a happens-to-be-zeroed buffer.
    assert_eq!(out.intrinsics[8], 1.0, "K[2][2] must be exactly 1.0 per decode()'s construction");

    let _ = std::fs::remove_file(&path);
}
