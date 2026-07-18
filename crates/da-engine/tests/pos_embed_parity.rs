use da_engine::{ModelConfig, PosEmbedCache};
use da_gguf::GgufFile;
use da_graph::Weights;
use da_parity::{assert_parity, dumps_path, Dumps};
use std::path::Path;

/// Real DA3-BASE model, provided via `../scripts/download_model.py` — same
/// convention as `tests/config_from_model.rs`.
fn model() -> Option<GgufFile> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../models/da3-base-f16.gguf");
    if !p.exists() {
        eprintln!("[skip] no model at {}", p.display());
        return None;
    }
    Some(GgufFile::open(&p).unwrap())
}

/// Loads the handful of weight tensors `patch_embed`/`prepare_tokens` need
/// straight out of the model GGUF into a `Weights` map. This is a
/// test-local, minimal stand-in for Task 20's real weight-loading — it only
/// loads the tensors this test exercises, by the tensor names documented on
/// `da_engine::{PATCH_EMBED_WEIGHT, PATCH_EMBED_BIAS, POS_EMBED_WEIGHT,
/// CLS_TOKEN_WEIGHT}`.
fn load_weights(g: &GgufFile) -> Weights {
    let mut w = Weights::new();
    for name in [
        da_engine::PATCH_EMBED_WEIGHT,
        da_engine::PATCH_EMBED_BIAS,
        da_engine::POS_EMBED_WEIGHT,
        da_engine::CLS_TOKEN_WEIGHT,
    ] {
        let t = g.tensor_f32(name).unwrap_or_else(|e| panic!("missing/unreadable tensor {name:?}: {e}"));
        w.insert_f32(name, t.data);
    }
    // Register tokens are optional (see REGISTER_TOKENS_WEIGHT doc comment) —
    // load only if present, so models without them still work.
    if let Ok(t) = g.tensor_f32(da_engine::REGISTER_TOKENS_WEIGHT) {
        w.insert_f32(da_engine::REGISTER_TOKENS_WEIGHT, t.data);
    }
    w
}

/// Gates `prepare_tokens` (patch_embed -> CLS-prepend -> cached bicubic
/// pos-embed add) against the `pos_embed_added` reference dump, per Task 16's
/// brief.
///
/// SKIPS (does not fail) when either `../models/da3-base-f16.gguf` (real
/// model weights) or `../dumps/reference.gguf` (captured activations) is
/// absent, which is the case in this environment — same pattern as
/// `tests/preprocess_parity.rs` and `tests/config_from_model.rs`. That means
/// the bicubic pos-embed interpolation implemented in
/// `da_engine::pos_embed` is numerically UNVERIFIED against ground truth
/// here; it was instead transcribed directly from the C++ reference's
/// `DinoBackbone::interp_pos_embed` (`../src/dino_backbone.cpp`) — see that
/// module's doc comments for the byte-for-byte comparison.
#[test]
fn prepare_tokens_matches_reference_pos_embed_added() {
    let Some(model_gguf) = model() else { return };

    let (g, m) = (dumps_path("reference.gguf"), dumps_path("manifest.json"));
    if !g.exists() {
        eprintln!("[skip] no dumps");
        return;
    }
    let d = Dumps::open(&g, &m).unwrap();

    let cfg = ModelConfig::from_gguf(&model_gguf).expect("valid depthanything3 model should parse");
    let weights = load_weights(&model_gguf);

    let input = d.reference("input_image").unwrap(); // (3,H,W) NCHW, normalized
    let (h, w) = match input.shape.as_slice() {
        [3, hh, ww] => (*hh as usize, *ww as usize),
        other => panic!("unexpected input_image shape: {other:?}"),
    };

    let expected = d.reference("pos_embed_added").unwrap();
    let embed = cfg.embed_dim as usize;
    assert_eq!(
        expected.data.len() % embed,
        0,
        "pos_embed_added length must be a multiple of embed_dim"
    );

    let mut cache = PosEmbedCache::new();
    let mut tokens = Vec::new();
    da_engine::prepare_tokens(&input.data, h, w, &cfg, &weights, &mut cache, &mut tokens);

    assert_parity(&tokens, &expected.data, d.atol(), d.rtol(), "prepare_tokens/pos_embed_added");
}
