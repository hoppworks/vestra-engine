use da_engine::{Backbone, ModelConfig, PosEmbedCache};
use da_gguf::GgufFile;
use da_graph::{CpuBackend, Weights};
use da_parity::{assert_parity, dumps_path, Dumps};
use std::path::Path;

/// Real DA3-BASE model, provided via `../scripts/download_model.py` — same
/// convention as `tests/config_from_model.rs`/`tests/pos_embed_parity.rs`.
fn model() -> Option<GgufFile> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../models/da3-base-f16.gguf");
    if !p.exists() {
        eprintln!("[skip] no model at {}", p.display());
        return None;
    }
    Some(GgufFile::open(&p).unwrap())
}

/// Loads a 2D linear weight tensor and transposes it from GGUF/PyTorch's
/// `[out_features, in_features]` row-major layout into the
/// `[in_features, out_features]` layout `da_graph::Op::Gemm` (and therefore
/// `da_engine::vit_block::run_linear`) requires — see `vit_block.rs`'s
/// module doc comment ("Linear-weight orientation") for the full rationale.
/// This is a test-local, minimal stand-in for Task 20's real weight-loading.
fn load_transposed_2d(g: &GgufFile, name: &str, out_features: usize, in_features: usize) -> Vec<f32> {
    let t = g.tensor_f32(name).unwrap_or_else(|e| panic!("missing/unreadable tensor {name:?}: {e}"));
    assert_eq!(t.data.len(), out_features * in_features, "{name} unexpected element count");
    let mut out = vec![0f32; in_features * out_features];
    for o in 0..out_features {
        for i in 0..in_features {
            out[i * out_features + o] = t.data[o * in_features + i];
        }
    }
    out
}

fn load_1d(g: &GgufFile, name: &str) -> Option<Vec<f32>> {
    g.tensor_f32(name).ok().map(|t| t.data)
}

/// Loads every weight tensor `Backbone::forward` needs for all `cfg.depth`
/// layers, transposing the linear weights per `load_transposed_2d`'s
/// convention. `ls1`/`ls2`/`attn_qnorm`/`attn_knorm` are loaded only if
/// present (presence-gating — see `vit_block.rs`'s trap #3 discussion).
fn load_weights(g: &GgufFile, cfg: &ModelConfig) -> Weights {
    let embed = cfg.embed_dim as usize;
    let mlp_hidden = cfg.mlp_hidden as usize;
    let head_dim = cfg.head_dim as usize;
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
    if let Some(rt) = load_1d(g, da_engine::REGISTER_TOKENS_WEIGHT) {
        w.insert_f32(da_engine::REGISTER_TOKENS_WEIGHT, rt);
    }
    // Final vit.norm + camera_token: both required by Backbone::forward's
    // post-process (vit.norm always) and camera-token injection (only when
    // cfg.alt_start >= 0, but harmless to load unconditionally).
    for name in [da_engine::VIT_NORM_WEIGHT, da_engine::VIT_NORM_BIAS] {
        let t = g.tensor_f32(name).unwrap_or_else(|e| panic!("missing/unreadable tensor {name:?}: {e}"));
        w.insert_f32(name, t.data);
    }
    if let Some(ct) = load_1d(g, da_engine::CAMERA_TOKEN_WEIGHT) {
        w.insert_f32(da_engine::CAMERA_TOKEN_WEIGHT, ct);
    }

    for i in 0..cfg.depth as usize {
        let p = |suffix: &str| format!("vit.blk.{i}.{suffix}");

        for suffix in ["norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias", "attn_qkv.bias", "attn_proj.bias"] {
            let name = p(suffix);
            if let Some(v) = load_1d(g, &name) {
                w.insert_f32(name, v);
            }
        }

        let qkv_name = p("attn_qkv.weight");
        w.insert_f32(qkv_name.clone(), load_transposed_2d(g, &qkv_name, 3 * embed, embed));
        let proj_name = p("attn_proj.weight");
        w.insert_f32(proj_name.clone(), load_transposed_2d(g, &proj_name, embed, embed));
        let fc1w = p("mlp_fc1.weight");
        w.insert_f32(fc1w.clone(), load_transposed_2d(g, &fc1w, mlp_hidden, embed));
        let fc1b = p("mlp_fc1.bias");
        if let Some(v) = load_1d(g, &fc1b) {
            w.insert_f32(fc1b, v);
        }
        let fc2w = p("mlp_fc2.weight");
        w.insert_f32(fc2w.clone(), load_transposed_2d(g, &fc2w, embed, mlp_hidden));
        let fc2b = p("mlp_fc2.bias");
        if let Some(v) = load_1d(g, &fc2b) {
            w.insert_f32(fc2b, v);
        }

        for suffix in ["ls1", "ls2"] {
            let name = p(suffix);
            if let Some(v) = load_1d(g, &name) {
                assert_eq!(v.len(), embed, "{name} expected embed_dim");
                w.insert_f32(name, v);
            }
        }
        for suffix in ["attn_qnorm.weight", "attn_qnorm.bias", "attn_knorm.weight", "attn_knorm.bias"] {
            let name = p(suffix);
            if let Some(v) = load_1d(g, &name) {
                assert_eq!(v.len(), head_dim, "{name} expected head_dim");
                w.insert_f32(name, v);
            }
        }
    }
    w
}

/// Gates `Backbone::forward` (the full 12-layer `vit_block` stack, plus
/// camera-token injection / local-global alternation / final-norm
/// doubled-width post-processing — see `da_engine::backbone`'s module doc
/// comment) against the `feat_{5,7,9,11}` AND `cam_token_{5,7,9,11}`
/// reference dumps — "the most important milestone of M5" per this task's
/// brief.
///
/// SKIPS (does not fail) when either `../models/da3-base-f16.gguf` or
/// `../dumps/reference.gguf` is absent, which is the case in this
/// environment (no dumps available) — same pattern as every other
/// dump-gated test in this crate (`preprocess_parity.rs`,
/// `pos_embed_parity.rs`, `config_from_model.rs`). That means the full
/// `vit_block`/`Backbone` forward pass is numerically UNVERIFIED against
/// ground truth here.
///
/// DA3-BASE is expected to have `cat_token == true` (confirmed against
/// `../src/model_loader.cpp`'s default and `../scripts/dump_reference.py`'s
/// documented `[1,256,1536]`/`[1,1536]` shapes, `1536 = 2*embed_dim=2*768`),
/// so `feat_*` is expected `[256, 1536]` and `cam_token_*` is expected
/// `[1536]`. If the real model turns out to have `cat_token == false`
/// (unverified — no real GGUF read directly in this environment), the
/// shapes below would be `[256, 768]`/`[768]` instead and this test's
/// hardcoded `[256, 1536]` assumption documented in this comment would need
/// updating (the `assert_parity` shape check would fail loudly, not
/// silently, if that's the case).
#[test]
fn backbone_forward_matches_reference_feat_and_cam_layers() {
    let Some(model_gguf) = model() else { return };

    let (g, m) = (dumps_path("reference.gguf"), dumps_path("manifest.json"));
    if !g.exists() {
        eprintln!("[skip] no dumps");
        return;
    }
    let d = Dumps::open(&g, &m).unwrap();

    let cfg = ModelConfig::from_gguf(&model_gguf).expect("valid depthanything3 model should parse");
    if cfg.ffn_type == "swiglu" {
        eprintln!("[skip] ffn_type=swiglu is not implemented by vit_block (Task 17)");
        return;
    }
    let weights = load_weights(&model_gguf, &cfg);

    let input = d.reference("input_image").unwrap();
    let (h, w) = match input.shape.as_slice() {
        [3, hh, ww] => (*hh as usize, *ww as usize),
        other => panic!("unexpected input_image shape: {other:?}"),
    };

    let mut cache = PosEmbedCache::new();
    let mut tokens = Vec::new();
    let (gh, gw) = da_engine::prepare_tokens(&input.data, h, w, &cfg, &weights, &mut cache, &mut tokens);

    let backend = CpuBackend::new();
    let bb = Backbone::new(&cfg, &weights, &backend);
    let out_layers = [5, 7, 9, 11];
    let out = bb.forward(&mut tokens, gh, gw, &out_layers);

    for (i, idx) in out_layers.iter().enumerate() {
        let feat_dump = format!("feat_{idx}");
        let expected_feat = d.reference(&feat_dump).unwrap();
        assert_parity(&out.feats[i], &expected_feat.data, d.atol(), d.rtol(), &feat_dump);

        let cam_dump = format!("cam_token_{idx}");
        let expected_cam = d.reference(&cam_dump).unwrap();
        assert_parity(&out.cam_tokens[i], &expected_cam.data, d.atol(), d.rtol(), &cam_dump);
    }
}
