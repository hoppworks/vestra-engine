use da_engine::{dpt_head_debug, ModelConfig, UvEmbedCache};
use da_gguf::GgufFile;
use da_graph::Weights;
use da_parity::{assert_parity, dumps_path, Dumps};
use std::path::Path;

/// Real DA3-BASE model, provided via `../scripts/download_model.py` — same
/// convention as `tests/backbone_parity.rs`/`tests/pos_embed_parity.rs`.
fn model() -> Option<GgufFile> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../models/da3-base-f16.gguf");
    if !p.exists() {
        eprintln!("[skip] no model at {}", p.display());
        return None;
    }
    Some(GgufFile::open(&p).unwrap())
}

/// Loads every `head.*`-prefixed f32 tensor present in the model file
/// verbatim (no transpose needed — unlike `vit_block`'s linear weights,
/// `dpt_head`'s conv weights are consumed in the same `[out_c, in_c, kh,
/// kw]`/`[in_c, out_c, kh, kw]` OIHW/IOHW layout the GGUF converter writes
/// them in, per `da_kernels::conv::conv2d`/`conv_transpose2d`'s doc
/// comments). This is a test-local, minimal stand-in for Task 20's real
/// weight-loading, mirroring `backbone_parity.rs::load_weights`'s pattern.
fn load_head_weights(g: &GgufFile) -> Weights {
    let mut w = Weights::new();
    let names: Vec<String> = g
        .tensor_names()
        .filter(|n| n.starts_with("head."))
        .map(str::to_string)
        .collect();
    for name in names {
        let t = g
            .tensor_f32(&name)
            .unwrap_or_else(|e| panic!("missing/unreadable tensor {name:?}: {e}"));
        w.insert_f32(name, t.data);
    }
    w
}

/// Gates [`dpt_head_debug`] (reassemble -> resize -> RefineNet fusion ->
/// output convs -> depth/conf activations) against the
/// `head_stage{0..3}`/`head_fused`/`head_depth`/`head_depth_conf` reference
/// dumps, driven by the `feat_{5,7,9,11}` dumps as input (matching the task
/// brief's Step 1: "Zwischenstufen ... aus den gedumpten `feat_*` als
/// Eingabe").
///
/// SKIPS (does not fail) when either `../models/da3-base-f16.gguf` or
/// `../dumps/reference.gguf` is absent, which is the case in this
/// environment — same pattern as every other dump-gated test in this crate.
/// That means `dpt_head`'s full forward pass (the most architecturally
/// complex component built so far, rivaling Task 17's `vit_block`) is
/// numerically UNVERIFIED against ground truth here. See `dpt_head.rs`'s
/// module doc comment for exactly what IS verified (line-for-line
/// transcription from the C++ source, mechanically-testable pieces like the
/// UV-embed formula / align_corners resize / expp1 activation) versus what
/// remains a documented assumption (square patch grid, no
/// `head.out_channels` GGUF override).
#[test]
fn dpt_head_matches_reference_stages_fused_and_depth() {
    let Some(model_gguf) = model() else { return };

    let (g, m) = (dumps_path("reference.gguf"), dumps_path("manifest.json"));
    if !g.exists() {
        eprintln!("[skip] no dumps");
        return;
    }
    let d = Dumps::open(&g, &m).unwrap();

    let cfg = ModelConfig::from_gguf(&model_gguf).expect("valid depthanything3 model should parse");
    let weights = load_head_weights(&model_gguf);

    let input = d.reference("input_image").unwrap();
    let (h, w) = match input.shape.as_slice() {
        [3, hh, ww] => (*hh as usize, *ww as usize),
        other => panic!("unexpected input_image shape: {other:?}"),
    };

    let out_layers = [5, 7, 9, 11];
    let feats: Vec<Vec<f32>> = out_layers
        .iter()
        .map(|idx| d.reference(&format!("feat_{idx}")).unwrap().data)
        .collect();

    let mut cache = UvEmbedCache::new();
    let (depth_out, debug) = dpt_head_debug(&feats, h, w, &cfg, &weights, &mut cache);

    for s in 0..4 {
        let name = format!("head_stage{s}");
        if let Ok(expected) = d.reference(&name) {
            assert_parity(&debug.stages[s], &expected.data, d.atol(), d.rtol(), &name);
        } else {
            eprintln!("[skip] no {name} dump");
        }
    }

    if let Ok(expected) = d.reference("head_fused") {
        assert_parity(
            &debug.fused,
            &expected.data,
            d.atol(),
            d.rtol(),
            "head_fused",
        );
    } else {
        eprintln!("[skip] no head_fused dump");
    }

    if let Ok(expected) = d.reference("head_depth") {
        assert_parity(
            &depth_out.depth,
            &expected.data,
            d.atol(),
            d.rtol(),
            "head_depth",
        );
    } else {
        eprintln!("[skip] no head_depth dump");
    }

    if let Ok(expected) = d.reference("head_depth_conf") {
        assert_parity(
            &depth_out.conf,
            &expected.data,
            d.atol(),
            d.rtol(),
            "head_depth_conf",
        );
    } else {
        eprintln!("[skip] no head_depth_conf dump");
    }

    // Sanity asserts on the activation formulas, independent of dumps:
    // depth = exp(logit) > 0 always; conf = exp(logit)+1 >= 1.0 always.
    assert!(depth_out.depth.iter().all(|&v| v > 0.0 && v.is_finite()));
    assert!(depth_out.conf.iter().all(|&v| v >= 1.0 && v.is_finite()));
}
