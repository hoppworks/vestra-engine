use da_gguf::GgufFile;
use da_graph::Weights;
use da_parity::{assert_parity, dumps_path, Dumps};
use std::path::Path;
use vestra_engine::{cam_pose, ModelConfig};

/// Real DA3-BASE model, provided via `../scripts/download_model.py` — same
/// convention as `tests/backbone_parity.rs`/`tests/dpt_parity.rs`.
fn model() -> Option<GgufFile> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../models/da3-base-f16.gguf");
    if !p.exists() {
        eprintln!("[skip] no model at {}", p.display());
        return None;
    }
    Some(GgufFile::open(&p).unwrap())
}

/// Loads `cam.bb0`/`cam.bb2`/`cam.fc_t`/`cam.fc_q`/`cam.fc_fov`
/// weight/bias tensors, transposing the `.weight` matrices from GGUF's
/// `[out_features, in_features]` layout into `pose.rs`'s expected
/// `[in_features, out_features]` layout — same convention as
/// `backbone_parity.rs::load_transposed_2d`. This is a test-local, minimal
/// stand-in for Task 20's real weight-loading.
fn load_transposed_2d(g: &GgufFile, name: &str) -> (Vec<f32>, usize, usize) {
    let t = g
        .tensor_f32(name)
        .unwrap_or_else(|e| panic!("missing/unreadable tensor {name:?}: {e}"));
    // GGUF ne[] is [in_features, out_features] fastest-varying-first for a
    // torch nn.Linear.weight of shape [out_features, in_features] stored
    // row-major -> t.shape (our convention here) is [out_features, in_features].
    let (out_features, in_features) = match t.shape.as_slice() {
        [o, i] => (*o as usize, *i as usize),
        other => panic!("unexpected 2D weight shape for {name}: {other:?}"),
    };
    assert_eq!(
        t.data.len(),
        out_features * in_features,
        "{name} unexpected element count"
    );
    let mut out = vec![0f32; in_features * out_features];
    for o in 0..out_features {
        for i in 0..in_features {
            out[i * out_features + o] = t.data[o * in_features + i];
        }
    }
    (out, in_features, out_features)
}

fn load_cam_weights(g: &GgufFile) -> Weights {
    let mut w = Weights::new();
    for base in ["cam.bb0", "cam.bb2", "cam.fc_t", "cam.fc_q", "cam.fc_fov"] {
        let (wt, _in_f, _out_f) = load_transposed_2d(g, &format!("{base}.weight"));
        w.insert_f32(format!("{base}.weight"), wt);
        let bias = g
            .tensor_f32(&format!("{base}.bias"))
            .unwrap_or_else(|e| panic!("missing/unreadable tensor {base}.bias: {e}"));
        w.insert_f32(format!("{base}.bias"), bias.data);
    }
    w
}

/// Gates [`cam_pose`] against the `cam_token_in`/`pose_enc`/`extrinsics`/
/// `intrinsics` reference dumps (`cam_token_in` == `cam_token_11`, the
/// layer-11 camera token — see task brief).
///
/// SKIPS (does not fail) when either `../models/da3-base-f16.gguf` or
/// `../dumps/reference.gguf` is absent, which is the case in this
/// environment — same pattern as every other dump-gated test in this crate.
/// That means `cam_pose`'s MLP/head forward pass is numerically UNVERIFIED
/// against ground truth here. See `pose.rs`'s module doc comment for what
/// IS independently verified (the quaternion/matrix/intrinsics decode math,
/// via synthetic unit tests in `pose.rs` itself) versus what remains a
/// documented, structurally-transcribed-but-unverified assumption (the MLP
/// weight forward pass).
#[test]
fn cam_pose_matches_reference_pose_enc_extrinsics_intrinsics() {
    let Some(model_gguf) = model() else { return };

    let (g, m) = (dumps_path("reference.gguf"), dumps_path("manifest.json"));
    if !g.exists() {
        eprintln!("[skip] no dumps");
        return;
    }
    let d = Dumps::open(&g, &m).unwrap();

    let cfg = ModelConfig::from_gguf(&model_gguf).expect("valid depthanything3 model should parse");
    let weights = load_cam_weights(&model_gguf);

    let Ok(cam_token_in) = d
        .reference("cam_token_in")
        .or_else(|_| d.reference("cam_token_11"))
    else {
        eprintln!("[skip] no cam_token_in/cam_token_11 dump");
        return;
    };

    let input = d.reference("input_image").unwrap();
    let (h, w) = match input.shape.as_slice() {
        [3, hh, ww] => (*hh as usize, *ww as usize),
        other => panic!("unexpected input_image shape: {other:?}"),
    };

    let out = cam_pose(&cam_token_in.data, h, w, &cfg, &weights)
        .expect("cam_pose should succeed on real weights");

    if let Ok(expected) = d.reference("pose_enc") {
        assert_parity(
            &out.pose_enc,
            &expected.data,
            d.atol(),
            d.rtol(),
            "pose_enc",
        );
    } else {
        eprintln!("[skip] no pose_enc dump");
    }

    if let Ok(expected) = d.reference("extrinsics") {
        assert_parity(
            &out.extrinsics,
            &expected.data,
            d.atol(),
            d.rtol(),
            "extrinsics",
        );
    } else {
        eprintln!("[skip] no extrinsics dump");
    }

    if let Ok(expected) = d.reference("intrinsics") {
        assert_parity(
            &out.intrinsics,
            &expected.data,
            d.atol(),
            d.rtol(),
            "intrinsics",
        );
    } else {
        eprintln!("[skip] no intrinsics dump");
    }
}
