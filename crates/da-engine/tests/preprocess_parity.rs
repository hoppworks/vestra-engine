use da_engine::{preprocess, ModelConfig};
use da_parity::{assert_parity, dumps_path, Dumps};

/// Gates `preprocess` against `raw_image` -> `input_image` in the reference
/// dump, per Task 15's brief. `raw_image` is documented as 224x224x3 HWC,
/// values 0..255 (stored as f32 in the gguf tensor format, per
/// `da_gguf::tensor_f32`); this test rounds it to `u8` before feeding it to
/// `preprocess`, since `preprocess`'s public interface takes raw `u8` HWC
/// input (matching the real capture pipeline, which produces `u8` pixels).
///
/// NOTE: this test SKIPS (does not fail) when `../dumps/reference.gguf` is
/// absent, which is the case in this environment. That means the
/// normalization convention implemented in `preprocess` (255-scaling,
/// mean/std order, channel order, NCHW layout) is currently UNVERIFIED
/// against ground truth here — see the doc comment on
/// `da_engine::preprocess` for what it *was* cross-checked against
/// (the C++ reference engine's `src/preprocess.cpp`).
#[test]
fn preprocess_matches_reference_input_image() {
    let (g, m) = (dumps_path("reference.gguf"), dumps_path("manifest.json"));
    if !g.exists() {
        eprintln!("[skip] no dumps");
        return;
    }
    let d = Dumps::open(&g, &m).unwrap();
    let raw = d.reference("raw_image").unwrap(); // (224,224,3) HWC, values 0..255
    let expected = d.reference("input_image").unwrap(); // (3,H,W) NCHW, normalized

    // shape[..] is (H, W, C); derive H/W from the dump rather than hardcoding
    // 224, in case the fixture ever changes.
    let (h, w) = match raw.shape.as_slice() {
        [hh, ww, 3] => (*hh as usize, *ww as usize),
        [1, hh, ww, 3] => (*hh as usize, *ww as usize),
        other => panic!("unexpected raw_image shape: {other:?}"),
    };
    let raw_u8: Vec<u8> = raw.data.iter().map(|&v| v.round().clamp(0.0, 255.0) as u8).collect();

    // ModelConfig fields relevant to preprocess; image_size and mean/std are
    // read from the same dump's manifest-adjacent metadata where available,
    // but since Dumps only exposes tensors here, derive image_size from the
    // expected output shape and use standard DA3 ImageNet mean/std as the
    // config (matches `depthanything3.img.mean`/`std` defaults used
    // elsewhere in this workspace, e.g. crates/da-engine/src/config.rs
    // tests).
    let out_side = match expected.shape.as_slice() {
        [3, hh, ww] => {
            assert_eq!(hh, ww, "expected square output image_size");
            *hh as u32
        }
        other => panic!("unexpected input_image shape: {other:?}"),
    };
    let cfg = ModelConfig {
        arch: "depthanything3".to_string(),
        patch_size: 14,
        image_size: out_side,
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
        img_resize_mode: "bilinear".to_string(),
        alt_start: -1,
        cat_token: true,
        cam_dim_in: 8,
    };

    let mut out = Vec::new();
    preprocess(&raw_u8, h, w, &cfg, &mut out);
    assert_parity(&out, &expected.data, d.atol(), d.rtol(), "preprocess");
}
