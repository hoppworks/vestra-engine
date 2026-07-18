//! Smoke test for the `da infer` binary (Task 21's Step 1): on a small
//! image, `da infer` should run and write a non-empty depth file + valid
//! pose JSON. Model-gated: SKIPS (does not fail) when no real GGUF model is
//! present, matching every other model-gated test in this workspace (see
//! `da-engine/tests/e2e_native.rs`'s `model_path`/`[skip]` pattern) — there
//! is no `../models/*.gguf` in this environment.

use std::path::{Path, PathBuf};

fn model_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../models/da3-base-f16.gguf")
}

#[test]
fn infer_writes_nonempty_depth_and_valid_pose_json() {
    let model = model_path();
    if !model.exists() {
        eprintln!("[skip] no model at {}", model.display());
        return;
    }

    let tmp_dir = std::env::temp_dir();
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();

    // Build a tiny synthetic 4x4 RGB PNG as the input image.
    let image_path = tmp_dir.join(format!("da_cli_smoke_in_{pid}_{nanos}.png"));
    let mut img = image::RgbImage::new(4, 4);
    for y in 0..4 {
        for x in 0..4 {
            img.put_pixel(x, y, image::Rgb([(x * 40) as u8, (y * 40) as u8, 128]));
        }
    }
    img.save(&image_path).expect("failed to write synthetic input PNG");

    let out_depth = tmp_dir.join(format!("da_cli_smoke_depth_{pid}_{nanos}.pfm"));
    let out_pose = tmp_dir.join(format!("da_cli_smoke_pose_{pid}_{nanos}.json"));

    let mut cmd = assert_cmd::Command::cargo_bin("da").expect("da binary should build");
    cmd.arg("infer")
        .arg("--model")
        .arg(&model)
        .arg("--image")
        .arg(&image_path)
        .arg("--out-depth")
        .arg(&out_depth)
        .arg("--out-pose")
        .arg(&out_pose);
    cmd.assert().success();

    let depth_bytes = std::fs::read(&out_depth).expect("depth output file should exist");
    assert!(!depth_bytes.is_empty(), "depth output file should be non-empty");

    let pose_contents = std::fs::read_to_string(&out_pose).expect("pose output file should exist");
    let pose_json: serde_json::Value = serde_json::from_str(&pose_contents).expect("pose output should be valid JSON");
    assert!(pose_json["extrinsics"].is_array());
    assert!(pose_json["intrinsics"].is_array());

    let _ = std::fs::remove_file(&image_path);
    let _ = std::fs::remove_file(&out_depth);
    let _ = std::fs::remove_file(&out_pose);
}
