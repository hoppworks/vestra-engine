#![cfg(feature = "cuda-residual-oracle")]

use std::path::PathBuf;

use vestra_engine::{Engine, QuantPref};

/// This deliberately opt-in integration test proves that the Engine-owned
/// CUDA residual slice participates in an actual DA3 forward pass. It is
/// skipped unless a real F32 GGUF and image are supplied because those large
/// fixtures do not belong in the source repository.
#[test]
fn cuda_residual_slice_matches_cpu_depth_and_confidence() {
    let (Some(model), Some(image)) = (
        std::env::var_os("VESTRA_CUDA_MODEL"),
        std::env::var_os("VESTRA_CUDA_IMAGE"),
    ) else {
        return;
    };
    let image = image::open(PathBuf::from(image))
        .expect("configured CUDA parity image must decode")
        .to_rgb8();
    let (width, height) = (image.width() as usize, image.height() as usize);
    let rgb = image.into_raw();

    let mut cpu = Engine::load(PathBuf::from(&model).as_path(), QuantPref::PreferF32)
        .expect("CPU Engine must load the configured F32 model");
    let expected = cpu
        .infer(&rgb, height, width)
        .expect("CPU inference must succeed");

    let mut cuda = Engine::load(PathBuf::from(model).as_path(), QuantPref::PreferF32)
        .expect("CUDA parity Engine must load the configured F32 model");
    cuda.enable_cuda_residual_oracle(0)
        .expect("CUDA residual runtime must initialize");
    assert!(cuda.cuda_residual_oracle_enabled());
    let actual = cuda
        .infer(&rgb, height, width)
        .expect("CUDA residual inference must succeed");

    assert_eq!((actual.w, actual.h), (expected.w, expected.h));
    assert_close("depth", &expected.depth, &actual.depth);
    assert_close("confidence", &expected.conf, &actual.conf);
}

fn assert_close(label: &str, expected: &[f32], actual: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    let mae = expected
        .iter()
        .zip(actual)
        .map(|(left, right)| (left - right).abs())
        .sum::<f32>()
        / expected.len() as f32;
    let max = expected
        .iter()
        .zip(actual)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max);
    assert!(mae <= 1e-6, "{label} MAE {mae} exceeds 1e-6");
    assert!(max <= 1e-5, "{label} max error {max} exceeds 1e-5");
}
