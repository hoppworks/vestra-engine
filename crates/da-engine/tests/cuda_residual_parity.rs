#![cfg(feature = "cuda-residual-oracle")]

use std::path::PathBuf;

use vestra_engine::{Engine, QuantPref, ViewInput};

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

/// Verifies the complete cached FC1 → GELU → FC2 CUDA branch in every
/// transformer block. This intentionally leaves LayerNorm and residuals on
/// CPU so a numerical failure has one clear owner.
#[test]
fn cuda_mlp_slice_matches_cpu_depth_and_confidence() {
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
    cuda.enable_cuda_mlp_oracle(0)
        .expect("CUDA MLP runtime must initialize and cache parameters");
    assert!(cuda.cuda_mlp_oracle_enabled());
    let actual = cuda
        .infer(&rgb, height, width)
        .expect("CUDA MLP inference must succeed");

    assert_eq!((actual.w, actual.h), (expected.w, expected.h));
    assert_close("MLP depth", &expected.depth, &actual.depth);
    assert_close("MLP confidence", &expected.conf, &actual.conf);
}

/// Exercises the PR #2-style ordered multi-view path. Set
/// `VESTRA_CUDA_IMAGES` to two or more colon-separated image paths; without
/// it the configured single image is duplicated solely to keep this fixture
/// runnable on a minimal workhorse installation.
#[test]
fn cuda_residual_slice_matches_cpu_ordered_multiview() {
    let (Some(model), Some(image)) = (
        std::env::var_os("VESTRA_CUDA_MODEL"),
        std::env::var_os("VESTRA_CUDA_IMAGE"),
    ) else {
        return;
    };
    let paths = std::env::var("VESTRA_CUDA_IMAGES")
        .ok()
        .map(|value| value.split(':').map(PathBuf::from).collect::<Vec<_>>())
        .filter(|paths| paths.len() >= 2)
        .unwrap_or_else(|| vec![PathBuf::from(&image), PathBuf::from(&image)]);
    let images = paths
        .iter()
        .map(|path| {
            let decoded = image::open(path)
                .expect("configured CUDA parity image must decode")
                .to_rgb8();
            (
                decoded.width() as usize,
                decoded.height() as usize,
                decoded.into_raw(),
            )
        })
        .collect::<Vec<_>>();
    let inputs = images
        .iter()
        .map(|(width, height, rgb)| ViewInput {
            rgb_hwc_u8: rgb,
            h: *height,
            w: *width,
        })
        .collect::<Vec<_>>();

    let mut cpu = Engine::load(PathBuf::from(&model).as_path(), QuantPref::PreferF32)
        .expect("CPU Engine must load the configured F32 model");
    let expected = cpu
        .infer_multi_view_ordered(&inputs)
        .expect("CPU ordered multiview inference must succeed");

    let mut cuda = Engine::load(PathBuf::from(model).as_path(), QuantPref::PreferF32)
        .expect("CUDA parity Engine must load the configured F32 model");
    cuda.enable_cuda_residual_oracle(0)
        .expect("CUDA residual runtime must initialize");
    let actual = cuda
        .infer_multi_view_ordered(&inputs)
        .expect("CUDA ordered multiview inference must succeed");

    assert_eq!(actual.reference_view_index, expected.reference_view_index);
    assert_eq!(actual.views.len(), expected.views.len());
    for (index, (expected, actual)) in expected.views.iter().zip(&actual.views).enumerate() {
        assert_eq!(
            (actual.w, actual.h),
            (expected.w, expected.h),
            "view {index}"
        );
        assert_close(
            &format!("view {index} depth"),
            &expected.depth,
            &actual.depth,
        );
        assert_close(
            &format!("view {index} confidence"),
            &expected.conf,
            &actual.conf,
        );
    }
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
