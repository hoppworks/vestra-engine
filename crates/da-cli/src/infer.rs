//! Core logic behind `da infer`: load a model + image via `da_engine::Engine`,
//! then write the resulting depth map (PFM or PNG) and camera pose (JSON) to
//! disk.
//!
//! Split out of `main.rs` so the individual pieces (PFM writer, pose-JSON
//! reshaping) are unit-testable without a real GGUF model or PNG file — see
//! this module's `#[cfg(test)]` block.

use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use da_engine::{Engine, InferOut, QuantPref};
use serde_json::json;

/// `da infer` subcommand arguments (kept as a plain struct, separate from
/// the `clap`-derived CLI type in `main.rs`, so this module has no `clap`
/// dependency of its own — `main.rs` maps its parsed `InferArgs` into this
/// on the call site).
pub struct InferRequest {
    pub model: PathBuf,
    pub image: PathBuf,
    pub out_depth: PathBuf,
    pub out_pose: PathBuf,
}

/// Runs the full `da infer` pipeline: decode `req.image` to raw HWC `u8`
/// RGB, load `req.model` via [`Engine::load`], run [`Engine::infer`], then
/// write the depth map to `req.out_depth` (format chosen by file extension —
/// `.pfm` or `.png`) and the camera pose to `req.out_pose` as JSON.
pub fn run_infer(req: &InferRequest) -> Result<(), Box<dyn Error>> {
    let img = image::open(&req.image)?.to_rgb8();
    let w = img.width() as usize;
    let h = img.height() as usize;
    let raw_hwc_u8 = img.into_raw();

    let mut engine = Engine::load(&req.model, QuantPref::PreferF32)?;
    let out = engine.infer(&raw_hwc_u8, h, w)?;

    write_depth(&req.out_depth, &out)?;
    write_pose_json(&req.out_pose, &out.extrinsics, &out.intrinsics)?;

    Ok(())
}

/// Dispatches on `path`'s extension: `.pfm` -> [`write_pfm`], `.png` (or
/// anything else) -> [`write_depth_png`]. Extension matching is
/// case-insensitive.
fn write_depth(path: &Path, out: &InferOut) -> Result<(), Box<dyn Error>> {
    let is_pfm = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("pfm"))
        .unwrap_or(false);
    if is_pfm {
        write_pfm(path, &out.depth, out.w, out.h)?;
    } else {
        write_depth_png(path, &out.depth, out.w, out.h)?;
    }
    Ok(())
}

/// Writes `depth` (row-major, top-to-bottom, `h*w` elements) as a PFM
/// ("Portable Float Map") grayscale file: header `"Pf\n{w} {h}\n{scale}\n"`
/// (`Pf` = single-channel; `scale` negative = little-endian, per the PFM
/// spec), followed by raw native-endian `f32` scanlines in PFM's canonical
/// bottom-to-top row order (row 0 of the file is the LAST row of `depth`).
pub fn write_pfm(path: &Path, depth: &[f32], w: usize, h: usize) -> std::io::Result<()> {
    assert_eq!(depth.len(), w * h, "depth buffer must be exactly w*h elements");
    let mut f = BufWriter::new(File::create(path)?);
    write!(f, "Pf\n{w} {h}\n-1.0\n")?;
    for row in (0..h).rev() {
        let start = row * w;
        let row_slice = &depth[start..start + w];
        for &v in row_slice {
            f.write_all(&v.to_le_bytes())?;
        }
    }
    f.flush()?;
    Ok(())
}

/// Writes `depth` as an 8-bit grayscale PNG, min-max normalized to `0..=255`
/// (the min depth value maps to 0, the max to 255). A fixed physical depth
/// range (e.g. `cfg.head_max_depth`) would be more directly comparable
/// across images, but `Engine` doesn't expose its loaded `ModelConfig`
/// through its public API (`cfg` is a private field — see
/// `da-engine/src/engine.rs`), so min-max normalization per-image is the
/// only option available from this crate without widening that API. This is
/// a visualization aid, not a precision-preserving format — use `.pfm` for
/// exact float data.
pub fn write_depth_png(path: &Path, depth: &[f32], w: usize, h: usize) -> Result<(), Box<dyn Error>> {
    assert_eq!(depth.len(), w * h, "depth buffer must be exactly w*h elements");
    let (min, max) = depth.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), &v| (mn.min(v), mx.max(v)));
    let range = if (max - min).abs() > f32::EPSILON { max - min } else { 1.0 };
    let pixels: Vec<u8> = depth.iter().map(|&v| (((v - min) / range) * 255.0).round().clamp(0.0, 255.0) as u8).collect();
    let img = image::GrayImage::from_raw(w as u32, h as u32, pixels).ok_or("depth buffer size mismatch building GrayImage")?;
    img.save(path)?;
    Ok(())
}

/// Reshapes the flat row-major `extrinsics`/`intrinsics` arrays into nested
/// `[[..]]` JSON arrays (extrinsics: 3 rows x 4 cols; intrinsics: 3 rows x 3
/// cols) and writes `{ "extrinsics": [...], "intrinsics": [...] }` to
/// `path`.
pub fn write_pose_json(path: &Path, extrinsics: &[f32; 12], intrinsics: &[f32; 9]) -> Result<(), Box<dyn Error>> {
    let value = pose_json_value(extrinsics, intrinsics);
    let mut f = BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(&mut f, &value)?;
    f.flush()?;
    Ok(())
}

/// Builds the `serde_json::Value` for the pose JSON: a flat row-major
/// `[f32; 12]` extrinsics array reshaped into 3 rows of 4, and a flat
/// row-major `[f32; 9]` intrinsics array reshaped into 3 rows of 3.
pub fn pose_json_value(extrinsics: &[f32; 12], intrinsics: &[f32; 9]) -> serde_json::Value {
    let ext_rows: Vec<Vec<f32>> = extrinsics.chunks(4).map(|c| c.to_vec()).collect();
    let int_rows: Vec<Vec<f32>> = intrinsics.chunks(3).map(|c| c.to_vec()).collect();
    json!({
        "extrinsics": ext_rows,
        "intrinsics": int_rows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Unique temp-file path helper, same pattern used elsewhere in this
    /// workspace (`da-engine/tests/e2e_native.rs`'s `write_temp_gguf`):
    /// atomic counter + PID + nanos, safe under parallel test execution.
    fn temp_path(suffix: &str) -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let counter = COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        std::env::temp_dir().join(format!("da_cli_test_{pid}_{nanos}_{counter}{suffix}"))
    }

    // ---------------------------------------------------------------
    // PFM writer: write then read back, verify header + byte-exact data.
    // ---------------------------------------------------------------

    #[test]
    fn pfm_header_and_data_roundtrip() {
        let w = 3;
        let h = 2;
        // row 0 (top): 1,2,3 ; row 1 (bottom): 4,5,6
        let depth = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let path = temp_path(".pfm");

        write_pfm(&path, &depth, w, h).expect("write_pfm should succeed");

        let mut bytes = Vec::new();
        File::open(&path).unwrap().read_to_end(&mut bytes).unwrap();

        let header = "Pf\n3 2\n-1.0\n";
        assert!(bytes.starts_with(header.as_bytes()), "PFM header mismatch: {:?}", String::from_utf8_lossy(&bytes[..header.len().min(bytes.len())]));

        let data = &bytes[header.len()..];
        assert_eq!(data.len(), w * h * 4, "PFM data section should be w*h f32s");

        // PFM row order is bottom-to-top: file row 0 = image row (h-1) = [4,5,6],
        // file row 1 = image row 0 = [1,2,3].
        let mut floats = Vec::new();
        for chunk in data.chunks_exact(4) {
            floats.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }
        assert_eq!(floats, vec![4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn pfm_single_row_is_unchanged_by_bottom_to_top_reorder() {
        let w = 4;
        let h = 1;
        let depth = vec![10.0f32, 20.0, 30.0, 40.0];
        let path = temp_path(".pfm");
        write_pfm(&path, &depth, w, h).unwrap();

        let mut bytes = Vec::new();
        File::open(&path).unwrap().read_to_end(&mut bytes).unwrap();
        let header = "Pf\n4 1\n-1.0\n";
        let data = &bytes[header.len()..];
        let floats: Vec<f32> = data.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        assert_eq!(floats, depth);

        let _ = std::fs::remove_file(&path);
    }

    // ---------------------------------------------------------------
    // Pose-JSON reshaping: flat array -> nested JSON, round-trip via serde_json.
    // ---------------------------------------------------------------

    #[test]
    fn pose_json_value_reshapes_flat_arrays_into_nested_rows() {
        let extrinsics: [f32; 12] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let intrinsics: [f32; 9] = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0];

        let value = pose_json_value(&extrinsics, &intrinsics);
        let s = serde_json::to_string(&value).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&s).unwrap();

        let ext = parsed["extrinsics"].as_array().expect("extrinsics should be an array");
        assert_eq!(ext.len(), 3, "extrinsics should reshape into 3 rows");
        for row in ext {
            assert_eq!(row.as_array().unwrap().len(), 4, "each extrinsics row should have 4 cols");
        }
        assert_eq!(ext[0][0].as_f64().unwrap(), 0.0);
        assert_eq!(ext[2][3].as_f64().unwrap(), 11.0);

        let int_ = parsed["intrinsics"].as_array().expect("intrinsics should be an array");
        assert_eq!(int_.len(), 3, "intrinsics should reshape into 3 rows");
        for row in int_ {
            assert_eq!(row.as_array().unwrap().len(), 3, "each intrinsics row should have 3 cols");
        }
        assert_eq!(int_[0][0].as_f64().unwrap(), 100.0);
        assert_eq!(int_[2][2].as_f64().unwrap(), 108.0);
    }

    #[test]
    fn write_pose_json_writes_valid_file_matching_pose_json_value() {
        let extrinsics: [f32; 12] = [1.0; 12];
        let intrinsics: [f32; 9] = [2.0; 9];
        let path = temp_path(".json");

        write_pose_json(&path, &extrinsics, &intrinsics).expect("write_pose_json should succeed");

        let contents = std::fs::read_to_string(&path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&contents).expect("output should be valid JSON");
        let expected = pose_json_value(&extrinsics, &intrinsics);
        assert_eq!(parsed, expected);

        let _ = std::fs::remove_file(&path);
    }

    // ---------------------------------------------------------------
    // Depth PNG writer: sanity-check it actually produces a decodable image
    // of the right size, with min-max normalization landing at 0/255.
    // ---------------------------------------------------------------

    #[test]
    fn write_depth_png_produces_normalized_grayscale_image() {
        let w = 2;
        let h = 2;
        let depth = vec![0.0f32, 5.0, 10.0, 2.5];
        let path = temp_path(".png");

        write_depth_png(&path, &depth, w, h).expect("write_depth_png should succeed");

        let img = image::open(&path).expect("output should be a decodable image").to_luma8();
        assert_eq!(img.width() as usize, w);
        assert_eq!(img.height() as usize, h);
        // min (0.0, at x=0,y=0) -> 0; max (10.0, at x=0,y=1) -> 255
        assert_eq!(img.get_pixel(0, 0).0[0], 0);
        assert_eq!(img.get_pixel(0, 1).0[0], 255);

        let _ = std::fs::remove_file(&path);
    }
}
