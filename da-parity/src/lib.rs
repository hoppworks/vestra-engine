use da_gguf::{GgufFile, TensorF32};
use std::path::{Path, PathBuf};

#[derive(thiserror::Error, Debug)]
pub enum ParityError {
    #[error("gguf: {0}")]
    Gguf(#[from] da_gguf::GgufError),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("manifest: {0}")]
    Manifest(String),
}

pub struct Dumps {
    gguf: GgufFile,
    atol: f32,
    rtol: f32,
}

#[derive(serde::Deserialize)]
struct Manifest {
    #[serde(default = "d_atol")]
    atol: f32,
    #[serde(default = "d_rtol")]
    rtol: f32,
}
fn d_atol() -> f32 {
    2e-3
}
fn d_rtol() -> f32 {
    2e-3
}

impl Dumps {
    pub fn open(gguf: &Path, manifest: &Path) -> Result<Dumps, ParityError> {
        let g = GgufFile::open(gguf)?;
        let m: Manifest = serde_json::from_slice(&std::fs::read(manifest)?)
            .map_err(|e| ParityError::Manifest(e.to_string()))?;
        Ok(Dumps {
            gguf: g,
            atol: m.atol,
            rtol: m.rtol,
        })
    }
    pub fn reference(&self, name: &str) -> Result<TensorF32, ParityError> {
        Ok(self.gguf.tensor_f32(name)?)
    }
    pub fn atol(&self) -> f32 {
        self.atol
    }
    pub fn rtol(&self) -> f32 {
        self.rtol
    }
}

pub struct CompareReport {
    pub ok: bool,
    pub max_abs: f64,
    pub mean_abs: f64,
    pub worst: usize,
    pub n: usize,
}

pub fn compare(got: &[f32], reference: &[f32], atol: f32, rtol: f32, label: &str) -> CompareReport {
    if got.len() != reference.len() {
        eprintln!(
            "[{label}] size mismatch got={} ref={}",
            got.len(),
            reference.len()
        );
        return CompareReport {
            ok: false,
            max_abs: f64::INFINITY,
            mean_abs: f64::INFINITY,
            worst: 0,
            n: 0,
        };
    }
    if got.is_empty() {
        eprintln!("[{label}] empty vectors -> FAIL");
        return CompareReport {
            ok: false,
            max_abs: 0.0,
            mean_abs: 0.0,
            worst: 0,
            n: 0,
        };
    }
    let (mut max_abs, mut sum, mut worst) = (0.0f64, 0.0f64, 0usize);
    for i in 0..got.len() {
        let d = (got[i] as f64 - reference[i] as f64).abs();
        sum += d;
        if d > max_abs {
            max_abs = d;
            worst = i;
        }
    }
    let mean = sum / got.len() as f64;
    let mut ok = true;
    for i in 0..got.len() {
        let tol = atol as f64 + rtol as f64 * (reference[i] as f64).abs();
        if (got[i] as f64 - reference[i] as f64).abs() > tol {
            ok = false;
            break;
        }
    }
    eprintln!(
        "[{label}] n={} max|d|={:.3e} mean|d|={:.3e} (worst@{} got={:.5} ref={:.5}) -> {}",
        got.len(),
        max_abs,
        mean,
        worst,
        got[worst],
        reference[worst],
        if ok { "OK" } else { "FAIL" }
    );
    CompareReport {
        ok,
        max_abs,
        mean_abs: mean,
        worst,
        n: got.len(),
    }
}

pub fn assert_parity(got: &[f32], reference: &[f32], atol: f32, rtol: f32, label: &str) {
    let r = compare(got, reference, atol, rtol, label);
    assert!(r.ok, "[{label}] parity FAIL max|d|={:.3e}", r.max_abs);
}

pub fn dumps_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../dumps")
        .join(rel)
}
