use da_gguf::GgufFile;
use std::path::Path;

fn dumps() -> std::path::PathBuf {
    // Tests laufen im Crate-Verzeichnis; ../../../dumps relativ dazu ist der C++-Repo-Root/dumps.
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../dumps/reference.gguf")
}

#[test]
fn reads_input_image_shape() {
    let p = dumps();
    if !p.exists() {
        eprintln!("[skip] no reference dumps at {}", p.display());
        return;
    }
    let f = GgufFile::open(&p).expect("open reference.gguf");
    let t = f.tensor_f32("input_image").expect("input_image tensor");
    // dump_reference.py: input_image ist das (1,3,H,W) DA3-BASE-Eingabebild, H=W=224.
    let n: i64 = t.shape.iter().product();
    assert_eq!(n as usize, t.data.len());
    assert!(t.data.iter().all(|v| v.is_finite()));
    assert!(
        t.shape.contains(&224),
        "expected a 224 dim, got {:?}",
        t.shape
    );
}
