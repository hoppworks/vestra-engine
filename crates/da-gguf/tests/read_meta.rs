use da_gguf::GgufFile;
use std::path::Path;

fn model() -> Option<GgufFile> {
    // Ein echtes DA3-BASE-Modell wird via ../scripts/download_model.py bereitgestellt.
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../models/da3-base-f16.gguf");
    if !p.exists() {
        eprintln!("[skip] no model at {}", p.display());
        return None;
    }
    Some(GgufFile::open(&p).unwrap())
}

#[test]
fn reads_arch_and_vit_dims() {
    let Some(m) = model() else { return };
    assert_eq!(
        m.meta_str("depthanything3.arch").as_deref(),
        Some("depthanything3")
    );
    assert!(m.meta_u32("depthanything3.vit.embed_dim").unwrap() >= 384);
    assert!(m.meta_u32("depthanything3.vit.depth").unwrap() >= 12);
}
