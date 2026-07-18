use da_engine::ModelConfig;
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
fn reads_config_from_real_model() {
    let Some(g) = model() else { return };
    let cfg = ModelConfig::from_gguf(&g).expect("valid depthanything3 model should parse");

    assert_eq!(cfg.arch, "depthanything3");
    assert!(cfg.embed_dim >= 384, "embed_dim plausible: {}", cfg.embed_dim);
    assert!(cfg.depth >= 12, "depth plausible: {}", cfg.depth);
    assert!(cfg.num_heads > 0);
    assert!(cfg.head_dim > 0);
    assert!(!cfg.out_layers.is_empty());
    assert!(cfg.head_max_depth > 0.0);
}
