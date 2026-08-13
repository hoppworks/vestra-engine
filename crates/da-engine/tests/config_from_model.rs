use da_engine::ModelConfig;
use da_gguf::GgufFile;
use std::path::{Path, PathBuf};

fn model() -> Option<GgufFile> {
    // The benchmark host supplies the exact pinned model through
    // DA_TEST_MODEL. Keep the legacy local path for developer convenience.
    let p = std::env::var_os("DA_TEST_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../models/da3-base-f16.gguf")
        });
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
    assert!(
        cfg.embed_dim >= 384,
        "embed_dim plausible: {}",
        cfg.embed_dim
    );
    assert!(cfg.depth >= 12, "depth plausible: {}", cfg.depth);
    assert!(cfg.num_heads > 0);
    assert!(cfg.head_dim > 0);
    assert!(!cfg.out_layers.is_empty());
    assert_eq!(cfg.image_size, 504);
    assert!(cfg.head_max_depth >= 0.0);
}
