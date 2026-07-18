use da_gguf::GgufFile;

/// GGUF metadata key names for the `depthanything3` architecture.
/// Verbatim from `include/da_gguf_keys.h` — do not rename.
mod keys {
    pub const ARCH: &str = "depthanything3.arch";
    pub const PATCH_SIZE: &str = "depthanything3.patch_size";
    pub const IMAGE_SIZE: &str = "depthanything3.image_size";
    pub const VIT_EMBED_DIM: &str = "depthanything3.vit.embed_dim";
    pub const VIT_DEPTH: &str = "depthanything3.vit.depth";
    pub const VIT_NUM_HEADS: &str = "depthanything3.vit.num_heads";
    pub const VIT_HEAD_DIM: &str = "depthanything3.vit.head_dim";
    pub const VIT_MLP_HIDDEN: &str = "depthanything3.vit.mlp_hidden";
    pub const VIT_NUM_REGISTER: &str = "depthanything3.vit.num_register_tokens";
    pub const VIT_ROPE_START: &str = "depthanything3.vit.rope_start";
    pub const VIT_QKNORM_START: &str = "depthanything3.vit.qknorm_start";
    pub const VIT_ROPE_FREQ: &str = "depthanything3.vit.rope_freq";
    pub const VIT_LN_EPS: &str = "depthanything3.vit.ln_eps";
    pub const VIT_OUT_LAYERS: &str = "depthanything3.vit.out_layers";
    pub const IMG_MEAN: &str = "depthanything3.img.mean";
    pub const IMG_STD: &str = "depthanything3.img.std";
    pub const IMG_RESIZE_MODE: &str = "depthanything3.img.resize_mode";
    pub const HEAD_FEATURES: &str = "depthanything3.head.features";
    pub const HEAD_MAX_DEPTH: &str = "depthanything3.head.max_depth";
    pub const CAM_DIM_IN: &str = "depthanything3.cam.dim_in";
}

const EXPECTED_ARCH: &str = "depthanything3";

#[derive(thiserror::Error, Debug)]
pub enum EngineError {
    #[error("unsupported model architecture: expected '{EXPECTED_ARCH}', found {0:?}")]
    UnsupportedModel(Option<String>),
    #[error("missing required gguf metadata key: {0}")]
    MissingKey(&'static str),
    #[error("malformed gguf metadata for key {0}: {1}")]
    Malformed(&'static str, String),
}

/// Model hyperparameters and preprocessing config read from `depthanything3.*` GGUF metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    pub arch: String,
    pub patch_size: u32,
    pub image_size: u32,
    pub embed_dim: u32,
    pub depth: u32,
    pub num_heads: u32,
    pub head_dim: u32,
    pub mlp_hidden: u32,
    pub num_register: u32,
    pub rope_start: i32,
    pub qknorm_start: i32,
    pub rope_freq: f32,
    pub ln_eps: f32,
    pub out_layers: Vec<i32>,
    pub head_features: u32,
    pub head_max_depth: f32,
    pub img_mean: [f32; 3],
    pub img_std: [f32; 3],
    pub img_resize_mode: String,
    pub cam_dim_in: u32,
}

fn req_u32(f: &GgufFile, key: &'static str) -> Result<u32, EngineError> {
    f.meta_u32(key).ok_or(EngineError::MissingKey(key))
}

fn req_i32(f: &GgufFile, key: &'static str) -> Result<i32, EngineError> {
    f.meta_i32(key).ok_or(EngineError::MissingKey(key))
}

fn req_f32(f: &GgufFile, key: &'static str) -> Result<f32, EngineError> {
    f.meta_f32(key).ok_or(EngineError::MissingKey(key))
}

fn req_str(f: &GgufFile, key: &'static str) -> Result<String, EngineError> {
    f.meta_str(key).ok_or(EngineError::MissingKey(key))
}

fn req_arr_i32(f: &GgufFile, key: &'static str) -> Result<Vec<i32>, EngineError> {
    f.meta_arr_i32(key).ok_or(EngineError::MissingKey(key))
}

fn req_vec3(f: &GgufFile, key: &'static str) -> Result<[f32; 3], EngineError> {
    let v = f.meta_arr_f32(key).ok_or(EngineError::MissingKey(key))?;
    v.as_slice()
        .try_into()
        .map_err(|_| EngineError::Malformed(key, format!("expected 3 elements, got {}", v.len())))
}

impl ModelConfig {
    /// Reads `depthanything3.*` metadata from a GGUF file into a `ModelConfig`.
    ///
    /// Returns `Err(EngineError::UnsupportedModel)` if the `depthanything3.arch` key is
    /// missing or does not equal `"depthanything3"`.
    pub fn from_gguf(f: &GgufFile) -> Result<ModelConfig, EngineError> {
        let arch = f.meta_str(keys::ARCH);
        if arch.as_deref() != Some(EXPECTED_ARCH) {
            return Err(EngineError::UnsupportedModel(arch));
        }
        let arch = arch.unwrap();

        Ok(ModelConfig {
            arch,
            patch_size: req_u32(f, keys::PATCH_SIZE)?,
            image_size: req_u32(f, keys::IMAGE_SIZE)?,
            embed_dim: req_u32(f, keys::VIT_EMBED_DIM)?,
            depth: req_u32(f, keys::VIT_DEPTH)?,
            num_heads: req_u32(f, keys::VIT_NUM_HEADS)?,
            head_dim: req_u32(f, keys::VIT_HEAD_DIM)?,
            mlp_hidden: req_u32(f, keys::VIT_MLP_HIDDEN)?,
            num_register: req_u32(f, keys::VIT_NUM_REGISTER)?,
            rope_start: req_i32(f, keys::VIT_ROPE_START)?,
            qknorm_start: req_i32(f, keys::VIT_QKNORM_START)?,
            rope_freq: req_f32(f, keys::VIT_ROPE_FREQ)?,
            ln_eps: req_f32(f, keys::VIT_LN_EPS)?,
            out_layers: req_arr_i32(f, keys::VIT_OUT_LAYERS)?,
            head_features: req_u32(f, keys::HEAD_FEATURES)?,
            head_max_depth: req_f32(f, keys::HEAD_MAX_DEPTH)?,
            img_mean: req_vec3(f, keys::IMG_MEAN)?,
            img_std: req_vec3(f, keys::IMG_STD)?,
            img_resize_mode: req_str(f, keys::IMG_RESIZE_MODE)?,
            cam_dim_in: req_u32(f, keys::CAM_DIM_IN)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use da_gguf::GgufFile;
    use std::io::Write;

    /// Builds a minimal in-memory GGUF byte buffer with the given KV entries and no tensors,
    /// writes it to a temp file, and opens it as a `GgufFile`.
    ///
    /// KV entries are `(key, value)` where value is one of the small helper variants below.
    enum Kv<'a> {
        Str(&'a str, &'a str),
        U32(&'a str, u32),
        I32(&'a str, i32),
        F32(&'a str, f32),
        ArrF32(&'a str, &'a [f32]),
        ArrI32(&'a str, &'a [i32]),
    }

    fn write_kv(buf: &mut Vec<u8>, key: &str, vtype: u32) {
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key.as_bytes());
        buf.extend_from_slice(&vtype.to_le_bytes());
    }

    fn build_gguf(entries: &[Kv]) -> GgufFile {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&2u32.to_le_bytes()); // version
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&(entries.len() as u64).to_le_bytes()); // kv_count

        for e in entries {
            match e {
                Kv::Str(k, v) => {
                    write_kv(&mut buf, k, 8);
                    buf.extend_from_slice(&(v.len() as u64).to_le_bytes());
                    buf.extend_from_slice(v.as_bytes());
                }
                Kv::U32(k, v) => {
                    write_kv(&mut buf, k, 4);
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                Kv::I32(k, v) => {
                    write_kv(&mut buf, k, 5);
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                Kv::F32(k, v) => {
                    write_kv(&mut buf, k, 6);
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                Kv::ArrF32(k, v) => {
                    write_kv(&mut buf, k, 9);
                    buf.extend_from_slice(&6u32.to_le_bytes()); // elem type: float32
                    buf.extend_from_slice(&(v.len() as u64).to_le_bytes());
                    for x in *v {
                        buf.extend_from_slice(&x.to_le_bytes());
                    }
                }
                Kv::ArrI32(k, v) => {
                    write_kv(&mut buf, k, 9);
                    buf.extend_from_slice(&5u32.to_le_bytes()); // elem type: int32
                    buf.extend_from_slice(&(v.len() as u64).to_le_bytes());
                    for x in *v {
                        buf.extend_from_slice(&x.to_le_bytes());
                    }
                }
            }
        }

        // Pad to alignment (32, the default) before the (empty) tensor data block.
        let pad = (32 - (buf.len() % 32)) % 32;
        buf.extend_from_slice(&vec![0u8; pad]);

        let path = std::env::temp_dir().join(format!(
            "da_engine_test_{}.gguf",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut file = std::fs::File::create(&path).expect("create temp gguf");
        file.write_all(&buf).expect("write temp gguf");
        drop(file);
        let g = GgufFile::open(&path).expect("open temp gguf");
        let _ = std::fs::remove_file(&path);
        g
    }

    fn full_valid_entries() -> Vec<Kv<'static>> {
        vec![
            Kv::Str("depthanything3.arch", "depthanything3"),
            Kv::U32("depthanything3.patch_size", 14),
            Kv::U32("depthanything3.image_size", 518),
            Kv::U32("depthanything3.vit.embed_dim", 384),
            Kv::U32("depthanything3.vit.depth", 12),
            Kv::U32("depthanything3.vit.num_heads", 6),
            Kv::U32("depthanything3.vit.head_dim", 64),
            Kv::U32("depthanything3.vit.mlp_hidden", 1536),
            Kv::U32("depthanything3.vit.num_register_tokens", 4),
            Kv::I32("depthanything3.vit.rope_start", 0),
            Kv::I32("depthanything3.vit.qknorm_start", 0),
            Kv::F32("depthanything3.vit.rope_freq", 100.0),
            Kv::F32("depthanything3.vit.ln_eps", 1e-6),
            Kv::ArrI32("depthanything3.vit.out_layers", &[2, 5, 8, 11]),
            Kv::U32("depthanything3.head.features", 256),
            Kv::F32("depthanything3.head.max_depth", 20.0),
            Kv::ArrF32("depthanything3.img.mean", &[0.485, 0.456, 0.406]),
            Kv::ArrF32("depthanything3.img.std", &[0.229, 0.224, 0.225]),
            Kv::Str("depthanything3.img.resize_mode", "bicubic"),
            Kv::U32("depthanything3.cam.dim_in", 8),
        ]
    }

    #[test]
    fn from_gguf_parses_all_fields() {
        let g = build_gguf(&full_valid_entries());
        let cfg = ModelConfig::from_gguf(&g).expect("should parse valid config");
        assert_eq!(cfg.arch, "depthanything3");
        assert_eq!(cfg.patch_size, 14);
        assert_eq!(cfg.image_size, 518);
        assert_eq!(cfg.embed_dim, 384);
        assert_eq!(cfg.depth, 12);
        assert_eq!(cfg.num_heads, 6);
        assert_eq!(cfg.head_dim, 64);
        assert_eq!(cfg.mlp_hidden, 1536);
        assert_eq!(cfg.num_register, 4);
        assert_eq!(cfg.rope_start, 0);
        assert_eq!(cfg.qknorm_start, 0);
        assert_eq!(cfg.rope_freq, 100.0);
        assert_eq!(cfg.ln_eps, 1e-6);
        assert_eq!(cfg.out_layers, vec![2, 5, 8, 11]);
        assert_eq!(cfg.head_features, 256);
        assert_eq!(cfg.head_max_depth, 20.0);
        assert_eq!(cfg.img_mean, [0.485, 0.456, 0.406]);
        assert_eq!(cfg.img_std, [0.229, 0.224, 0.225]);
        assert_eq!(cfg.img_resize_mode, "bicubic");
        assert_eq!(cfg.cam_dim_in, 8);
    }

    #[test]
    fn from_gguf_rejects_wrong_arch() {
        let mut entries = full_valid_entries();
        entries[0] = Kv::Str("depthanything3.arch", "some-other-arch");
        let g = build_gguf(&entries);
        match ModelConfig::from_gguf(&g) {
            Err(EngineError::UnsupportedModel(Some(a))) => assert_eq!(a, "some-other-arch"),
            other => panic!("expected UnsupportedModel, got {other:?}"),
        }
    }

    #[test]
    fn from_gguf_rejects_missing_arch() {
        let entries: Vec<Kv> = full_valid_entries()
            .into_iter()
            .filter(|e| !matches!(e, Kv::Str(k, _) if *k == "depthanything3.arch"))
            .collect();
        let g = build_gguf(&entries);
        match ModelConfig::from_gguf(&g) {
            Err(EngineError::UnsupportedModel(None)) => {}
            other => panic!("expected UnsupportedModel(None), got {other:?}"),
        }
    }

    #[test]
    fn from_gguf_reports_missing_key() {
        let entries: Vec<Kv> = full_valid_entries()
            .into_iter()
            .filter(|e| !matches!(e, Kv::U32(k, _) if *k == "depthanything3.vit.embed_dim"))
            .collect();
        let g = build_gguf(&entries);
        match ModelConfig::from_gguf(&g) {
            Err(EngineError::MissingKey(k)) => assert_eq!(k, "depthanything3.vit.embed_dim"),
            other => panic!("expected MissingKey, got {other:?}"),
        }
    }
}
