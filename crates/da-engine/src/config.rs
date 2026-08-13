use da_gguf::GgufFile;

/// GGUF metadata key names for the `depthanything3` architecture.
/// Verbatim from `include/da_gguf_keys.h` — do not rename.
mod keys {
    pub const ARCH: &str = "depthanything3.arch";
    pub const PATCH_SIZE: &str = "depthanything3.patch_size";
    /// Current converter key used by the C++ reference implementation.
    pub const IMG_RESIZE_TARGET: &str = "depthanything3.img.resize_target";
    /// Legacy key retained for GGUFs produced before the resize contract was
    /// made explicit.
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
    /// FFN flavor: `"mlp"` (classic fc1/fc2, DA3-BASE) or `"swiglu"` (giant
    /// models only). Confirmed against `../scripts/gguf_keys.py` /
    /// `../scripts/convert_da3_to_gguf.py` (`w.add_string(K.KV["vit.ffn_type"], ...)`,
    /// always written by the real converter) and `../src/vit_block.cpp`'s
    /// `load_block` (`w.swiglu = (ml.config().ffn_type == "swiglu")`).
    pub const VIT_FFN_TYPE: &str = "depthanything3.vit.ffn_type";
    /// Layer index at which camera-token injection + local/global attention
    /// alternation begins (`-1` = never, i.e. this model has no alt path).
    /// Confirmed against `include/da_gguf_keys.h`
    /// (`DA_KV_VIT_ALT_START = "depthanything3.vit.alt_start"`) and
    /// `../src/model_loader.cpp` (`kv_i32(gguf_, DA_KV_VIT_ALT_START, -1)`,
    /// default `-1`).
    pub const VIT_ALT_START: &str = "depthanything3.vit.alt_start";
    /// Whether `get_intermediate_layers`-style feats double the channel
    /// width via `cat([local_x, vit_norm(x)])` (`true`, DA3-BASE/giant) or
    /// use the single-width `vit_norm(x)` only (`false`, da2/mono models).
    /// Confirmed against `include/da_gguf_keys.h`
    /// (`DA_KV_VIT_CAT_TOKEN = "depthanything3.vit.cat_token"`) and
    /// `../src/model_loader.cpp` (`kv_bool(gguf_, DA_KV_VIT_CAT_TOKEN,
    /// true)`, default `true`).
    pub const VIT_CAT_TOKEN: &str = "depthanything3.vit.cat_token";
    pub const IMG_MEAN: &str = "depthanything3.img.mean";
    pub const IMG_STD: &str = "depthanything3.img.std";
    pub const IMG_RESIZE_MODE: &str = "depthanything3.img.resize_mode";
    pub const HEAD_FEATURES: &str = "depthanything3.head.features";
    pub const HEAD_MAX_DEPTH: &str = "depthanything3.head.max_depth";
    /// Whether the DPT head adds a UV positional embedding at each
    /// projection stage and after the final upsample. Confirmed against
    /// `include/da_gguf_keys.h` (`DA_KV_HEAD_POS_EMBED =
    /// "depthanything3.head.pos_embed"`) and `../src/model_loader.cpp`
    /// (`cfg_.head_pos_embed = kv_bool(gguf_, DA_KV_HEAD_POS_EMBED, true)`,
    /// default `true` on the non-metric-DPT loader path this workspace
    /// targets).
    pub const HEAD_POS_EMBED: &str = "depthanything3.head.pos_embed";
    pub const CAM_DIM_IN: &str = "depthanything3.cam.dim_in";
}

const EXPECTED_ARCH: &str = "depthanything3";

/// Default `ffn_type` when `depthanything3.vit.ffn_type` is absent from the
/// GGUF metadata. The real converter always writes this key, but a defensive
/// default keeps `ModelConfig::from_gguf` working against older/hand-built
/// files (matching `../src/model_loader.cpp`'s general "kv with default"
/// pattern used for other optional keys like `interp_offset`).
const DEFAULT_FFN_TYPE: &str = "mlp";

/// Default `alt_start` when `depthanything3.vit.alt_start` is absent:
/// "never" (no camera-token injection, no local/global alternation).
/// Matches `../src/model_loader.cpp`'s `kv_i32(..., -1)` default.
const DEFAULT_ALT_START: i32 = -1;

/// Default `cat_token` when `depthanything3.vit.cat_token` is absent:
/// `true` (doubled-width feat/cam). Matches `../src/model_loader.cpp`'s
/// `kv_bool(..., true)` default.
const DEFAULT_CAT_TOKEN: bool = true;

/// Default `head_pos_embed` when `depthanything3.head.pos_embed` is absent:
/// `true` (UV pos-embed added). Matches `../src/model_loader.cpp`'s
/// `kv_bool(..., true)` default — see `keys::HEAD_POS_EMBED`'s doc comment.
const DEFAULT_HEAD_POS_EMBED: bool = true;

#[derive(thiserror::Error, Debug)]
pub enum EngineError {
    #[error("unsupported model architecture: expected '{EXPECTED_ARCH}', found {0:?}")]
    UnsupportedModel(Option<String>),
    #[error("missing required gguf metadata key: {0}")]
    MissingKey(&'static str),
    #[error("malformed gguf metadata for key {0}: {1}")]
    Malformed(&'static str, String),
    /// Matches `../src/cam_pose.cpp::CamPose::pose`'s runtime validation:
    /// `if (cam_token.empty() || !bb0 || (int64_t)cam_token.size() != bb0->ne[0]) return false;`.
    /// `expected` is `cam.bb0.weight`'s input dim (derived from the loaded
    /// weight tensor's shape, not hardcoded); `got` is the caller-supplied
    /// `cam_token.len()`.
    #[error("camera token dimension mismatch: expected {expected} (cam.bb0.weight input dim), got {got}")]
    CamTokenDimMismatch { expected: usize, got: usize },
    /// Wraps a GGUF-level read/parse failure (bad magic, unsupported dtype,
    /// out-of-bounds tensor data, missing tensor, ...) encountered while
    /// opening a model file or bulk-loading its tensors — see
    /// `engine.rs::weights_from_gguf`/`Engine::load`.
    #[error("gguf error: {0}")]
    Gguf(#[from] da_gguf::GgufError),
    /// `Engine::infer` needs at least one out-layer to select a `cam_token`
    /// for pose regression (`BackboneOutputs::cam_tokens`'s LAST entry, by
    /// construction the deepest/final out-layer). A `ModelConfig` with an
    /// empty `out_layers` (malformed GGUF metadata) can't produce one.
    #[error("model config has empty out_layers; cannot select a camera-pose token")]
    EmptyOutLayers,
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
    /// `"mlp"` (default, when the GGUF key is absent) or `"swiglu"`. Only
    /// the `"mlp"` path is implemented by `vit_block` (Task 17); `"swiglu"`
    /// is a deliberate, honest not-yet-supported hard error there.
    pub ffn_type: String,
    /// Layer index at which camera-token injection + local/global attention
    /// alternation begins; `-1` means this model never does either (default
    /// when `depthanything3.vit.alt_start` is absent from GGUF metadata —
    /// see `keys::VIT_ALT_START`'s doc comment for provenance).
    pub alt_start: i32,
    /// Whether captured `feat`/`cam` outputs are the doubled-width
    /// `cat([local_x, vit_norm(x)])`/`cat([local_x[tok0], x[tok0]])` form
    /// (`true`, default when `depthanything3.vit.cat_token` is absent) or
    /// the single-width `vit_norm(x)`/`x[tok0]` form (`false`) — see
    /// `keys::VIT_CAT_TOKEN`'s doc comment for provenance.
    pub cat_token: bool,
    pub head_features: u32,
    pub head_max_depth: f32,
    /// Whether the DPT head adds a UV positional embedding — see
    /// `keys::HEAD_POS_EMBED`'s doc comment for provenance/default.
    pub head_pos_embed: bool,
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
            image_size: f
                .meta_u32(keys::IMG_RESIZE_TARGET)
                .or_else(|| f.meta_u32(keys::IMAGE_SIZE))
                .ok_or(EngineError::MissingKey(keys::IMG_RESIZE_TARGET))?,
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
            ffn_type: f
                .meta_str(keys::VIT_FFN_TYPE)
                .unwrap_or_else(|| DEFAULT_FFN_TYPE.to_string()),
            alt_start: f.meta_i32(keys::VIT_ALT_START).unwrap_or(DEFAULT_ALT_START),
            cat_token: f
                .meta_bool(keys::VIT_CAT_TOKEN)
                .unwrap_or(DEFAULT_CAT_TOKEN),
            head_features: req_u32(f, keys::HEAD_FEATURES)?,
            // DA3-BASE is relative depth and the canonical converter omits
            // this key. This mirrors model_loader.cpp's 0.0 default.
            head_max_depth: f.meta_f32(keys::HEAD_MAX_DEPTH).unwrap_or(0.0),
            head_pos_embed: f
                .meta_bool(keys::HEAD_POS_EMBED)
                .unwrap_or(DEFAULT_HEAD_POS_EMBED),
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
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Atomic counter for generating unique temp filenames in parallel tests.
    static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

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

        // Generate a unique filename: combine process ID, timestamp, and atomic counter
        // to guarantee uniqueness even under parallel test execution.
        let counter = TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("da_engine_test_{}_{}_{}.gguf", pid, nanos, counter));
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
        assert_eq!(cfg.ffn_type, "mlp");
        assert_eq!(cfg.head_features, 256);
        assert_eq!(cfg.head_max_depth, 20.0);
        assert_eq!(cfg.img_mean, [0.485, 0.456, 0.406]);
        assert_eq!(cfg.img_std, [0.229, 0.224, 0.225]);
        assert_eq!(cfg.img_resize_mode, "bicubic");
        assert_eq!(cfg.cam_dim_in, 8);
        assert_eq!(cfg.alt_start, -1);
        assert_eq!(cfg.cat_token, true);
        assert_eq!(cfg.head_pos_embed, true);
    }

    #[test]
    fn head_pos_embed_defaults_to_true_when_key_absent() {
        // full_valid_entries() never includes `depthanything3.head.pos_embed`.
        let g = build_gguf(&full_valid_entries());
        let cfg = ModelConfig::from_gguf(&g).expect("should parse valid config");
        assert_eq!(cfg.head_pos_embed, true);
    }

    #[test]
    fn alt_start_defaults_to_minus_one_when_key_absent() {
        // full_valid_entries() never includes `depthanything3.vit.alt_start`.
        let g = build_gguf(&full_valid_entries());
        let cfg = ModelConfig::from_gguf(&g).expect("should parse valid config");
        assert_eq!(cfg.alt_start, -1);
    }

    #[test]
    fn alt_start_explicit_value_round_trips() {
        let mut entries = full_valid_entries();
        entries.push(Kv::I32("depthanything3.vit.alt_start", 4));
        let g = build_gguf(&entries);
        let cfg = ModelConfig::from_gguf(&g).expect("should parse valid config");
        assert_eq!(cfg.alt_start, 4);
    }

    #[test]
    fn cat_token_defaults_to_true_when_key_absent() {
        // full_valid_entries() never includes `depthanything3.vit.cat_token`.
        let g = build_gguf(&full_valid_entries());
        let cfg = ModelConfig::from_gguf(&g).expect("should parse valid config");
        assert_eq!(cfg.cat_token, true);
    }

    #[test]
    fn cat_token_explicit_false_round_trips() {
        // Kv has no Bool variant, so assemble the buffer directly here
        // (mirroring build_gguf's layout) with a GGUF bool-typed KV entry.
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        let base_entries = full_valid_entries();
        buf.extend_from_slice(&((base_entries.len() + 1) as u64).to_le_bytes());
        for e in &base_entries {
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
                    buf.extend_from_slice(&6u32.to_le_bytes());
                    buf.extend_from_slice(&(v.len() as u64).to_le_bytes());
                    for x in *v {
                        buf.extend_from_slice(&x.to_le_bytes());
                    }
                }
                Kv::ArrI32(k, v) => {
                    write_kv(&mut buf, k, 9);
                    buf.extend_from_slice(&5u32.to_le_bytes());
                    buf.extend_from_slice(&(v.len() as u64).to_le_bytes());
                    for x in *v {
                        buf.extend_from_slice(&x.to_le_bytes());
                    }
                }
            }
        }
        // GGUF bool vtype is 7, stored as a single byte (0/1).
        write_kv(&mut buf, "depthanything3.vit.cat_token", 7);
        buf.push(0u8);
        let pad = (32 - (buf.len() % 32)) % 32;
        buf.extend_from_slice(&vec![0u8; pad]);
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let counter = TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "da_engine_test_cat_token_{}_{}_{}.gguf",
            pid, nanos, counter
        ));
        std::fs::write(&path, &buf).expect("write temp gguf");
        let g = GgufFile::open(&path).expect("open temp gguf");
        let _ = std::fs::remove_file(&path);

        let cfg = ModelConfig::from_gguf(&g).expect("should parse valid config");
        assert_eq!(cfg.cat_token, false);
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
    fn ffn_type_defaults_to_mlp_when_key_absent() {
        // full_valid_entries() never includes `depthanything3.vit.ffn_type`.
        let g = build_gguf(&full_valid_entries());
        let cfg = ModelConfig::from_gguf(&g).expect("should parse valid config");
        assert_eq!(cfg.ffn_type, "mlp");
    }

    #[test]
    fn ffn_type_explicit_swiglu_round_trips() {
        let mut entries = full_valid_entries();
        entries.push(Kv::Str("depthanything3.vit.ffn_type", "swiglu"));
        let g = build_gguf(&entries);
        let cfg = ModelConfig::from_gguf(&g).expect("should parse valid config");
        assert_eq!(cfg.ffn_type, "swiglu");
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
