//! Integration test for the `da bench` binary (Task 22's Step 1/2): runs
//! `da bench --repeat 2 --warmup 1` end-to-end against a from-scratch
//! synthetic GGUF (real binary GGUF bytes, tiny pseudo-random weights) and
//! asserts the output contains a parsable, finite, non-negative
//! `median_ms=`/`p95_ms=` line.
//!
//! This is dump-independent, model-independent coverage of the bench
//! subcommand's own timing/parsing/output logic — distinct from (and not a
//! substitute for) a real numerically-meaningful latency measurement, which
//! needs a real DA3-BASE GGUF (`../models/*.gguf`, absent in this
//! environment — see `cli_smoke.rs`'s `model_path`/`[skip]` pattern for the
//! real-model-gated test, and `docs/optimization-log.md`'s "Task 22" entry
//! for the honesty note on why no real E2E number can be produced here).
//!
//! The synthetic-GGUF builder (`GgufBuilder`/`build_synthetic_gguf`) is a
//! near-verbatim copy of `da-engine/tests/e2e_native.rs`'s helper of the
//! same name — that file documents this exact pattern is "the same pattern
//! already used by `backbone.rs`'s `synthetic_weights` test helper" and
//! deliberately not shared across test binaries (integration tests in
//! different crates can't `use` each other's `tests/` modules), so
//! duplicating it here (rather than reaching into `da-engine`'s test-only
//! code) matches this workspace's established convention.

use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------
// Minimal binary GGUF writer (see da-engine/tests/e2e_native.rs for the
// authoritative doc comments on each piece — kept terse here).
// ---------------------------------------------------------------------

struct GgufBuilder {
    kv: Vec<u8>,
    kv_count: u64,
    tensor_info: Vec<u8>,
    tensor_count: u64,
    data: Vec<u8>,
}

impl GgufBuilder {
    fn new() -> Self {
        GgufBuilder {
            kv: Vec::new(),
            kv_count: 0,
            tensor_info: Vec::new(),
            tensor_count: 0,
            data: Vec::new(),
        }
    }

    fn write_gguf_string(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    fn kv_str(&mut self, key: &str, val: &str) {
        Self::write_gguf_string(&mut self.kv, key);
        self.kv.extend_from_slice(&8u32.to_le_bytes());
        Self::write_gguf_string(&mut self.kv, val);
        self.kv_count += 1;
    }

    fn kv_u32(&mut self, key: &str, val: u32) {
        Self::write_gguf_string(&mut self.kv, key);
        self.kv.extend_from_slice(&4u32.to_le_bytes());
        self.kv.extend_from_slice(&val.to_le_bytes());
        self.kv_count += 1;
    }

    fn kv_i32(&mut self, key: &str, val: i32) {
        Self::write_gguf_string(&mut self.kv, key);
        self.kv.extend_from_slice(&5u32.to_le_bytes());
        self.kv.extend_from_slice(&val.to_le_bytes());
        self.kv_count += 1;
    }

    fn kv_f32(&mut self, key: &str, val: f32) {
        Self::write_gguf_string(&mut self.kv, key);
        self.kv.extend_from_slice(&6u32.to_le_bytes());
        self.kv.extend_from_slice(&val.to_le_bytes());
        self.kv_count += 1;
    }

    fn kv_arr_f32(&mut self, key: &str, vals: &[f32]) {
        Self::write_gguf_string(&mut self.kv, key);
        self.kv.extend_from_slice(&9u32.to_le_bytes());
        self.kv.extend_from_slice(&6u32.to_le_bytes());
        self.kv
            .extend_from_slice(&(vals.len() as u64).to_le_bytes());
        for v in vals {
            self.kv.extend_from_slice(&v.to_le_bytes());
        }
        self.kv_count += 1;
    }

    fn kv_arr_i32(&mut self, key: &str, vals: &[i32]) {
        Self::write_gguf_string(&mut self.kv, key);
        self.kv.extend_from_slice(&9u32.to_le_bytes());
        self.kv.extend_from_slice(&5u32.to_le_bytes());
        self.kv
            .extend_from_slice(&(vals.len() as u64).to_le_bytes());
        for v in vals {
            self.kv.extend_from_slice(&v.to_le_bytes());
        }
        self.kv_count += 1;
    }

    fn tensor_f32(&mut self, name: &str, values: &[f32]) {
        Self::write_gguf_string(&mut self.tensor_info, name);
        self.tensor_info.extend_from_slice(&1u32.to_le_bytes());
        self.tensor_info
            .extend_from_slice(&(values.len() as u64).to_le_bytes());
        self.tensor_info.extend_from_slice(&0u32.to_le_bytes());
        self.tensor_info
            .extend_from_slice(&(self.data.len() as u64).to_le_bytes());
        for v in values {
            self.data.extend_from_slice(&v.to_le_bytes());
        }
        self.tensor_count += 1;
    }

    fn build(self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&self.tensor_count.to_le_bytes());
        buf.extend_from_slice(&self.kv_count.to_le_bytes());
        buf.extend_from_slice(&self.kv);
        buf.extend_from_slice(&self.tensor_info);
        let pad = (32 - (buf.len() % 32)) % 32;
        buf.extend_from_slice(&vec![0u8; pad]);
        buf.extend_from_slice(&self.data);
        buf
    }
}

struct Xorshift32(u32);
impl Xorshift32 {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        (((self.0 as f32) / (u32::MAX as f32)) * 2.0 - 1.0) * 0.02
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32()).collect()
    }
}

/// Same tiny-scale synthetic model as `da-engine/tests/e2e_native.rs`'s
/// `build_synthetic_gguf` (patch_size=2, image_size=4, embed_dim=4, depth=4
/// blocks, DPT head's fixed [96,192,384,768]/128 channel counts, cam_dim_in
/// doubled for `cat_token=true`) — see that file's doc comment for the full
/// rationale behind each constant.
fn build_synthetic_gguf() -> Vec<u8> {
    const PATCH: usize = 2;
    const IMAGE_SIZE: usize = 4;
    const EMBED: usize = 4;
    const DEPTH: usize = 4;
    const MLP_HIDDEN: usize = 8;
    const GRID: usize = IMAGE_SIZE / PATCH;
    const C_IN: usize = 2 * EMBED;
    const OC: [usize; 4] = [96, 192, 384, 768];
    const FUSION_C: usize = 128;
    const FEAT_HALF: usize = 4;
    const OUTPUT_DIM: usize = 2;

    let mut g = GgufBuilder::new();
    let mut rng = Xorshift32(0xC0FF_EE42);

    g.kv_str("depthanything3.arch", "depthanything3");
    g.kv_u32("depthanything3.patch_size", PATCH as u32);
    g.kv_u32("depthanything3.image_size", IMAGE_SIZE as u32);
    g.kv_u32("depthanything3.vit.embed_dim", EMBED as u32);
    g.kv_u32("depthanything3.vit.depth", DEPTH as u32);
    g.kv_u32("depthanything3.vit.num_heads", 2);
    g.kv_u32("depthanything3.vit.head_dim", 2);
    g.kv_u32("depthanything3.vit.mlp_hidden", MLP_HIDDEN as u32);
    g.kv_u32("depthanything3.vit.num_register_tokens", 0);
    g.kv_i32("depthanything3.vit.rope_start", -1);
    g.kv_i32("depthanything3.vit.qknorm_start", -1);
    g.kv_f32("depthanything3.vit.rope_freq", 100.0);
    g.kv_f32("depthanything3.vit.ln_eps", 1e-6);
    g.kv_arr_i32("depthanything3.vit.out_layers", &[0, 1, 2, 3]);
    g.kv_str("depthanything3.vit.ffn_type", "mlp");
    g.kv_i32("depthanything3.vit.alt_start", -1);
    g.kv_u32("depthanything3.head.features", 8);
    g.kv_f32("depthanything3.head.max_depth", 20.0);
    g.kv_arr_f32("depthanything3.img.mean", &[0.0, 0.0, 0.0]);
    g.kv_arr_f32("depthanything3.img.std", &[1.0, 1.0, 1.0]);
    g.kv_str("depthanything3.img.resize_mode", "bilinear");
    g.kv_u32("depthanything3.cam.dim_in", C_IN as u32);

    g.tensor_f32(
        "vit.patch_embed.weight",
        &rng.vec(EMBED * 3 * PATCH * PATCH),
    );
    g.tensor_f32("vit.patch_embed.bias", &rng.vec(EMBED));
    g.tensor_f32("vit.pos_embed", &rng.vec((GRID * GRID + 1) * EMBED));
    g.tensor_f32("vit.cls_token", &rng.vec(EMBED));
    g.tensor_f32("vit.norm.weight", &[1.0; EMBED]);
    g.tensor_f32("vit.norm.bias", &[0.0; EMBED]);
    g.tensor_f32("vit.camera_token", &rng.vec(2 * EMBED));

    for i in 0..DEPTH {
        let p = |suffix: &str| format!("vit.blk.{i}.{suffix}");
        g.tensor_f32(&p("norm1.weight"), &[1.0; EMBED]);
        g.tensor_f32(&p("norm1.bias"), &[0.0; EMBED]);
        g.tensor_f32(&p("norm2.weight"), &[1.0; EMBED]);
        g.tensor_f32(&p("norm2.bias"), &[0.0; EMBED]);
        g.tensor_f32(&p("attn_qkv.weight"), &rng.vec(EMBED * 3 * EMBED));
        g.tensor_f32(&p("attn_qkv.bias"), &rng.vec(3 * EMBED));
        g.tensor_f32(&p("attn_proj.weight"), &rng.vec(EMBED * EMBED));
        g.tensor_f32(&p("attn_proj.bias"), &rng.vec(EMBED));
        g.tensor_f32(&p("mlp_fc1.weight"), &rng.vec(EMBED * MLP_HIDDEN));
        g.tensor_f32(&p("mlp_fc1.bias"), &rng.vec(MLP_HIDDEN));
        g.tensor_f32(&p("mlp_fc2.weight"), &rng.vec(MLP_HIDDEN * EMBED));
        g.tensor_f32(&p("mlp_fc2.bias"), &rng.vec(EMBED));
    }

    for (s, &channels) in OC.iter().enumerate() {
        g.tensor_f32(&format!("head.proj.{s}.weight"), &rng.vec(channels * C_IN));
        g.tensor_f32(&format!("head.proj.{s}.bias"), &rng.vec(channels));
    }
    g.tensor_f32("head.resize.0.weight", &rng.vec(OC[0] * OC[0] * 4 * 4));
    g.tensor_f32("head.resize.0.bias", &rng.vec(OC[0]));
    g.tensor_f32("head.resize.1.weight", &rng.vec(OC[1] * OC[1] * 2 * 2));
    g.tensor_f32("head.resize.1.bias", &rng.vec(OC[1]));
    g.tensor_f32("head.resize.3.weight", &rng.vec(OC[3] * OC[3] * 3 * 3));
    g.tensor_f32("head.resize.3.bias", &rng.vec(OC[3]));

    for (s, &channels) in OC.iter().enumerate() {
        g.tensor_f32(
            &format!("head.scratch.layer{}_rn.weight", s + 1),
            &rng.vec(FUSION_C * channels * 3 * 3),
        );
    }

    for i in 1..=4 {
        if i != 4 {
            g.tensor_f32(
                &format!("head.scratch.rn{i}.rc1.c1.weight"),
                &rng.vec(FUSION_C * FUSION_C * 3 * 3),
            );
            g.tensor_f32(
                &format!("head.scratch.rn{i}.rc1.c1.bias"),
                &rng.vec(FUSION_C),
            );
            g.tensor_f32(
                &format!("head.scratch.rn{i}.rc1.c2.weight"),
                &rng.vec(FUSION_C * FUSION_C * 3 * 3),
            );
            g.tensor_f32(
                &format!("head.scratch.rn{i}.rc1.c2.bias"),
                &rng.vec(FUSION_C),
            );
        }
        g.tensor_f32(
            &format!("head.scratch.rn{i}.rc2.c1.weight"),
            &rng.vec(FUSION_C * FUSION_C * 3 * 3),
        );
        g.tensor_f32(
            &format!("head.scratch.rn{i}.rc2.c1.bias"),
            &rng.vec(FUSION_C),
        );
        g.tensor_f32(
            &format!("head.scratch.rn{i}.rc2.c2.weight"),
            &rng.vec(FUSION_C * FUSION_C * 3 * 3),
        );
        g.tensor_f32(
            &format!("head.scratch.rn{i}.rc2.c2.bias"),
            &rng.vec(FUSION_C),
        );
        g.tensor_f32(
            &format!("head.scratch.rn{i}.out.weight"),
            &rng.vec(FUSION_C * FUSION_C),
        );
        g.tensor_f32(&format!("head.scratch.rn{i}.out.bias"), &rng.vec(FUSION_C));
    }

    g.tensor_f32(
        "head.scratch.out1.weight",
        &rng.vec(FEAT_HALF * FUSION_C * 3 * 3),
    );
    g.tensor_f32("head.scratch.out1.bias", &rng.vec(FEAT_HALF));
    g.tensor_f32(
        "head.scratch.out2a.weight",
        &rng.vec(32 * FEAT_HALF * 3 * 3),
    );
    g.tensor_f32("head.scratch.out2a.bias", &rng.vec(32));
    g.tensor_f32("head.scratch.out2b.weight", &rng.vec(OUTPUT_DIM * 32));
    g.tensor_f32("head.scratch.out2b.bias", &rng.vec(OUTPUT_DIM));

    const HIDDEN0: usize = 6;
    const HIDDEN1: usize = 6;
    g.tensor_f32("cam.bb0.weight", &rng.vec(C_IN * HIDDEN0));
    g.tensor_f32("cam.bb0.bias", &rng.vec(HIDDEN0));
    g.tensor_f32("cam.bb2.weight", &rng.vec(HIDDEN0 * HIDDEN1));
    g.tensor_f32("cam.bb2.bias", &rng.vec(HIDDEN1));
    g.tensor_f32("cam.fc_t.weight", &rng.vec(HIDDEN1 * 3));
    g.tensor_f32("cam.fc_t.bias", &rng.vec(3));
    g.tensor_f32("cam.fc_q.weight", &rng.vec(HIDDEN1 * 4));
    g.tensor_f32("cam.fc_q.bias", &[0.0, 0.0, 0.0, 1.0]);
    g.tensor_f32("cam.fc_fov.weight", &rng.vec(HIDDEN1 * 2));
    g.tensor_f32("cam.fc_fov.bias", &[0.8, 0.8]);

    g.build()
}

fn temp_path(suffix: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let counter = COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "da_cli_bench_native_{pid}_{nanos}_{counter}{suffix}"
    ))
}

#[test]
fn bench_prints_parsable_median_and_p95_against_synthetic_model() {
    let model_path = temp_path(".gguf");
    std::fs::File::create(&model_path)
        .unwrap()
        .write_all(&build_synthetic_gguf())
        .expect("write synthetic gguf");

    // 4x4 RGB PNG matches this synthetic model's `image_size=4` (identity
    // resize regime — see build_synthetic_gguf's doc comment).
    let image_path = temp_path(".png");
    let mut img = image::RgbImage::new(4, 4);
    for y in 0..4 {
        for x in 0..4 {
            img.put_pixel(x, y, image::Rgb([(x * 40) as u8, (y * 40) as u8, 128]));
        }
    }
    img.save(&image_path).expect("write synthetic input PNG");

    let mut cmd =
        assert_cmd::Command::cargo_bin("vestra-engine").expect("Vestra Engine binary should build");
    cmd.arg("bench")
        .arg("--model")
        .arg(&model_path)
        .arg("--image")
        .arg(&image_path)
        .arg("--repeat")
        .arg("2")
        .arg("--warmup")
        .arg("1");
    let assert = cmd.assert().success();
    let output = assert.get_output();
    let stdout = String::from_utf8_lossy(&output.stdout);

    let median_line = stdout
        .lines()
        .find(|l| l.starts_with("median_ms="))
        .unwrap_or_else(|| panic!("no median_ms= line in stdout:\n{stdout}"));
    let median: f64 = median_line
        .trim_start_matches("median_ms=")
        .parse()
        .expect("median_ms value should parse as f64");
    assert!(
        median.is_finite() && median >= 0.0,
        "median_ms should be finite and non-negative, got {median}"
    );

    let p95_line = stdout
        .lines()
        .find(|l| l.starts_with("p95_ms="))
        .unwrap_or_else(|| panic!("no p95_ms= line in stdout:\n{stdout}"));
    let p95: f64 = p95_line
        .trim_start_matches("p95_ms=")
        .parse()
        .expect("p95_ms value should parse as f64");
    assert!(
        p95.is_finite() && p95 >= 0.0,
        "p95_ms should be finite and non-negative, got {p95}"
    );
    assert!(
        p95 >= median - 1e-9,
        "p95 ({p95}) should be >= median ({median}) over 2 samples"
    );

    // Exactly `repeat=2` iter lines, per-iteration timings.
    let iter_lines: Vec<&str> = stdout.lines().filter(|l| l.starts_with("iter[")).collect();
    assert_eq!(
        iter_lines.len(),
        2,
        "expected 2 iter[..]_ms lines for --repeat 2, got: {iter_lines:?}"
    );

    let _ = std::fs::remove_file(&model_path);
    let _ = std::fs::remove_file(&image_path);
}
