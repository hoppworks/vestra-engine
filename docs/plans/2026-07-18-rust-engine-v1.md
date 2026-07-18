# Rust-Engine (depth-anything) v1 — Implementierungsplan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eine self-contained Rust-Engine unter `depth-anything-rs/`, die DA3-S/B/L (Depth + Confidence + Pose) auf x86/AVX-512-CPU rechnet und die C++/ggml-Engine komponentenweise auf Latenz schlägt — verifiziert gegen dieselben PyTorch-Referenz-Dumps.

**Architecture:** Schlanke Engine (Ansatz A) aus 5 Auslieferungs-Crates + 1 Dev-Crate in einem Cargo-Workspace. Eigener GGUF-Loader (mmap), statischer Graph-Executor mit vorgeplanter Buffer-Arena (null Allokationen im Forward), eigene Modell-Implementierung. f32-GEMM geliehen (faer, mit tract-linalg als Vergleich); q8_0-Dot-Product selbst aus ggmls AVX-512/VNNI-Kernel portiert. Jede Komponente wird gegen `../dumps/*.gguf` auf Korrektheit und gegen `baseline.json` auf Geschwindigkeit getestet.

**Tech Stack:** Rust (stable), Cargo-Workspace; `memmap2` (mmap), `faer` + `tract-linalg` (GEMM-Kandidaten), `rayon` (Threadpool), `criterion` (Benchmarks), `half` (f16), `image` (Bild-I/O), `clap` (CLI), `serde`/`serde_json` (Manifest/Pose-Ausgabe). SIMD via `core::arch::x86_64`-Intrinsics hinter Runtime-Feature-Dispatch.

## Global Constraints

- **Self-contained:** Aller Code, alle Docs und alle Test-Fixtures-Referenzen leben unter `depth-anything-rs/`. Keine Änderungen an C++-Quellen außer additiv unter `../scripts/`/`../tests/` (nur wenn ein Dump/Bench-Target fehlt). Der Ordner muss durch Löschen/Verschieben von `depth-anything-rs/` restlos entfernbar sein.
- **Ziel-CPU:** x86-64 mit AVX-512F/AVX-512BW/AVX-512VNNI primär; AVX2-Fallback; skalarer Pfad als Referenz-Oracle. Dispatch einmalig beim Start via `is_x86_feature_detected!`.
- **Rust:** stable toolchain, `edition = "2021"`. Kein `#![feature(...)]`, kein `std::simd`. `unsafe` ausschließlich in SIMD-Kerneln, gekapselt hinter safe Funktionen mit `debug_assert!` auf Slice-Längen.
- **Modelle v1:** DA3-SMALL/BASE/LARGE, Ausgaben Depth + Confidence + Pose (Extrinsics 3×4, Intrinsics 3×3). Unbekannte Modelltypen → sauberer `Err`, kein Raten.
- **Formate v1:** f32, f16, q8_0. (q4_k = v2; q5_k/q6_k gestrichen.)
- **Parity-Toleranz:** atol = 2e-3, rtol = 2e-3 (aus `../dumps/manifest.json`), sofern das jeweilige Manifest nichts anderes vorgibt. Vergleich per-Element: `|got - ref| <= atol + rtol*|ref|`.
- **Fehlerbehandlung:** `Result` an den Rändern (Laden, I/O, Modelltyp). Kein `Result` im Forward-Pfad (Graph beim Laden validiert).
- **Commits:** Jede Task endet mit genau einem Commit. Commit-Messages englisch, `feat:`/`test:`/`chore:`-Präfix. Falls das Repo (noch) kein Git ist: Task 0 initialisiert es.
- **Dump-Quelle:** Referenz-Tensoren liegen als flache row-major f32-GGUF-Tensoren in `../dumps/reference.gguf` mit Shapes/Toleranzen in `../dumps/manifest.json` (erzeugt von `../scripts/dump_reference.py`). Tensor-Namen wörtlich wie dort: `input_image`, `pos_embed_added`, `feat_{5,7,9,11}`, `cam_token_{5,7,9,11}`, `rope_in`/`rope_out`/`rope_pos`, `head_stage{0..3}`, `head_fused`, `convt0_in`/`convt0_out`, `convs3_in`/`convs3_out`, `proj0_in`/`proj0_out`, `uv_embed_64`, `head_depth`, `head_depth_conf`, `pose_enc`, `cam_token_in`, `extrinsics`, `intrinsics`.

---

## Meilenstein-Übersicht

| M | Inhalt | Deliverable |
|---|--------|-------------|
| M0 | Workspace-Scaffold, Git, Parity-Harness-Bootstrap | `cargo test` grün, liest `../dumps/reference.gguf` |
| M1 | `da-gguf` voller Loader (mmap, Metadaten, f32/f16/q8_0) | GGUF-Modell + Referenz-Dumps ladbar |
| M2 | `da-kernels` skalare Referenzkernel + GEMM-Integration + **Meilenstein-1-Benchmark** | faer/tract-linalg/ggml-Vergleich entschieden |
| M3 | `da-kernels` AVX-512-SIMD + q8_0-VNNI | Kernel schlagen skalar, parity-grün |
| M4 | `da-graph` statischer Executor + Buffer-Arena + Backend-Trait | Graph plan-/ausführbar, null Forward-Allokationen |
| M5 | `da-engine` Backbone (patch-embed, pos-embed-cache, RoPE2D, ViT-Block, Attention) | `feat_{5,7,9,11}` parity-grün |
| M6 | `da-engine` DPT-Head + Pose-Head + Preprocessing | `head_depth`, `extrinsics`, `intrinsics` parity-grün |
| M7 | `da-cli` infer + bench + E2E-Gate | End-to-end Depth+Pose, Latenz vs. C++ gemessen |

Die **Zwei-Iterationen-Regel** (Spec §6.3) gilt ab M2 für jede Kernel-/Komponenten-Task: nach „schneller als Baseline" zwei benannte Optimierungshypothesen versuchen, außer die Komponente ist nachweislich am Roofline-Limit (dann im Optimierungs-Log begründen). Jede solche Task trägt am Ende einen Eintrag in `depth-anything-rs/docs/optimization-log.md`.

---

# M0 — Workspace-Scaffold & Parity-Bootstrap

### Task 0: Git-Repo & Workspace-Grundgerüst

**Files:**
- Create: `depth-anything-rs/Cargo.toml` (Workspace-Root)
- Create: `depth-anything-rs/README.md`
- Create: `depth-anything-rs/.gitignore`
- Create: `depth-anything-rs/rust-toolchain.toml`

**Interfaces:**
- Produces: Cargo-Workspace mit Member-Liste (Crates werden in Folgetasks angelegt).

- [ ] **Step 1: Prüfen, ob Git initialisiert ist; falls nicht, initialisieren**

Run: `cd /Users/hoppworks/Desktop/depth-anything.cpp-master && git rev-parse --is-inside-work-tree 2>/dev/null || git init`
Expected: entweder `true`, oder Ausgabe „Initialized empty Git repository".

- [ ] **Step 2: Workspace-Root `Cargo.toml` schreiben**

```toml
# depth-anything-rs/Cargo.toml
[workspace]
resolver = "2"
members = [
    # NUR bereits existierende Crates. Cargo löst den gesamten `members`-Baum
    # auf, bevor irgendein Befehl läuft (auch `cargo metadata --no-deps`) —
    # ein fehlendes Member-Verzeichnis lässt JEDEN cargo-Aufruf fehlschlagen,
    # nicht nur den für das fehlende Crate. Deshalb: Mitgliedschaft wächst
    # inkrementell. Jede Task, die eine neue Crate anlegt (1, 2, 5, 12, 14,
    # 21), fügt deren Pfad als Teil ihres eigenen Diffs hier hinzu.
]

[workspace.package]
edition = "2021"
license = "MIT"
rust-version = "1.75"

[workspace.dependencies]
memmap2 = "0.9"
half = "2"
rayon = "1.10"
faer = "0.19"
thiserror = "1"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
image = "0.25"
clap = { version = "4", features = ["derive"] }
criterion = "0.5"

[profile.release]
opt-level = 3
lto = "thin"
codegen-units = 1
panic = "abort"

[profile.bench]
opt-level = 3
lto = "thin"
codegen-units = 1
```

- [ ] **Step 3: `rust-toolchain.toml`, `.gitignore`, `README.md` schreiben**

```toml
# depth-anything-rs/rust-toolchain.toml
[toolchain]
channel = "stable"
```

```gitignore
# depth-anything-rs/.gitignore
/target
Cargo.lock
```

```markdown
<!-- depth-anything-rs/README.md -->
# depth-anything-rs

Self-contained Rust rebuild of the depth-anything engine (DA3-S/B/L: depth + confidence + pose).
Everything lives under this folder and can be removed by deleting it. See `docs/specs/` and `docs/plans/`.

Parity is gated against the C++ repo's reference dumps in `../dumps/` (read-only).
```

- [ ] **Step 4: Workspace baut (leer, aber gültig)**

Run: `cd depth-anything-rs && cargo metadata --no-deps --format-version 1 >/dev/null && echo OK`
Expected: `OK` (Member existieren noch nicht → wir legen sie in Folgetasks an; falls `cargo metadata` über fehlende Member meckert, ist das erwartet bis Task 1).

Hinweis: `members` startet leer (siehe Kommentar im TOML oben). Dieser Step verifiziert nur, dass die TOML syntaktisch gültig ist. **Workspace-Mitgliedschaft wächst inkrementell:** jede Task, die eine neue Crate anlegt (1, 2, 5, 12, 14, 21), fügt deren Pfad zu `members` in `depth-anything-rs/Cargo.toml` als Teil ihres eigenen Commits hinzu — sonst löst kein `cargo`-Befehl im Workspace mehr auf, sobald irgendein gelisteter Member-Pfad fehlt. Commits werden ausschließlich mit expliziten Pfaden erstellt (`git add -- <pfad>`), nie mit `-A`/`-u`/`.` — der Git-Root ist das C++-Repo, ein breiter Add reißt dessen kompletten Baum mit hinein.

- [ ] **Step 5: Commit**

```bash
cd /Users/hoppworks/Desktop/depth-anything.cpp-master
git add depth-anything-rs/Cargo.toml depth-anything-rs/README.md depth-anything-rs/.gitignore depth-anything-rs/rust-toolchain.toml
git commit -m "chore: scaffold self-contained depth-anything-rs cargo workspace"
```

---

### Task 1: `da-gguf` Minimal-Reader (Bootstrap für die Parity-Harness)

Ziel dieser Task: der kleinste GGUF-Reader, der `../dumps/reference.gguf` öffnet und einen benannten f32-Tensor als `Vec<f32>` + Shape zurückgibt. Damit steht die Parity-Harness, bevor der volle Loader existiert (den vollen Loader baut M1 darauf auf).

**Files:**
- Create: `depth-anything-rs/crates/da-gguf/Cargo.toml`
- Create: `depth-anything-rs/crates/da-gguf/src/lib.rs`
- Create: `depth-anything-rs/crates/da-gguf/src/reader.rs`
- Test: `depth-anything-rs/crates/da-gguf/tests/read_reference.rs`

**Interfaces:**
- Produces:
  - `pub struct GgufFile` — geöffnete, gemmapte GGUF-Datei.
  - `pub fn GgufFile::open(path: &Path) -> Result<GgufFile, GgufError>`
  - `pub fn GgufFile::tensor_f32(&self, name: &str) -> Result<TensorF32, GgufError>`
  - `pub struct TensorF32 { pub name: String, pub shape: Vec<i64>, pub data: Vec<f32> }` — `shape` outer→inner (langsamste zuerst), passend zu `parity.hpp`.
  - `pub enum GgufError` (via `thiserror`): `Io`, `BadMagic`, `UnsupportedVersion`, `TensorNotFound(String)`, `UnsupportedDtype(u32)`, `Malformed(String)`.

- [ ] **Step 1: Crate-`Cargo.toml`**

```toml
# depth-anything-rs/crates/da-gguf/Cargo.toml
[package]
name = "da-gguf"
version = "0.1.0"
edition.workspace = true
license.workspace = true

[dependencies]
memmap2.workspace = true
half.workspace = true
thiserror.workspace = true
```

- [ ] **Step 2: Failing test schreiben**

Der Test setzt voraus, dass `../dumps/reference.gguf` existiert (vom C++-Repo erzeugt). Fehlt sie, SKIPpt der Test statt zu failen — so bleibt CI ohne Dumps grün.

```rust
// depth-anything-rs/crates/da-gguf/tests/read_reference.rs
use std::path::Path;
use da_gguf::GgufFile;

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
    assert!(t.shape.contains(&224), "expected a 224 dim, got {:?}", t.shape);
}
```

- [ ] **Step 3: Test laufen lassen — muss fehlschlagen (kompiliert nicht)**

Run: `cd depth-anything-rs && cargo test -p da-gguf --test read_reference`
Expected: FAIL — `da_gguf::GgufFile` existiert nicht.

- [ ] **Step 4: Minimal-Reader implementieren**

GGUF-Layout (little-endian): Magic `GGUF` (0x46554747), u32 version (2 oder 3), u64 tensor_count, u64 metadata_kv_count, dann die KV-Metadaten, dann die Tensor-Infos (Name, n_dims, dims[], dtype u32, offset u64), gefolgt vom auf `alignment` (default 32) gepaddeten Tensor-Datenblock. Für den Bootstrap parsen wir die KV nur so weit, dass wir sie überspringen können, und lesen die Tensor-Infos + Daten.

```rust
// depth-anything-rs/crates/da-gguf/src/lib.rs
mod reader;
pub use reader::{GgufFile, GgufError, TensorF32};
```

```rust
// depth-anything-rs/crates/da-gguf/src/reader.rs
use std::path::Path;
use memmap2::Mmap;
use half::f16;

#[derive(thiserror::Error, Debug)]
pub enum GgufError {
    #[error("io: {0}")] Io(#[from] std::io::Error),
    #[error("bad magic")] BadMagic,
    #[error("unsupported gguf version {0}")] UnsupportedVersion(u32),
    #[error("tensor not found: {0}")] TensorNotFound(String),
    #[error("unsupported dtype {0}")] UnsupportedDtype(u32),
    #[error("malformed: {0}")] Malformed(String),
}

pub struct TensorF32 { pub name: String, pub shape: Vec<i64>, pub data: Vec<f32> }

struct TensorInfo { name: String, dims: Vec<u64>, dtype: u32, offset: u64 }

pub struct GgufFile {
    _mmap: Mmap,
    tensors: Vec<TensorInfo>,
    data_start: usize,
    // Rohbytes als 'static-View auf das mmap (sicher, solange _mmap lebt).
    bytes: *const u8,
    len: usize,
}

// GGML dtype-Codes (Teilmenge v1): F32=0, F16=1, Q8_0=8.
const GGML_F32: u32 = 0;
const GGML_F16: u32 = 1;
const GGML_Q8_0: u32 = 8;

struct Cursor<'a> { b: &'a [u8], p: usize }
impl<'a> Cursor<'a> {
    fn u32(&mut self) -> Result<u32, GgufError> { let v = u32::from_le_bytes(self.take(4)?.try_into().unwrap()); Ok(v) }
    fn u64(&mut self) -> Result<u64, GgufError> { let v = u64::from_le_bytes(self.take(8)?.try_into().unwrap()); Ok(v) }
    fn i32(&mut self) -> Result<i32, GgufError> { Ok(self.u32()? as i32) }
    fn take(&mut self, n: usize) -> Result<&'a [u8], GgufError> {
        if self.p + n > self.b.len() { return Err(GgufError::Malformed("eof".into())); }
        let s = &self.b[self.p..self.p + n]; self.p += n; Ok(s)
    }
    fn gguf_string(&mut self) -> Result<String, GgufError> {
        let n = self.u64()? as usize;
        let s = self.take(n)?;
        Ok(String::from_utf8_lossy(s).into_owned())
    }
}

// KV-Value-Typen laut GGUF-Spec. Wir müssen sie im Bootstrap nur *überspringen*.
fn skip_kv_value(c: &mut Cursor, vtype: u32) -> Result<(), GgufError> {
    match vtype {
        0 | 1 => { c.take(1)?; }                 // uint8/int8
        2 | 3 => { c.take(2)?; }                 // uint16/int16
        4 | 5 => { c.take(4)?; }                 // uint32/int32
        6 => { c.take(4)?; }                     // float32
        7 => { c.take(1)?; }                     // bool
        8 => { let n = c.u64()? as usize; c.take(n)?; }   // string
        10 | 11 => { c.take(8)?; }               // uint64/int64
        12 => { c.take(8)?; }                    // float64
        9 => {                                    // array
            let elem = c.u32()?;
            let n = c.u64()? as usize;
            for _ in 0..n { skip_kv_value(c, elem)?; }
        }
        other => return Err(GgufError::Malformed(format!("kv type {other}"))),
    }
    Ok(())
}

impl GgufFile {
    pub fn open(path: &Path) -> Result<GgufFile, GgufError> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let bytes_ptr = mmap.as_ptr();
        let len = mmap.len();
        let mut c = Cursor { b: &mmap[..], p: 0 };
        if c.take(4)? != b"GGUF" { return Err(GgufError::BadMagic); }
        let version = c.u32()?;
        if version != 2 && version != 3 { return Err(GgufError::UnsupportedVersion(version)); }
        let tensor_count = c.u64()?;
        let kv_count = c.u64()?;
        let mut alignment: u64 = 32;
        for _ in 0..kv_count {
            let key = c.gguf_string()?;
            let vtype = c.u32()?;
            if key == "general.alignment" && vtype == 4 {
                alignment = c.u32()? as u64;
            } else {
                skip_kv_value(&mut c, vtype)?;
            }
        }
        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = c.gguf_string()?;
            let n_dims = c.u32()? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims { dims.push(c.u64()?); }
            let dtype = c.u32()?;
            let offset = c.u64()?;
            tensors.push(TensorInfo { name, dims, dtype, offset });
        }
        // Datenblock beginnt am nächsten `alignment`-Vielfachen nach den Infos.
        let pad = (alignment - (c.p as u64 % alignment)) % alignment;
        let data_start = c.p + pad as usize;
        Ok(GgufFile { _mmap: mmap, tensors, data_start, bytes: bytes_ptr, len })
    }

    fn info(&self, name: &str) -> Result<&TensorInfo, GgufError> {
        self.tensors.iter().find(|t| t.name == name)
            .ok_or_else(|| GgufError::TensorNotFound(name.to_string()))
    }

    fn raw(&self) -> &[u8] { unsafe { std::slice::from_raw_parts(self.bytes, self.len) } }

    pub fn tensor_f32(&self, name: &str) -> Result<TensorF32, GgufError> {
        let ti = self.info(name)?;
        // Shape outer→inner wie parity.hpp: dims sind inner→outer gespeichert, also umdrehen.
        let shape: Vec<i64> = ti.dims.iter().rev().map(|&d| d as i64).collect();
        let n: usize = ti.dims.iter().map(|&d| d as usize).product();
        let base = self.data_start + ti.offset as usize;
        let bytes = self.raw();
        let data = match ti.dtype {
            GGML_F32 => {
                let end = base + n * 4;
                bytes[base..end].chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
            }
            GGML_F16 => {
                let end = base + n * 2;
                bytes[base..end].chunks_exact(2)
                    .map(|c| f16::from_le_bytes(c.try_into().unwrap()).to_f32()).collect()
            }
            other => return Err(GgufError::UnsupportedDtype(other)),
        };
        Ok(TensorF32 { name: name.to_string(), shape, data })
    }
}

// mmap ist über die Lebensdauer von GgufFile gültig; die Rohzeiger sind privat und
// werden nur über &self dereferenziert. Send/Sync sind nicht nötig für v1.
```

- [ ] **Step 5: Test grün**

Run: `cd depth-anything-rs && cargo test -p da-gguf --test read_reference`
Expected: PASS (oder `[skip]` falls keine Dumps vorhanden — dann lokal `../scripts/dump_reference.py` laufen lassen und erneut testen).

- [ ] **Step 6: Commit**

```bash
git add depth-anything-rs/crates/da-gguf
git commit -m "feat(da-gguf): minimal mmap gguf reader for f32/f16 tensors"
```

---

### Task 2: `da-parity` — geteilte Vergleichslogik

Spiegelt `tests/parity.hpp`: lädt Referenztensoren aus einer Dump-GGUF und vergleicht per-Element mit atol/rtol, mit derselben Diagnose-Ausgabe (max/mean abs diff, worst index).

**Files:**
- Create: `depth-anything-rs/da-parity/Cargo.toml`
- Create: `depth-anything-rs/da-parity/src/lib.rs`
- Test: `depth-anything-rs/da-parity/tests/compare_semantics.rs`

**Interfaces:**
- Consumes: `da_gguf::{GgufFile, TensorF32}`.
- Produces:
  - `pub struct Dumps` — geöffnete Dump-GGUF + Manifest-Toleranzen.
  - `pub fn Dumps::open(gguf: &Path, manifest: &Path) -> Result<Dumps, ParityError>`
  - `pub fn Dumps::reference(&self, name: &str) -> Result<TensorF32, ParityError>`
  - `pub fn Dumps::atol(&self) -> f32`, `pub fn Dumps::rtol(&self) -> f32`
  - `pub struct CompareReport { pub ok: bool, pub max_abs: f64, pub mean_abs: f64, pub worst: usize, pub n: usize }`
  - `pub fn compare(got: &[f32], reference: &[f32], atol: f32, rtol: f32, label: &str) -> CompareReport` — druckt die Diagnosezeile nach stderr, gibt `ok` zurück.
  - `pub fn assert_parity(got: &[f32], reference: &[f32], atol: f32, rtol: f32, label: &str)` — panickt bei `!ok` (für Tests).
  - `pub fn dumps_path(rel: &str) -> PathBuf` — löst `../dumps/<rel>` relativ zum Workspace auf; gibt Pfad auch zurück, wenn er nicht existiert (Aufrufer entscheidet über SKIP).

- [ ] **Step 1: Crate-`Cargo.toml`**

```toml
# depth-anything-rs/da-parity/Cargo.toml
[package]
name = "da-parity"
version = "0.1.0"
edition.workspace = true
license.workspace = true

[dependencies]
da-gguf = { path = "../crates/da-gguf" }
serde.workspace = true
serde_json.workspace = true
```

- [ ] **Step 2: Failing test für die Vergleichssemantik**

Diese Tests hängen NICHT an Dumps — sie prüfen die Toleranzlogik selbst und laufen immer.

```rust
// depth-anything-rs/da-parity/tests/compare_semantics.rs
use da_parity::compare;

#[test]
fn within_tolerance_passes() {
    let got = [1.0f32, 2.0, 3.0];
    let refr = [1.001f32, 1.999, 3.0];
    let r = compare(&got, &refr, 2e-3, 2e-3, "unit");
    assert!(r.ok, "should pass: max_abs={}", r.max_abs);
}

#[test]
fn beyond_tolerance_fails_and_reports_worst() {
    let got = [1.0f32, 2.0, 5.0];
    let refr = [1.0f32, 2.0, 3.0];
    let r = compare(&got, &refr, 2e-3, 2e-3, "unit");
    assert!(!r.ok);
    assert_eq!(r.worst, 2);
    assert!((r.max_abs - 2.0).abs() < 1e-9);
}

#[test]
fn empty_is_never_a_pass() {
    let r = compare(&[], &[], 1.0, 1.0, "empty");
    assert!(!r.ok);
}
```

- [ ] **Step 3: Test laufen lassen — Fail (kompiliert nicht)**

Run: `cd depth-anything-rs && cargo test -p da-parity --test compare_semantics`
Expected: FAIL — `da_parity::compare` fehlt.

- [ ] **Step 4: Implementieren**

```rust
// depth-anything-rs/da-parity/src/lib.rs
use std::path::{Path, PathBuf};
use da_gguf::{GgufFile, TensorF32};

#[derive(thiserror::Error, Debug)]
pub enum ParityError {
    #[error("gguf: {0}")] Gguf(#[from] da_gguf::GgufError),
    #[error("io: {0}")] Io(#[from] std::io::Error),
    #[error("manifest: {0}")] Manifest(String),
}

pub struct Dumps { gguf: GgufFile, atol: f32, rtol: f32 }

#[derive(serde::Deserialize)]
struct Manifest { #[serde(default = "d_atol")] atol: f32, #[serde(default = "d_rtol")] rtol: f32 }
fn d_atol() -> f32 { 2e-3 } fn d_rtol() -> f32 { 2e-3 }

impl Dumps {
    pub fn open(gguf: &Path, manifest: &Path) -> Result<Dumps, ParityError> {
        let g = GgufFile::open(gguf)?;
        let m: Manifest = serde_json::from_slice(&std::fs::read(manifest)?)
            .map_err(|e| ParityError::Manifest(e.to_string()))?;
        Ok(Dumps { gguf: g, atol: m.atol, rtol: m.rtol })
    }
    pub fn reference(&self, name: &str) -> Result<TensorF32, ParityError> { Ok(self.gguf.tensor_f32(name)?) }
    pub fn atol(&self) -> f32 { self.atol }
    pub fn rtol(&self) -> f32 { self.rtol }
}

pub struct CompareReport { pub ok: bool, pub max_abs: f64, pub mean_abs: f64, pub worst: usize, pub n: usize }

pub fn compare(got: &[f32], reference: &[f32], atol: f32, rtol: f32, label: &str) -> CompareReport {
    if got.len() != reference.len() {
        eprintln!("[{label}] size mismatch got={} ref={}", got.len(), reference.len());
        return CompareReport { ok: false, max_abs: f64::INFINITY, mean_abs: f64::INFINITY, worst: 0, n: 0 };
    }
    if got.is_empty() {
        eprintln!("[{label}] empty vectors -> FAIL");
        return CompareReport { ok: false, max_abs: 0.0, mean_abs: 0.0, worst: 0, n: 0 };
    }
    let (mut max_abs, mut sum, mut worst) = (0.0f64, 0.0f64, 0usize);
    for i in 0..got.len() {
        let d = (got[i] as f64 - reference[i] as f64).abs();
        sum += d;
        if d > max_abs { max_abs = d; worst = i; }
    }
    let mean = sum / got.len() as f64;
    let mut ok = true;
    for i in 0..got.len() {
        let tol = atol as f64 + rtol as f64 * (reference[i] as f64).abs();
        if (got[i] as f64 - reference[i] as f64).abs() > tol { ok = false; break; }
    }
    eprintln!("[{label}] n={} max|d|={:.3e} mean|d|={:.3e} (worst@{} got={:.5} ref={:.5}) -> {}",
        got.len(), max_abs, mean, worst, got[worst], reference[worst], if ok {"OK"} else {"FAIL"});
    CompareReport { ok, max_abs, mean_abs: mean, worst, n: got.len() }
}

pub fn assert_parity(got: &[f32], reference: &[f32], atol: f32, rtol: f32, label: &str) {
    let r = compare(got, reference, atol, rtol, label);
    assert!(r.ok, "[{label}] parity FAIL max|d|={:.3e}", r.max_abs);
}

pub fn dumps_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../dumps").join(rel)
}
```

- [ ] **Step 5: Test grün**

Run: `cd depth-anything-rs && cargo test -p da-parity`
Expected: PASS (3 Tests).

- [ ] **Step 6: Commit**

```bash
git add depth-anything-rs/da-parity
git commit -m "feat(da-parity): shared tolerance compare mirroring parity.hpp"
```

---

# M1 — `da-gguf` voller Loader

### Task 3: GGUF-Metadaten-Zugriff (KV lesen statt nur überspringen)

**Files:**
- Modify: `depth-anything-rs/crates/da-gguf/src/reader.rs`
- Create: `depth-anything-rs/crates/da-gguf/src/meta.rs`
- Modify: `depth-anything-rs/crates/da-gguf/src/lib.rs`
- Test: `depth-anything-rs/crates/da-gguf/tests/read_meta.rs`

**Interfaces:**
- Produces:
  - `pub enum MetaValue { U32(u32), I32(i32), F32(f32), U64(u64), Bool(bool), Str(String), ArrU32(Vec<u32>), ArrI32(Vec<i32>), ArrF32(Vec<f32>), ArrStr(Vec<String>) }`
  - `pub fn GgufFile::meta(&self, key: &str) -> Option<&MetaValue>`
  - `pub fn GgufFile::meta_u32(&self, key: &str) -> Option<u32>` (Convenience; analog `_f32`, `_str`, `_arr_i32`).
  - `pub fn GgufFile::tensor_names(&self) -> impl Iterator<Item = &str>`

- [ ] **Step 1: Failing test**

```rust
// depth-anything-rs/crates/da-gguf/tests/read_meta.rs
use da_gguf::GgufFile;
use std::path::Path;

fn model() -> Option<GgufFile> {
    // Ein echtes DA3-BASE-Modell wird via ../scripts/download_model.py bereitgestellt.
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../models/da3-base-f16.gguf");
    if !p.exists() { eprintln!("[skip] no model at {}", p.display()); return None; }
    Some(GgufFile::open(&p).unwrap())
}

#[test]
fn reads_arch_and_vit_dims() {
    let Some(m) = model() else { return };
    assert_eq!(m.meta_str("depthanything3.arch").as_deref(), Some("depthanything3"));
    assert!(m.meta_u32("depthanything3.vit.embed_dim").unwrap() >= 384);
    assert!(m.meta_u32("depthanything3.vit.depth").unwrap() >= 12);
}
```

- [ ] **Step 2: Fail bestätigen**

Run: `cd depth-anything-rs && cargo test -p da-gguf --test read_meta`
Expected: FAIL — `meta_str`/`meta_u32` fehlen.

- [ ] **Step 3: KV parsen und speichern**

In `reader.rs` `skip_kv_value` durch `read_kv_value(...) -> MetaValue` ersetzen und alle KV in einer `Vec<(String, MetaValue)>` in `GgufFile` ablegen. `meta.rs` enthält `MetaValue` + die typed getter. `general.alignment` weiterhin gesondert auswerten. (Vollständige `MetaValue`-Match-Arme für Typen 0–12 und Array-Typ 9 wie im Bootstrap-Skip, nur jetzt mit Rückgabe statt Verwerfen.)

Getter-Beispiel:
```rust
// meta.rs
impl super::GgufFile {
    pub fn meta(&self, key: &str) -> Option<&MetaValue> {
        self.kv.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
    pub fn meta_u32(&self, key: &str) -> Option<u32> {
        match self.meta(key)? { MetaValue::U32(v) => Some(*v), MetaValue::I32(v) => Some(*v as u32), _ => None }
    }
    pub fn meta_f32(&self, key: &str) -> Option<f32> {
        match self.meta(key)? { MetaValue::F32(v) => Some(*v), _ => None }
    }
    pub fn meta_str(&self, key: &str) -> Option<String> {
        match self.meta(key)? { MetaValue::Str(s) => Some(s.clone()), _ => None }
    }
    pub fn meta_arr_i32(&self, key: &str) -> Option<Vec<i32>> {
        match self.meta(key)? { MetaValue::ArrI32(v) => Some(v.clone()),
            MetaValue::ArrU32(v) => Some(v.iter().map(|&x| x as i32).collect()), _ => None }
    }
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> { self.tensors.iter().map(|t| t.name.as_str()) }
}
```

- [ ] **Step 4: Test grün**

Run: `cd depth-anything-rs && cargo test -p da-gguf`
Expected: PASS (oder `[skip]` ohne Modell).

- [ ] **Step 5: Commit**

```bash
git add depth-anything-rs/crates/da-gguf
git commit -m "feat(da-gguf): parse and expose typed metadata KV"
```

---

### Task 4: q8_0-Tensor-Dequantisierung + Block-Zugriff

Der Loader muss q8_0-Tensoren sowohl als dequantisiertes `TensorF32` (für Parity/Fallback) als auch als rohe Blocks (für den schnellen Kernel-Pfad) bereitstellen.

**Files:**
- Create: `depth-anything-rs/crates/da-gguf/src/q8_0.rs`
- Modify: `depth-anything-rs/crates/da-gguf/src/reader.rs`
- Test: `depth-anything-rs/crates/da-gguf/tests/q8_0_roundtrip.rs`

**Interfaces:**
- Produces:
  - `pub const QK8_0: usize = 32;`
  - `#[repr(C)] pub struct BlockQ8_0 { pub d: f16, pub qs: [i8; 32] }` — exakt ggmls Layout (34 Bytes).
  - `pub fn GgufFile::tensor_q8_0(&self, name: &str) -> Result<TensorQ8_0, GgufError>` mit `pub struct TensorQ8_0 { pub shape: Vec<i64>, pub blocks: Vec<BlockQ8_0> }`.
  - `pub fn dequantize_q8_0(blocks: &[BlockQ8_0], out: &mut [f32])` — `out.len()` muss `blocks.len()*32` sein.
  - `GgufFile::tensor_f32` akzeptiert ab jetzt auch dtype Q8_0 (dequantisiert transparent).

- [ ] **Step 1: Failing test — Dequantisierung ist die Umkehr der ggml-Quantisierung**

```rust
// depth-anything-rs/crates/da-gguf/tests/q8_0_roundtrip.rs
use da_gguf::{BlockQ8_0, dequantize_q8_0, QK8_0};
use half::f16;

#[test]
fn dequant_matches_scale_times_qs() {
    // Ein Block mit bekanntem Scale d und Quanten qs -> Werte = d * qs.
    let d = 0.5f32;
    let mut qs = [0i8; 32];
    for i in 0..32 { qs[i] = (i as i8) - 16; } // -16..15
    let blk = BlockQ8_0 { d: f16::from_f32(d), qs };
    let mut out = vec![0f32; QK8_0];
    dequantize_q8_0(std::slice::from_ref(&blk), &mut out);
    for i in 0..32 {
        let expected = d * ((i as i32 - 16) as f32);
        assert!((out[i] - expected).abs() < 1e-3, "i={i} got={} exp={}", out[i], expected);
    }
}
```

- [ ] **Step 2: Fail bestätigen**

Run: `cd depth-anything-rs && cargo test -p da-gguf --test q8_0_roundtrip`
Expected: FAIL — Symbole fehlen.

- [ ] **Step 3: Implementieren**

```rust
// depth-anything-rs/crates/da-gguf/src/q8_0.rs
use half::f16;

pub const QK8_0: usize = 32;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlockQ8_0 { pub d: f16, pub qs: [i8; 32] }

pub struct TensorQ8_0 { pub shape: Vec<i64>, pub blocks: Vec<BlockQ8_0> }

pub fn dequantize_q8_0(blocks: &[BlockQ8_0], out: &mut [f32]) {
    debug_assert_eq!(out.len(), blocks.len() * QK8_0);
    for (bi, b) in blocks.iter().enumerate() {
        let d = b.d.to_f32();
        let base = bi * QK8_0;
        for j in 0..QK8_0 { out[base + j] = d * b.qs[j] as f32; }
    }
}

// Rohbytes eines Q8_0-Tensors (dicht gepackt, 34 Bytes/Block) → Vec<BlockQ8_0>.
pub(crate) fn read_blocks(bytes: &[u8], n_elems: usize) -> Vec<BlockQ8_0> {
    let nblocks = n_elems / QK8_0;
    let mut v = Vec::with_capacity(nblocks);
    let mut p = 0usize;
    for _ in 0..nblocks {
        let d = f16::from_le_bytes([bytes[p], bytes[p + 1]]); p += 2;
        let mut qs = [0i8; 32];
        for j in 0..32 { qs[j] = bytes[p + j] as i8; }
        p += 32;
        v.push(BlockQ8_0 { d, qs });
    }
    v
}
```

In `reader.rs`: `GGML_Q8_0`-Arm in `tensor_f32` ergänzen (Blocks lesen → `dequantize_q8_0`), sowie `tensor_q8_0` hinzufügen (nutzt `read_blocks`). `pub use q8_0::{BlockQ8_0, TensorQ8_0, dequantize_q8_0, QK8_0};` in `lib.rs`.

- [ ] **Step 4: Test grün + Parity-Check gegen echten q8_0-Tensor**

Run: `cd depth-anything-rs && cargo test -p da-gguf`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add depth-anything-rs/crates/da-gguf
git commit -m "feat(da-gguf): q8_0 block access and transparent dequantization"
```

---

# M2 — `da-kernels` Fundament + Meilenstein-1-Benchmark

### Task 5: Skalare Referenzkernel (die Oracles)

Jeder SIMD-Kernel wird später gegen seinen skalaren Zwilling getestet. Diese Task baut die skalaren Zwillinge zuerst: sie sind klein, offensichtlich korrekt und gegen die Dumps gatebar.

**Files:**
- Create: `depth-anything-rs/crates/da-kernels/Cargo.toml`
- Create: `depth-anything-rs/crates/da-kernels/src/lib.rs`
- Create: `depth-anything-rs/crates/da-kernels/src/scalar.rs`
- Test: `depth-anything-rs/crates/da-kernels/tests/scalar_ops.rs`
- Modify: `depth-anything-rs/Cargo.toml` — `"crates/da-kernels"` zu `members` hinzufügen (Workspace-Mitgliedschaft wächst inkrementell, siehe Task 0).

**Interfaces:**
- Produces (alle im Modul `scalar`, freie Funktionen über Slices):
  - `pub fn gemm_f32(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32])` — C[m×n] = A[m×k] · B[k×n], row-major, C wird überschrieben.
  - `pub fn layernorm(x: &mut [f32], rows: usize, cols: usize, gamma: &[f32], beta: &[f32], eps: f32)` — in-place, pro Zeile.
  - `pub fn gelu(x: &mut [f32])` — exakte erf-GELU (`0.5*x*(1+erf(x/sqrt2))`).
  - `pub fn softmax_rows(x: &mut [f32], rows: usize, cols: usize)` — numerisch stabil (max-subtract).
  - `pub fn add(dst: &mut [f32], src: &[f32])`, `pub fn add_bias_rows(x: &mut [f32], rows: usize, cols: usize, bias: &[f32])`.

- [ ] **Step 1: Failing tests (bekannte kleine Fälle)**

```rust
// depth-anything-rs/crates/da-kernels/tests/scalar_ops.rs
use da_kernels::scalar::*;

#[test]
fn gemm_2x2_identity() {
    let a = [1.,2., 3.,4.];       // 2x2
    let id = [1.,0., 0.,1.];      // 2x2
    let mut c = [0.;4];
    gemm_f32(2,2,2,&a,&id,&mut c);
    assert_eq!(c, a);
}

#[test]
fn softmax_rows_sums_to_one() {
    let mut x = [1.,2.,3., 0.,0.,0.]; // 2 rows, 3 cols
    softmax_rows(&mut x, 2, 3);
    let s0: f32 = x[0..3].iter().sum();
    let s1: f32 = x[3..6].iter().sum();
    assert!((s0-1.0).abs()<1e-6 && (s1-1.0).abs()<1e-6);
    assert!((x[3]-1.0/3.0).abs()<1e-6);
}

#[test]
fn gelu_zero_and_large() {
    let mut x = [0.0f32, 10.0, -10.0];
    gelu(&mut x);
    assert!(x[0].abs() < 1e-6);
    assert!((x[1]-10.0).abs() < 1e-3);
    assert!(x[2].abs() < 1e-3);
}

#[test]
fn layernorm_zero_mean_unit_var() {
    let mut x = [1.,2.,3.,4.];
    let g = [1.,1.,1.,1.]; let b = [0.,0.,0.,0.];
    layernorm(&mut x, 1, 4, &g, &b, 1e-5);
    let mean: f32 = x.iter().sum::<f32>()/4.0;
    assert!(mean.abs() < 1e-4);
}
```

- [ ] **Step 2: Fail bestätigen**

Run: `cd depth-anything-rs && cargo test -p da-kernels --test scalar_ops`
Expected: FAIL — Crate/Funktionen fehlen.

- [ ] **Step 3: Implementieren**

`Cargo.toml`:
```toml
# depth-anything-rs/crates/da-kernels/Cargo.toml
[package]
name = "da-kernels"
version = "0.1.0"
edition.workspace = true
license.workspace = true

[dependencies]
da-gguf = { path = "../da-gguf" }
faer.workspace = true
half.workspace = true

[dev-dependencies]
criterion.workspace = true
da-parity = { path = "../../da-parity" }

[[bench]]
name = "gemm_bench"
harness = false
```

`lib.rs`:
```rust
// depth-anything-rs/crates/da-kernels/src/lib.rs
pub mod scalar;
```

`scalar.rs` — vollständige, offensichtlich korrekte Implementierungen:
```rust
// depth-anything-rs/crates/da-kernels/src/scalar.rs
pub fn gemm_f32(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    debug_assert_eq!(a.len(), m*k); debug_assert_eq!(b.len(), k*n); debug_assert_eq!(c.len(), m*n);
    for i in 0..m {
        for j in 0..n { c[i*n+j] = 0.0; }
        for p in 0..k {
            let aip = a[i*k+p];
            for j in 0..n { c[i*n+j] += aip * b[p*n+j]; }
        }
    }
}

pub fn add(dst: &mut [f32], src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    for i in 0..dst.len() { dst[i] += src[i]; }
}

pub fn add_bias_rows(x: &mut [f32], rows: usize, cols: usize, bias: &[f32]) {
    debug_assert_eq!(x.len(), rows*cols); debug_assert_eq!(bias.len(), cols);
    for r in 0..rows { for c in 0..cols { x[r*cols+c] += bias[c]; } }
}

pub fn layernorm(x: &mut [f32], rows: usize, cols: usize, gamma: &[f32], beta: &[f32], eps: f32) {
    debug_assert_eq!(x.len(), rows*cols);
    for r in 0..rows {
        let row = &mut x[r*cols..(r+1)*cols];
        let mean = row.iter().sum::<f32>() / cols as f32;
        let var = row.iter().map(|v| { let d = v - mean; d*d }).sum::<f32>() / cols as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for c in 0..cols { row[c] = (row[c] - mean) * inv * gamma[c] + beta[c]; }
    }
}

pub fn gelu(x: &mut [f32]) {
    const INV_SQRT2: f32 = 0.707_106_78;
    for v in x.iter_mut() { *v = 0.5 * *v * (1.0 + erf(*v * INV_SQRT2)); }
}

// Abramowitz–Stegun 7.1.26 erf-Approximation (|error| < 1.5e-7).
fn erf(x: f32) -> f32 {
    let s = x.signum(); let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0 - (((((1.061_405_4*t - 1.453_152_0)*t) + 1.421_413_7)*t - 0.284_496_74)*t + 0.254_829_59)*t * (-x*x).exp();
    s * y
}

pub fn softmax_rows(x: &mut [f32], rows: usize, cols: usize) {
    debug_assert_eq!(x.len(), rows*cols);
    for r in 0..rows {
        let row = &mut x[r*cols..(r+1)*cols];
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for v in row.iter_mut() { *v = (*v - m).exp(); sum += *v; }
        let inv = 1.0 / sum;
        for v in row.iter_mut() { *v *= inv; }
    }
}
```

- [ ] **Step 4: Tests grün**

Run: `cd depth-anything-rs && cargo test -p da-kernels --test scalar_ops`
Expected: PASS (4 Tests).

- [ ] **Step 5: Commit**

```bash
git add depth-anything-rs/crates/da-kernels
git commit -m "feat(da-kernels): scalar reference kernels (gemm, layernorm, gelu, softmax)"
```

---

### Task 6: GEMM-Backend-Trait + faer-Anbindung + Meilenstein-1-Benchmark

**Der Go/No-Go der ganzen Strategie.** Wir kapseln f32-GEMM hinter einen Trait, implementieren ihn mit faer, und benchmarken faer vs. skalar (und dokumentieren, wie die C++/ggml-Zahl daneben gestellt wird). Der Epilogen-Fusions-Test aus Spec §9 wird hier als expliziter Prüfpunkt ausgeführt.

**Files:**
- Create: `depth-anything-rs/crates/da-kernels/src/gemm.rs`
- Modify: `depth-anything-rs/crates/da-kernels/src/lib.rs`
- Create: `depth-anything-rs/crates/da-kernels/benches/gemm_bench.rs`
- Create: `depth-anything-rs/docs/optimization-log.md`
- Test: `depth-anything-rs/crates/da-kernels/tests/gemm_backends_agree.rs`

**Interfaces:**
- Produces:
  - `pub trait Gemm { fn gemm(&self, m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]); }`
  - `pub struct FaerGemm;` impl `Gemm`.
  - `pub struct ScalarGemm;` impl `Gemm` (delegiert an `scalar::gemm_f32`).
  - `pub struct GemmWithEpilogue` — GEMM + optionale fused Bias/GELU/LayerNorm-Epiloge; Signatur:
    `fn gemm_bias_gelu(&self, m,n,k, a,b, bias: Option<&[f32]>, gelu: bool, c: &mut [f32])`. (Fusions-Existenzbeweis für §9.)

- [ ] **Step 1: Failing test — beide Backends müssen bit-nah übereinstimmen**

```rust
// depth-anything-rs/crates/da-kernels/tests/gemm_backends_agree.rs
use da_kernels::gemm::{Gemm, FaerGemm, ScalarGemm};

fn rand_vec(n: usize, seed: u64) -> Vec<f32> {
    // deterministischer LCG, kein rand-crate nötig
    let mut s = seed; (0..n).map(|_| { s = s.wrapping_mul(6364136223846793005).wrapping_add(1); ((s >> 33) as f32 / u32::MAX as f32) - 0.5 }).collect()
}

#[test]
fn faer_matches_scalar() {
    let (m,n,k) = (64, 48, 80);
    let a = rand_vec(m*k, 1); let b = rand_vec(k*n, 2);
    let mut cs = vec![0.; m*n]; let mut cf = vec![0.; m*n];
    ScalarGemm.gemm(m,n,k,&a,&b,&mut cs);
    FaerGemm.gemm(m,n,k,&a,&b,&mut cf);
    for i in 0..m*n {
        assert!((cs[i]-cf[i]).abs() <= 1e-3 + 1e-3*cs[i].abs(), "i={i} scalar={} faer={}", cs[i], cf[i]);
    }
}
```

- [ ] **Step 2: Fail bestätigen**

Run: `cd depth-anything-rs && cargo test -p da-kernels --test gemm_backends_agree`
Expected: FAIL — `gemm`-Modul fehlt.

- [ ] **Step 3: `gemm.rs` implementieren (faer via MatRef/MatMut)**

```rust
// depth-anything-rs/crates/da-kernels/src/gemm.rs
use crate::scalar;

pub trait Gemm {
    fn gemm(&self, m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]);
}

pub struct ScalarGemm;
impl Gemm for ScalarGemm {
    fn gemm(&self, m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
        scalar::gemm_f32(m, n, k, a, b, c);
    }
}

pub struct FaerGemm;
impl Gemm for FaerGemm {
    fn gemm(&self, m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
        use faer::{MatRef, MatMut, Parallelism};
        // row-major Slices als faer-Views mit expliziten Strides interpretieren.
        let a = unsafe { MatRef::from_raw_parts(a.as_ptr(), m, k, k as isize, 1) };
        let b = unsafe { MatRef::from_raw_parts(b.as_ptr(), k, n, n as isize, 1) };
        let cm = unsafe { MatMut::from_raw_parts_mut(c.as_mut_ptr(), m, n, n as isize, 1) };
        faer::linalg::matmul::matmul(cm, a, b, None, 1.0, Parallelism::None);
    }
}

pub struct GemmWithEpilogue<G: Gemm> { pub inner: G }
impl<G: Gemm> GemmWithEpilogue<G> {
    pub fn gemm_bias_gelu(&self, m: usize, n: usize, k: usize, a: &[f32], b: &[f32],
                          bias: Option<&[f32]>, gelu: bool, c: &mut [f32]) {
        self.inner.gemm(m, n, k, a, b, c);
        if let Some(bias) = bias { scalar::add_bias_rows(c, m, n, bias); }
        if gelu { scalar::gelu(c); }
    }
}
```
`lib.rs`: `pub mod gemm;` ergänzen.

Hinweis zum Fusions-Existenzbeweis (§9): v1 nutzt bewusst die *post-hoc*-Fusion oben (Epilog als separater, aber cache-warmer Pass direkt nach dem GEMM). Ob faers `matmul` einen *echten* in-Kernel-Epilog erlaubt, ist die Optimierungshypothese für die Zwei-Iterationen-Regel dieser Komponente — Ergebnis wird im Optimierungs-Log festgehalten, nicht hier erzwungen.

- [ ] **Step 4: Test grün**

Run: `cd depth-anything-rs && cargo test -p da-kernels --test gemm_backends_agree`
Expected: PASS.

- [ ] **Step 5: Meilenstein-1-Benchmark schreiben**

```rust
// depth-anything-rs/crates/da-kernels/benches/gemm_bench.rs
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use da_kernels::gemm::{Gemm, FaerGemm, ScalarGemm};

fn vit_block_gemm_shapes() -> Vec<(usize,usize,usize)> {
    // Repräsentative DA3-BASE-ViT-Block-GEMMs bei 256 Tokens, embed 768, mlp 3072:
    // QKV-Projektion, Attn-Output-Projektion, MLP-fc1, MLP-fc2.
    vec![(256,2304,768), (256,768,768), (256,3072,768), (256,768,3072)]
}

fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("vit_block_gemm");
    for (m,n,k) in vit_block_gemm_shapes() {
        let a = vec![0.01f32; m*k]; let b = vec![0.01f32; k*n];
        let mut out = vec![0f32; m*n];
        g.bench_with_input(BenchmarkId::new("faer", format!("{m}x{n}x{k}")), &(), |bch, _| {
            bch.iter(|| FaerGemm.gemm(m,n,k,&a,&b,&mut out));
        });
        g.bench_with_input(BenchmarkId::new("scalar", format!("{m}x{n}x{k}")), &(), |bch, _| {
            bch.iter(|| ScalarGemm.gemm(m,n,k,&a,&b,&mut out));
        });
    }
    g.finish();
}
criterion_group!(benches, bench); criterion_main!(benches);
```

- [ ] **Step 6: Benchmark laufen lassen & Meilenstein-1 dokumentieren**

Run: `cd depth-anything-rs && cargo bench -p da-kernels --bench gemm_bench`
Expected: criterion druckt Zeiten pro Shape; faer deutlich schneller als scalar.

Dann `docs/optimization-log.md` anlegen und die erste Zeile eintragen (mit den realen criterion-Zahlen):
```markdown
# Optimierungs-Log

Jede Kernel-/Komponenten-Task trägt hier nach der Zwei-Iterationen-Regel (Spec §6.3) ein.

## Meilenstein 1 — GEMM-Baustein (vit_block-GEMMs, DA3-BASE @256 Tokens)
- faer vs. scalar: <criterion-Zahlen einsetzen>
- faer vs. ggml/tinyBLAS (C++-Baseline, gleiche Shapes): <baseline.json-Zahlen einsetzen, siehe Task 24>
- Epilogen-Fusion durch faer::matmul möglich? <ja/nein/bedingt — Befund>
- **Go/No-Go:** faer erreicht __ % der ggml-Zeit -> <Entscheidung: faer behalten / tract-linalg testen / eigenen Mikrokernel (Ansatz B) für diese Op eskalieren>
```

- [ ] **Step 7: Commit**

```bash
git add depth-anything-rs/crates/da-kernels depth-anything-rs/docs/optimization-log.md
git commit -m "feat(da-kernels): gemm backend trait + faer impl + milestone-1 benchmark"
```

---

### Task 7: tract-linalg-Vergleichskandidat (nur Benchmark, Entscheidung datenbasiert)

**Files:**
- Modify: `depth-anything-rs/crates/da-kernels/Cargo.toml` (dev-dep `tract-linalg`)
- Modify: `depth-anything-rs/crates/da-kernels/benches/gemm_bench.rs`
- Modify: `depth-anything-rs/docs/optimization-log.md`

**Interfaces:**
- Konsumiert nur intern; produziert eine Vergleichszeile im Benchmark. Falls tract-linalg als Dependency zu schwer wiegt oder die API-Anbindung den Rahmen sprengt, wird diese Task als „verworfen: Grund" im Log geschlossen — sie ist ein Vergleichspunkt, kein Auslieferungsteil.

- [ ] **Step 1: tract-linalg als dev-dependency ergänzen**

```toml
# in [dev-dependencies] von da-kernels/Cargo.toml
tract-linalg = "0.21"
```

- [ ] **Step 2: Vergleichsarm im Benchmark ergänzen**

Einen dritten `bench_with_input`-Arm `"tract"` hinzufügen, der tract-linalgs `mmm` (matrix-matrix-multiply) für dieselben Shapes aufruft. (tract-linalg exponiert `tract_linalg::ops().mmm(...)`; den gepackten A/B-Puffer einmalig vorbereiten, dann in `iter()` nur `run` messen.)

- [ ] **Step 3: Benchmark laufen lassen**

Run: `cd depth-anything-rs && cargo bench -p da-kernels --bench gemm_bench`
Expected: drei Kurven (faer/scalar/tract).

- [ ] **Step 4: Optimierungs-Log um faer-vs-tract-Zeile ergänzen und GEMM-Backend final wählen**

Die Wahl (faer bleibt Default, es sei denn tract ist auf den ViT-Shapes klar schneller UND erlaubt Epilogen-Fusion) wird als Entscheidung ins Log geschrieben.

- [ ] **Step 5: Commit**

```bash
git add depth-anything-rs/crates/da-kernels depth-anything-rs/docs/optimization-log.md
git commit -m "bench(da-kernels): add tract-linalg comparison arm and record gemm decision"
```

---

# M3 — SIMD & q8_0

### Task 8: AVX-512-Dispatch-Gerüst + erster vektorisierter Kernel (add/gelu)

**Files:**
- Create: `depth-anything-rs/crates/da-kernels/src/dispatch.rs`
- Create: `depth-anything-rs/crates/da-kernels/src/simd_avx512.rs`
- Modify: `depth-anything-rs/crates/da-kernels/src/lib.rs`
- Test: `depth-anything-rs/crates/da-kernels/tests/simd_matches_scalar.rs`

**Interfaces:**
- Produces:
  - `pub struct Kernels { isa: Isa }` mit `pub fn Kernels::detect() -> Kernels` (einmal beim Start).
  - `pub enum Isa { Avx512, Avx2, Scalar }`, `pub fn Kernels::isa(&self) -> Isa`.
  - `pub fn Kernels::gelu(&self, x: &mut [f32])`, `pub fn Kernels::add(&self, dst: &mut [f32], src: &[f32])` — dispatchen auf SIMD oder Scalar. Ergebnis muss im Toleranzband des skalaren Zwillings liegen.

- [ ] **Step 1: Failing test — SIMD == Scalar auf Zufallsdaten**

```rust
// depth-anything-rs/crates/da-kernels/tests/simd_matches_scalar.rs
use da_kernels::{Kernels, scalar};

fn ramp(n: usize) -> Vec<f32> { (0..n).map(|i| (i as f32 * 0.017) - 3.0).collect() }

#[test]
fn gelu_simd_matches_scalar() {
    let k = Kernels::detect();
    let mut a = ramp(1000); let mut b = a.clone();
    k.gelu(&mut a);
    scalar::gelu(&mut b);
    for i in 0..a.len() { assert!((a[i]-b[i]).abs() < 1e-4, "i={i} simd={} scalar={}", a[i], b[i]); }
}

#[test]
fn add_simd_matches_scalar() {
    let k = Kernels::detect();
    let src = ramp(1000);
    let mut a = ramp(1000); let mut b = a.clone();
    k.add(&mut a, &src);
    scalar::add(&mut b, &src);
    assert_eq!(a, b);
}
```

- [ ] **Step 2: Fail bestätigen**

Run: `cd depth-anything-rs && cargo test -p da-kernels --test simd_matches_scalar`
Expected: FAIL — `Kernels` fehlt.

- [ ] **Step 3: Dispatch + AVX-512-Kernel implementieren**

`dispatch.rs`: `Kernels::detect()` via `is_x86_feature_detected!("avx512f")` → `Isa::Avx512`, sonst `avx2`, sonst `Scalar`. Die öffentlichen Methoden verzweigen einmal auf `self.isa`. `simd_avx512.rs` enthält `#[target_feature(enable = "avx512f")] unsafe fn gelu_avx512(...)` etc., jeweils mit `debug_assert!` auf Länge und Scalar-Tail für Rest < 16. Jede `unsafe fn` hat den skalaren Zwilling als Referenz (bereits in Task 5). `add` ist elementweise und triviale AVX-512-Ladung/Addition/Speicherung.

- [ ] **Step 4: Tests grün**

Run: `cd depth-anything-rs && cargo test -p da-kernels --test simd_matches_scalar`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add depth-anything-rs/crates/da-kernels
git commit -m "feat(da-kernels): runtime isa dispatch + avx512 gelu/add kernels"
```

---

### Task 9: q8_0-Vektor-Dot-Product (VNNI-Port)

Der teuerste Eigenbau. Port von ggmls `ggml_vec_dot_q8_0_q8_0` (AVX-512/VNNI `_mm512_dpbusd_epi32`-basiert): Aktivierungen werden pro Block nach int8 quantisiert, dann Block-Dot gegen die q8_0-Gewichte, Skalierung mit `d_a * d_w`.

**Files:**
- Create: `depth-anything-rs/crates/da-kernels/src/q8_0_dot.rs`
- Modify: `depth-anything-rs/crates/da-kernels/src/lib.rs`
- Test: `depth-anything-rs/crates/da-kernels/tests/q8_0_dot_matches_f32.rs`

**Interfaces:**
- Consumes: `da_gguf::{BlockQ8_0, QK8_0, dequantize_q8_0}`.
- Produces:
  - `pub fn quantize_row_q8_0(x: &[f32], out: &mut [BlockQ8_0])` — `x.len()==out.len()*32`.
  - `pub fn Kernels::gemm_q8_0(&self, m,n,k, a_q: &[BlockQ8_0], b_q: &[BlockQ8_0], c: &mut [f32])` — A[m×k], B[k×n] beide als q8_0-Zeilenblöcke (k Vielfaches von 32); C f32.
  - Skalarer Zwilling `scalar::gemm_q8_0(...)` als Oracle.

- [ ] **Step 1: Failing test — q8_0-GEMM ≈ f32-GEMM der dequantisierten Operanden**

```rust
// depth-anything-rs/crates/da-kernels/tests/q8_0_dot_matches_f32.rs
use da_kernels::{Kernels, scalar, quantize_row_q8_0};
use da_gguf::{BlockQ8_0, dequantize_q8_0, QK8_0};
use half::f16;

fn quantize_matrix(x: &[f32], rows: usize, k: usize) -> Vec<BlockQ8_0> {
    let mut out = vec![BlockQ8_0{ d: f16::from_f32(0.0), qs:[0;32] }; rows*(k/QK8_0)];
    for r in 0..rows {
        let blocks_per_row = k/QK8_0;
        quantize_row_q8_0(&x[r*k..(r+1)*k], &mut out[r*blocks_per_row..(r+1)*blocks_per_row]);
    }
    out
}

#[test]
fn q8_0_gemm_close_to_f32() {
    let (m,n,k) = (8, 8, 64);
    let a: Vec<f32> = (0..m*k).map(|i| ((i%17) as f32 - 8.0)*0.1).collect();
    let b: Vec<f32> = (0..k*n).map(|i| ((i%13) as f32 - 6.0)*0.1).collect();
    // B als q8_0 zeilenweise entlang k (also B^T-Blöcke): hier B ist k×n; wir quantisieren
    // die n Spalten als Zeilen -> transponieren zuerst.
    let mut bt = vec![0f32; n*k];
    for p in 0..k { for j in 0..n { bt[j*k+p] = b[p*n+j]; } }
    let aq = quantize_matrix(&a, m, k);
    let bq = quantize_matrix(&bt, n, k);
    let k_dev = Kernels::detect();
    let mut c = vec![0f32; m*n];
    k_dev.gemm_q8_0(m,n,k,&aq,&bq,&mut c);
    // Referenz: dequantisieren und f32-GEMM
    let mut a_de = vec![0f32; m*k]; dequantize_q8_0(&aq, &mut a_de);
    let mut bt_de = vec![0f32; n*k]; dequantize_q8_0(&bq, &mut bt_de);
    let mut b_de = vec![0f32; k*n];
    for j in 0..n { for p in 0..k { b_de[p*n+j] = bt_de[j*k+p]; } }
    let mut c_ref = vec![0f32; m*n]; scalar::gemm_f32(m,n,k,&a_de,&b_de,&mut c_ref);
    for i in 0..m*n { assert!((c[i]-c_ref[i]).abs() < 1e-2 + 1e-2*c_ref[i].abs(), "i={i} q8={} f32={}", c[i], c_ref[i]); }
}
```

- [ ] **Step 2: Fail bestätigen**

Run: `cd depth-anything-rs && cargo test -p da-kernels --test q8_0_dot_matches_f32`
Expected: FAIL.

- [ ] **Step 3: Implementieren (skalar zuerst, dann AVX-512-VNNI)**

`quantize_row_q8_0`: pro 32er-Block `amax = max|x|`, `d = amax/127`, `qs[j] = round(x[j]/d)`, `d` als f16. Skalarer `gemm_q8_0`: für jede (i,j) die k/32 Blockpaare akkumulieren: `sum += d_a*d_b * Σ (qa·qb)`. AVX-512-Variante nutzt `_mm512_dpbusd_epi32` über je 64 int8 (zwei Blöcke) mit i32-Akkumulator, dann `d_a*d_b`-Skalierung. Länge-Asserts + Scalar-Tail. Der skalare Pfad ist der Oracle des Tests.

- [ ] **Step 4: Tests grün (+ am Roofline denken)**

Run: `cd depth-anything-rs && cargo test -p da-kernels --test q8_0_dot_matches_f32`
Expected: PASS.

- [ ] **Step 5: Optimierungs-Log-Eintrag + Zwei-Iterationen-Regel**

Nach „schneller als scalar" die zwei Hypothesen versuchen (z. B. 2 Blöcke/Iteration entrollen; A-Quantisierung cachen statt pro GEMM neu). Ergebnis + evtl. „am Limit"-Begründung ins Log.

- [ ] **Step 6: Commit**

```bash
git add depth-anything-rs/crates/da-kernels
git commit -m "feat(da-kernels): q8_0 quantize + vnni dot-product gemm with scalar oracle"
```

---

### Task 10: Attention-Kernel (tiled, fused softmax) + RoPE2D

**Files:**
- Create: `depth-anything-rs/crates/da-kernels/src/attention.rs`
- Create: `depth-anything-rs/crates/da-kernels/src/rope.rs`
- Modify: `depth-anything-rs/crates/da-kernels/src/lib.rs`
- Test: `depth-anything-rs/crates/da-kernels/tests/rope_parity.rs`
- Test: `depth-anything-rs/crates/da-kernels/tests/attention_matches_naive.rs`

**Interfaces:**
- Produces:
  - `pub fn rope2d(x: &mut [f32], heads: usize, n: usize, head_dim: usize, pos_yx: &[i64], freq: f32)` — wendet 2D-RoPE in-place auf `[heads, n, head_dim]` an; `pos_yx` ist `[n*2]` (y,x). Muss `rope_out` aus den Dumps treffen.
  - `pub fn attention(q: &[f32], k: &[f32], v: &[f32], heads: usize, n: usize, head_dim: usize, out: &mut [f32])` — Scaled-Dot-Product-Attention pro Head (Skalierung `1/sqrt(head_dim)`), online-softmax; `out` = `[heads, n, head_dim]`.
  - Naive Referenz `attention_naive(...)` (gemm+softmax+gemm) als Oracle.

- [ ] **Step 1: Failing RoPE-Parity-Test gegen den Dump**

```rust
// depth-anything-rs/crates/da-kernels/tests/rope_parity.rs
use da_kernels::rope2d;
use da_parity::{Dumps, dumps_path, assert_parity};

#[test]
fn rope2d_matches_reference() {
    let (g, m) = (dumps_path("reference.gguf"), dumps_path("manifest.json"));
    if !g.exists() { eprintln!("[skip] no dumps"); return; }
    let d = Dumps::open(&g, &m).unwrap();
    let rin = d.reference("rope_in").unwrap();     // (1,1,4,64) -> heads=1,n=4,head_dim=64
    let rpos = d.reference("rope_pos").unwrap();    // (1,4,2) y,x als f32
    let rout = d.reference("rope_out").unwrap();
    let pos: Vec<i64> = rpos.data.iter().map(|&v| v as i64).collect();
    let mut x = rin.data.clone();
    rope2d(&mut x, 1, 4, 64, &pos, 100.0);
    assert_parity(&x, &rout.data, d.atol(), d.rtol(), "rope2d");
}
```

- [ ] **Step 2: Fail bestätigen**

Run: `cd depth-anything-rs && cargo test -p da-kernels --test rope_parity`
Expected: FAIL.

- [ ] **Step 3: RoPE2D implementieren** (gemäß `RotaryPositionEmbedding2D(frequency=100.0)`: head_dim wird in zwei Hälften y/x geteilt, je rotary über die Positionskoordinate). Danach Attention (naive Referenz zuerst, dann tiled online-softmax).

- [ ] **Step 4: Attention-Test (SIMD/tiled == naive)** — analog Task 8-Muster mit Zufallsdaten.

- [ ] **Step 5: Beide Tests grün**

Run: `cd depth-anything-rs && cargo test -p da-kernels --test rope_parity --test attention_matches_naive`
Expected: PASS (rope evtl. `[skip]` ohne Dumps).

- [ ] **Step 6: Commit**

```bash
git add depth-anything-rs/crates/da-kernels
git commit -m "feat(da-kernels): rope2d (dump-gated) + fused attention with naive oracle"
```

---

### Task 11: Conv2D (im2col + GEMM) + Upsampling

**Files:**
- Create: `depth-anything-rs/crates/da-kernels/src/conv.rs`
- Create: `depth-anything-rs/crates/da-kernels/src/resample.rs`
- Modify: `depth-anything-rs/crates/da-kernels/src/lib.rs`
- Test: `depth-anything-rs/crates/da-kernels/tests/conv_parity.rs`

**Interfaces:**
- Produces:
  - `pub fn conv2d(input: &[f32], in_c, ih, iw, weight: &[f32], out_c, kh, kw, stride, pad, bias: Option<&[f32]>, out: &mut [f32])` — NCHW, im2col+GEMM (nutzt den gewählten `Gemm`).
  - `pub fn conv_transpose2d(...)` — für die DPT-resize-Layer (ConvTranspose k4s4).
  - `pub fn bilinear_resize(input: &[f32], c, ih, iw, oh, ow, out: &mut [f32])`.
- Consumes: `gemm::Gemm`.

- [ ] **Step 1: Failing Conv-Parity gegen die Dump-Fixtures** (`convt0_in/out` = ConvTranspose k4s4 96→96; `convs3_in/out` = Conv k3s2p1 768→768; `proj0_in/out` = Conv 1×1 1536→96). Test lädt Eingabe, ruft Kernel, vergleicht gegen Ausgabe-Dump mit atol/rtol.

```rust
// depth-anything-rs/crates/da-kernels/tests/conv_parity.rs (Auszug für 1x1-Conv)
use da_kernels::conv::conv2d;
use da_parity::{Dumps, dumps_path, assert_parity};
#[test]
fn conv_1x1_proj0_matches_reference() {
    let (g,m)=(dumps_path("reference.gguf"),dumps_path("manifest.json"));
    if !g.exists() { eprintln!("[skip] no dumps"); return; }
    let d = Dumps::open(&g,&m).unwrap();
    let inp = d.reference("proj0_in").unwrap();   // (1,1536,16,16)
    let out_ref = d.reference("proj0_out").unwrap(); // (1,96,16,16)
    // weight/bias von projects[0] müssen aus dem Modell-GGUF kommen; falls hier nicht verfügbar,
    // wird der Test über die Engine-Ebene (Task 20) scharf geschaltet und hier nur die Form geprüft.
    assert_eq!(inp.shape.iter().product::<i64>(), 1*1536*16*16);
    let _ = (out_ref, conv2d as usize);
}
```
Hinweis: Die conv-Gewichte liegen im Modell-GGUF, nicht im Dump. Voll scharf wird die Conv-Parity in Task 20 (DPT-Head über die Engine). Diese Task testet die Conv-Mechanik gegen ein selbst erzeugtes Referenzergebnis (naive Conv) und die Formtreue gegen die Dumps.

- [ ] **Step 2–5:** Fail bestätigen → im2col+GEMM + naive-Conv-Oracle implementieren → Tests grün → Commit.

```bash
git add depth-anything-rs/crates/da-kernels
git commit -m "feat(da-kernels): conv2d/conv_transpose2d via im2col+gemm and bilinear resize"
```

---

# M4 — `da-graph` statischer Executor

### Task 12: Tensor-Typ + Buffer-Arena

**Files:**
- Create: `depth-anything-rs/crates/da-graph/Cargo.toml`
- Create: `depth-anything-rs/crates/da-graph/src/lib.rs`
- Create: `depth-anything-rs/crates/da-graph/src/tensor.rs`
- Create: `depth-anything-rs/crates/da-graph/src/arena.rs`
- Test: `depth-anything-rs/crates/da-graph/tests/arena_reuse.rs`
- Modify: `depth-anything-rs/Cargo.toml` — `"crates/da-graph"` zu `members` hinzufügen.

**Interfaces:**
- Produces:
  - `pub struct Shape(pub Vec<usize>)` mit `pub fn numel(&self) -> usize`.
  - `pub struct TensorId(pub usize)`.
  - `pub struct Arena` — vorab geplanter, wiederverwendeter f32-Speicher; `pub fn Arena::plan(sizes: &[usize], lifetimes: &[(usize,usize)]) -> Arena` (lifetimes = (first_use, last_use) je Tensor); `pub fn buf(&mut self, id: TensorId) -> &mut [f32]`.
  - Invariante: nach `plan` erfolgen im Forward keine Allokationen.

- [ ] **Step 1: Failing test — überlappende Lebenszeiten teilen keinen Puffer, disjunkte schon**

```rust
// depth-anything-rs/crates/da-graph/tests/arena_reuse.rs
use da_graph::arena::Arena;
#[test]
fn disjoint_lifetimes_reuse_memory() {
    // t0 lebt [0,1], t1 lebt [2,3] -> dürfen denselben Offset teilen.
    let a = Arena::plan(&[100, 100], &[(0,1),(2,3)]);
    assert_eq!(a.total_floats(), 100, "disjoint tensors should share buffer");
}
#[test]
fn overlapping_lifetimes_get_separate_memory() {
    let a = Arena::plan(&[100, 100], &[(0,3),(1,2)]);
    assert_eq!(a.total_floats(), 200);
}
```

- [ ] **Step 2–5:** Fail → Greedy-Offset-Planer (nach `last_use` freigeben, Offsets wiederverwenden) → Tests grün → Commit `feat(da-graph): tensor shape + lifetime-planned buffer arena`.

---

### Task 13: Graph-Beschreibung + Executor + Backend-Trait

**Files:**
- Create: `depth-anything-rs/crates/da-graph/src/graph.rs`
- Create: `depth-anything-rs/crates/da-graph/src/backend.rs`
- Create: `depth-anything-rs/crates/da-graph/src/cpu_backend.rs`
- Modify: `depth-anything-rs/crates/da-graph/src/lib.rs`
- Test: `depth-anything-rs/crates/da-graph/tests/graph_runs_linear.rs`

**Interfaces:**
- Produces:
  - `pub enum Op { Gemm{a:TensorId,b:TensorId,m:usize,n:usize,k:usize}, AddBias{x:TensorId,bias:TensorId,rows:usize,cols:usize}, Gelu{x:TensorId}, LayerNorm{x:TensorId,g:TensorId,b:TensorId,rows:usize,cols:usize,eps:f32}, Attention{...}, Conv2d{...}, ... }`
  - `pub struct Graph { pub ops: Vec<Op>, pub inputs: Vec<TensorId>, pub outputs: Vec<TensorId>, ... }` mit Builder-API `Graph::builder()`.
  - `pub trait Backend { fn execute(&self, op: &Op, arena: &mut Arena, weights: &Weights); }`
  - `pub struct CpuBackend { kernels: da_kernels::Kernels, gemm: Box<dyn da_kernels::gemm::Gemm> }` impl `Backend`.
  - `pub struct Plan { graph: Graph, arena_layout: ArenaLayout }` mit `pub fn Graph::compile(&self) -> Plan` (Lebenszeiten ableiten, Arena planen) und `pub fn Plan::run(&self, backend: &dyn Backend, inputs: &[&[f32]], weights: &Weights) -> Vec<Vec<f32>>`.
  - `pub struct Weights` — Name→Tensor-Map aus dem GGUF (f32 oder q8_0-Blocks).

- [ ] **Step 1: Failing test — ein Mini-Graph (Gemm→AddBias→Gelu) rechnet dasselbe wie die Kernel direkt**

```rust
// depth-anything-rs/crates/da-graph/tests/graph_runs_linear.rs
use da_graph::{Graph, CpuBackend, Weights};
#[test]
fn linear_gelu_graph_matches_manual() {
    // baue Graph: y = gelu(x·W + b), vergleiche mit da_kernels direkt.
    // (konkrete Builder-Aufrufe; Shapes m=2,k=3,n=2)
    // ... siehe Implementierung; Assertion: max|d| < 1e-5
}
```

- [ ] **Step 2–4:** Fail → Graph/Plan/CpuBackend implementieren (compile leitet (first,last)-Uses je Tensor ab, plant Arena; run füllt Inputs, iteriert Ops, dispatcht an Backend). „Null Allokationen im Forward" via Test, der `run` zweimal auf demselben `Plan` laufen lässt und Ergebnisgleichheit prüft. → grün.

- [ ] **Step 5: Commit** `feat(da-graph): static op graph, compile-to-plan, cpu backend`

---

# M5 — `da-engine` Backbone

### Task 14: Modell-Konfiguration aus GGUF-Metadaten

**Files:**
- Create: `depth-anything-rs/crates/da-engine/Cargo.toml`
- Create: `depth-anything-rs/crates/da-engine/src/lib.rs`
- Create: `depth-anything-rs/crates/da-engine/src/config.rs`
- Test: `depth-anything-rs/crates/da-engine/tests/config_from_model.rs`
- Modify: `depth-anything-rs/Cargo.toml` — `"crates/da-engine"` zu `members` hinzufügen.

**Interfaces:**
- Consumes: `da_gguf::GgufFile`.
- Produces:
  - `pub struct ModelConfig { pub arch: String, pub patch_size: u32, pub image_size: u32, pub embed_dim, pub depth, pub num_heads, pub head_dim, pub mlp_hidden: u32, pub num_register: u32, pub rope_start, pub qknorm_start: i32, pub rope_freq: f32, pub ln_eps: f32, pub out_layers: Vec<i32>, pub head_features: u32, pub head_max_depth: f32, pub img_mean: [f32;3], pub img_std: [f32;3], ... }`
  - `pub fn ModelConfig::from_gguf(f: &GgufFile) -> Result<ModelConfig, EngineError>` — liest die `depthanything3.*`-Keys; unbekannte arch → `Err(EngineError::UnsupportedModel)`.
  - `pub enum EngineError` (thiserror).

- [ ] **Step 1: Failing test** (lädt Modell falls vorhanden, prüft arch + embed_dim/depth plausibel; ohne Modell `[skip]`).
- [ ] **Step 2–5:** Fail → Keys via `meta_*`-Getter lesen (Namen wörtlich aus `da_gguf_keys.h`) → grün → Commit `feat(da-engine): model config from gguf metadata`.

---

### Task 15: Preprocessing (Resize + Normalisierung)

**Files:**
- Create: `depth-anything-rs/crates/da-engine/src/preprocess.rs`
- Test: `depth-anything-rs/crates/da-engine/tests/preprocess_parity.rs`

**Interfaces:**
- Produces: `pub fn preprocess(raw_hwc_u8: &[u8], h: usize, w: usize, cfg: &ModelConfig, out_nchw: &mut Vec<f32>) -> (usize, usize)` — resize gemäß `img.resize_mode/target`, normalisiere mit `img_mean/std`, Ausgabe NCHW; gibt (H,W) nach Resize zurück.
- Gate: gegen `input_image` (und `raw_image`) aus den Dumps — `raw_image` (224×224×3 HWC, 0..255) durch `preprocess` muss `input_image` treffen.

- [ ] **Step 1: Failing parity test** (`raw_image` → preprocess → vs `input_image`, atol/rtol).
- [ ] **Step 2–5:** Fail → implementieren (bilinear/bicubic je nach resize_mode) → grün → Commit `feat(da-engine): preprocessing gated against input_image dump`.

---

### Task 16: Patch-Embedding + CLS/Register-Token + Positional-Embedding-Cache

**Files:**
- Create: `depth-anything-rs/crates/da-engine/src/patch_embed.rs`
- Create: `depth-anything-rs/crates/da-engine/src/pos_embed.rs`
- Test: `depth-anything-rs/crates/da-engine/tests/pos_embed_parity.rs`

**Interfaces:**
- Produces:
  - `pub fn patch_embed(img_nchw, cfg, weights, out_tokens: &mut Vec<f32>)` — Conv-patchify → Tokens `[n_patches, embed_dim]`.
  - `pub struct PosEmbedCache { by_resolution: HashMap<(usize,usize), Vec<f32>> }` mit `pub fn get_or_build(&mut self, h, w, cfg, weights) -> &[f32]` — die ~95-ms-Lektion: bikubisch interpolierte Pos-Embeds werden pro Auflösung einmal gebaut und gecacht.
  - Kombinierte Funktion `prepare_tokens(...)` → Tokens nach CLS-Prepend + Pos-Embed-Add; Gate gegen `pos_embed_added`.

- [ ] **Step 1: Failing parity test** gegen `pos_embed_added`.
- [ ] **Step 2–5:** Fail → implementieren (Cache-Key = (h,w)) → grün → Commit `feat(da-engine): patch embed + cached positional embeddings (pos_embed_added parity)`.

---

### Task 17: ViT-Block + Backbone-Forward (feat-Parity)

**Files:**
- Create: `depth-anything-rs/crates/da-engine/src/vit_block.rs`
- Create: `depth-anything-rs/crates/da-engine/src/backbone.rs`
- Test: `depth-anything-rs/crates/da-engine/tests/backbone_parity.rs`

**Interfaces:**
- Produces:
  - `pub fn vit_block(tokens: &mut [f32], n, cfg, layer_idx, weights, graph_backend)` — LN→Attn(+RoPE ab rope_start, +QK-Norm ab qknorm_start)→Residual→LN→MLP(GELU)→Residual.
  - `pub struct Backbone` mit `pub fn forward(&self, tokens, out_layers: &[i32]) -> Vec<Vec<f32>>` — sammelt Features an den `out_layers` (5,7,9,11 für BASE).
- Gate: `feat_{5,7,9,11}` und `cam_token_{5,7,9,11}` aus den Dumps.

- [ ] **Step 1: Failing parity test** — Backbone-Forward auf `input_image` → `feat_11` (und die anderen out-layers) vs. Dumps.
- [ ] **Step 2: Fail bestätigen.**
- [ ] **Step 3: Implementieren** — ViT-Block über die `da-graph`-Ops zusammensetzen; QK-Norm und RoPE-Start aus config; register-Token beachten.
- [ ] **Step 4: Parity grün** (der wichtigste Meilenstein von M5).
- [ ] **Step 5: Geschwindigkeit** — `vit_block` gegen `baseline.json` benchmarken, Zwei-Iterationen-Regel, Optimierungs-Log.
- [ ] **Step 6: Commit** `feat(da-engine): vit block + backbone forward (feat_5/7/9/11 parity)`.

---

# M6 — `da-engine` DPT-Head & Pose

### Task 18: DPT-Head (Depth + Confidence)

**Files:**
- Create: `depth-anything-rs/crates/da-engine/src/dpt_head.rs`
- Create: `depth-anything-rs/crates/da-engine/src/uv_embed.rs`
- Test: `depth-anything-rs/crates/da-engine/tests/dpt_parity.rs`

**Interfaces:**
- Produces:
  - `pub fn uv_embed(h, w, dim, out: &mut Vec<f32>)` — UV-Grid-Positional-Embedding (gecacht wie pos_embed); Gate gegen `uv_embed_64`.
  - `pub fn dpt_head(feats: &[Vec<f32>], h, w, cfg, weights) -> DepthOut` mit `pub struct DepthOut { pub depth: Vec<f32>, pub conf: Vec<f32>, pub h: usize, pub w: usize }`.
- Gate: `head_stage{0..3}`, `head_fused`, `head_depth`, `head_depth_conf`.

- [ ] **Step 1: Failing parity test** — Zwischenstufen (`head_stage0..3`, `head_fused`) und Endausgabe (`head_depth`, `head_depth_conf`) aus den gedumpten `feat_*` als Eingabe.
- [ ] **Step 2: Fail bestätigen.**
- [ ] **Step 3: Implementieren** — resize_layers (ConvTranspose/Conv/Identity), projects (1×1), reassemble, fusion-blocks, output_conv; `depth = exp(...)`, `conf = expp1(...)` laut Dump-Asserts (`depth>0`, `conf>=1`).
- [ ] **Step 4: Parity grün.**
- [ ] **Step 5: Speed + Log** (der DPT-Head ist zeitlich relevant — UV-Embed-Cache ist hier die erste Optimierungshypothese).
- [ ] **Step 6: Commit** `feat(da-engine): dpt head depth+conf (head_depth parity)`.

---

### Task 19: Pose-Head (Extrinsics + Intrinsics)

**Files:**
- Create: `depth-anything-rs/crates/da-engine/src/pose.rs`
- Test: `depth-anything-rs/crates/da-engine/tests/pose_parity.rs`

**Interfaces:**
- Produces: `pub fn cam_pose(cam_token: &[f32], cfg, weights) -> PoseOut` mit `pub struct PoseOut { pub extrinsics: [f32;12], pub intrinsics: [f32;9], pub pose_enc: [f32;9] }`.
- Gate: `cam_token_in` (== `cam_token_11`), `pose_enc`, `extrinsics`, `intrinsics`.

- [ ] **Step 1: Failing parity test** — `cam_token_in` → cam_dec → `pose_enc` (9) → `extrinsics` (3×4) + `intrinsics` (3×3).
- [ ] **Step 2–4:** Fail → cam_dec-MLP + pose-enc→Matrix-Umrechnung (quaternion/translation → w2c; fov → K) → grün.
- [ ] **Step 5: Commit** `feat(da-engine): camera pose head (extrinsics/intrinsics parity)`.

---

### Task 20: Engine-Fassade (End-to-End Depth+Pose)

**Files:**
- Create: `depth-anything-rs/crates/da-engine/src/engine.rs`
- Modify: `depth-anything-rs/crates/da-engine/src/lib.rs`
- Test: `depth-anything-rs/crates/da-engine/tests/e2e_native.rs`

**Interfaces:**
- Produces:
  - `pub struct Engine { cfg: ModelConfig, weights: Weights, backbone: Backbone, pos_cache: PosEmbedCache, uv_cache: ... , backend: CpuBackend }`
  - `pub fn Engine::load(path: &Path, quant_prefer: QuantPref) -> Result<Engine, EngineError>`
  - `pub fn Engine::infer(&mut self, raw_hwc_u8: &[u8], h: usize, w: usize) -> InferOut` mit `pub struct InferOut { pub depth: Vec<f32>, pub conf: Vec<f32>, pub h: usize, pub w: usize, pub extrinsics: [f32;12], pub intrinsics: [f32;9] }`.
- Gate: End-to-End gegen `head_depth` + `extrinsics` + `intrinsics` aus dem Dump-Input `raw_image`.

- [ ] **Step 1: Failing e2e parity test** — `raw_image` durch `Engine::infer` → depth vs `head_depth`, pose vs `extrinsics`/`intrinsics`.
- [ ] **Step 2–4:** Fail → Fassade verdrahten (preprocess→backbone→dpt+pose) → grün.
- [ ] **Step 5: Commit** `feat(da-engine): end-to-end engine facade (e2e depth+pose parity)`.

---

# M7 — `da-cli` & E2E-Gate

### Task 21: CLI `infer`

**Files:**
- Create: `depth-anything-rs/crates/da-cli/Cargo.toml`
- Create: `depth-anything-rs/crates/da-cli/src/main.rs`
- Create: `depth-anything-rs/crates/da-cli/src/infer.rs`
- Test: `depth-anything-rs/crates/da-cli/tests/cli_smoke.rs`
- Modify: `depth-anything-rs/Cargo.toml` — `"crates/da-cli"` zu `members` hinzufügen (letztes Member — Workspace ist damit vollständig).

**Interfaces:**
- Produces: Binary `da` mit `da infer --model <gguf> --image <png> --out-depth <pfm|png> --out-pose <json>`. Pose-JSON: `{ "extrinsics": [[..]], "intrinsics": [[..]] }`.

- [ ] **Step 1: Failing smoke test** — auf einem kleinen Bild läuft `da infer` und schreibt eine nicht-leere Depth-Datei + valides Pose-JSON (Test via `assert_cmd` oder direktem Aufruf der `infer`-Funktion; ohne Modell `[skip]`).
- [ ] **Step 2–4:** Fail → clap-CLI + `image`-Laden + `Engine::infer` + PFM/PNG-Schreiber + serde-JSON → grün.
- [ ] **Step 5: Commit** `feat(da-cli): infer subcommand (depth + pose output)`.

---

### Task 22: CLI `bench` + E2E-Latenz-Gate gegen C++

**Files:**
- Create: `depth-anything-rs/crates/da-cli/src/bench.rs`
- Modify: `depth-anything-rs/crates/da-cli/src/main.rs`
- Create: `depth-anything-rs/scripts/compare_e2e.sh`

**Interfaces:**
- Produces: `da bench --model <gguf> --image <png> --repeat N --warmup W` → druckt Median/p95 ms (dasselbe Protokoll wie `../benchmarks/BENCHMARK.md`). `compare_e2e.sh` läuft die C++-CLI und `da bench` auf demselben Bild/derselben Auflösung und stellt die Zahlen nebeneinander.

- [ ] **Step 1: Failing test** — `da bench --repeat 2` gibt eine parsebare `median_ms=...`-Zeile aus (ohne Modell `[skip]`).
- [ ] **Step 2–4:** Fail → Timing-Schleife (warmup + repeat, Median/p95) → grün.
- [ ] **Step 5: `compare_e2e.sh`** ruft `../build/da-cli` (C++) und die Rust-`da` und druckt beide Medianwerte + Faktor.
- [ ] **Step 6: E2E-Latenz messen + Optimierungs-Log** — den Gesamtfaktor gegen die 346-ms-Baseline eintragen; offene Komponenten-Hebel notieren.
- [ ] **Step 7: Commit** `feat(da-cli): bench subcommand + e2e comparison script`.

---

### Task 23: `baseline.json`-Generator auf der C++-Seite (additiv)

Damit die criterion-Benchmarks gegen echte C++-Zahlen vergleichen, brauchen wir eine `baseline.json`. Die C++-Test-Executables existieren; wir ergänzen ein kleines Timing-Target additiv unter `../tests/` (ohne bestehende Dateien umzustrukturieren).

**Files:**
- Create: `../tests/bench_components.cpp` (additiv; misst vit_block, dpt_head, cam_pose isoliert)
- Modify: `../tests/CMakeLists.txt` (nur ein `add_executable` hinzufügen)
- Create: `depth-anything-rs/scripts/gen_baseline.sh` (baut C++, führt bench_components aus, schreibt `depth-anything-rs/baseline.json`)

**Interfaces:**
- Produces: `depth-anything-rs/baseline.json` mit `{ "vit_block_ms": ..., "dpt_head_ms": ..., "cam_pose_ms": ..., "e2e_ms": ..., "quant": "f32|q8_0", "cpu": "...", "resolution": "504x336" }`.

- [ ] **Step 1:** `bench_components.cpp` schreiben (nutzt `Engine`/Komponenten wie die vorhandenen Tests, misst mit `std::chrono`, druckt JSON nach stdout).
- [ ] **Step 2:** `add_executable(bench_components ...)` in `../tests/CMakeLists.txt` ergänzen.
- [ ] **Step 3:** `gen_baseline.sh` schreiben (cmake build + run + redirect nach `baseline.json`).
- [ ] **Step 4:** Laufen lassen (falls C++-Toolchain/Modell vorhanden) und `baseline.json` einchecken; sonst `[skip]` dokumentieren.
- [ ] **Step 5: Commit** `feat(bench): additive c++ component timing target + baseline.json generator`.

---

## Self-Review (Autor)

**Spec-Coverage** (Spec §2–§8 → Task):
- §2 Scope (S/B/L, Depth+Conf+Pose, f32/f16/q8_0) → Tasks 3,4,14,17–20. q4_k/q5_k/q6_k bewusst nicht → Global Constraints.
- §4 Crate-Struktur (5+1) → Tasks 1 (da-gguf), 2 (da-parity), 5 (da-kernels), 12 (da-graph), 14 (da-engine), 21 (da-cli). Alle sechs Crates angelegt. ✓
- §4 „Kernel kennen nur Slices" → Task 5/6/8/9/10/11 (freie Funktionen über Slices). ✓
- §4 „Graph dumm & statisch, Buffer-Arena, null Forward-Allokationen" → Tasks 12,13. ✓
- §4 Backend-Trait offen für GPU → Task 13 (`Backend`-Trait, eine CPU-Impl). ✓
- §4 da-gguf mmap → Task 1. ✓
- §5.1 Kernel-Liste (GEMM, q8_0, Attention, Fusion, Conv, Upsampling) → Tasks 6,9,10,11 + Fusion in 6/GemmWithEpilogue. ✓
- §5.2 Geometrie-Caching (Pos-Embed, UV, im2col-Indizes) → Tasks 16,18. ✓
- §5.2 Preprocessing zur Engine → Task 15. ✓
- §5.3 CLI infer+bench → Tasks 21,22. ✓
- §6 Parity-Flow (gleiche Dumps/Toleranzen) → Task 2 + jede Parity-Task. ✓
- §6.2 baseline.json → Task 23. ✓
- §6.3 Zwei-Iterationen-Regel + Optimierungs-Log → Task 6 legt Log an; Tasks 9,17,18,22 tragen ein. ✓
- §6.1 „kein Bit-für-Bit mit C++, Toleranz gegen PyTorch" → Task 2 (atol/rtol aus Manifest). ✓
- §7 Fehlerbehandlung (Result an Rändern, keiner im Forward, unsafe gekapselt) → Global Constraints + Tasks 8,9. ✓
- §8 Testing-Ebenen (parity, criterion, e2e, scalar-twin) → Tasks 2,6,20,22 + jeder SIMD-Task. ✓
- §9 Epilogen-Fusions-Prüfpunkt → Task 6 Step 3/6. ✓

**Platzhalter-Scan:** Task 11 Step 2–5 und einige M4–M7-Tasks fassen die TDD-Standardschritte (Fail→Impl→Grün→Commit) zusammen statt jeden Codeblock auszuschreiben — bewusst, weil das Muster in Tasks 1–10 vollständig etabliert ist und die Interfaces/Gates dort exakt benannt sind. Kein „TBD"/„TODO"; jede Task hat konkrete Datei-Pfade, Interface-Signaturen und das zu treffende Dump-Tensor-Gate.

**Typ-Konsistenz:** `TensorF32{shape,data}` (Task 1) durchgängig; `BlockQ8_0{d:f16,qs:[i8;32]}` (Task 4) in 9 wiederverwendet; `Gemm`-Trait (Task 6) in 11/13 konsumiert; `Kernels` (Task 8) in 9/10 erweitert; `ModelConfig` (14) in 15–20; `Weights` (13) in 13/14/18–20; `DepthOut`/`PoseOut`/`InferOut` konsistent.

**Scope-Check:** Ein zusammenhängendes Subsystem (eine Engine), sinnvoll als ein Plan mit 24 Tasks in 8 Meilensteinen. GPU/Server/q4_k explizit ausgeklammert (v2).
