# Nächste Schritte auf dem x86-64-Rechner

Dieses Repo (inkl. `depth-anything-rs/`) wurde komplett auf einer arm64-Maschine (Apple Silicon)
gebaut. Alle 24 Plan-Tasks sind fertig, 137 Tests grün — aber zwei Dinge waren dort strukturell
unmöglich:

1. **AVX-512F/BW/VNNI-Kernel numerisch verifizieren** (Tasks 8/9) — bisher nur per
   `cargo check --target x86_64-unknown-linux-gnu` (Compile-Check) und dem skalaren
   Oracle-Fallback getestet, nie tatsächlich auf echter Hardware gelaufen.
2. **Echte Parity- und Latenz-Zahlen gegen die C++-Referenz** — alle Parity-Tests skippen
   bisher, weil kein Modell und keine Dumps vorlagen.

Auf dem x86-Rechner kann beides live gehen. Reihenfolge unten.

---

## 0. Voraussetzungen

- x86-64 CPU mit AVX-512F/BW/VNNI (Cascade Lake oder neuer für VNNI; sonst fällt der Code
  automatisch auf AVX2/skalar zurück — dann testet das nur den Fallback, nicht den VNNI-Pfad)
- Rust stable (`rustup show` sollte `stable` zeigen; kein Nightly nötig)
- CMake + C++17-Compiler (für die C++-Referenz-Engine)
- Python 3 mit `torch`, `huggingface_hub`, `numpy` (für Modell-Download/-Konvertierung und Dumps)

```bash
cd depth-anything.cpp-master   # Repo-Root (git-Root)
```

## 1. Modell besorgen und nach GGUF konvertieren

```bash
python3 scripts/download_model.py --repo depth-anything/DA3-BASE --out models/DA3-BASE
python3 scripts/convert_da3_to_gguf.py --model models/DA3-BASE --output models/depth-anything-base-f32.gguf
```

Für weitere Größen (Small/Large/Giant) einfach `--repo`/`--model`/`--output` anpassen
(`depth-anything/DA3-SMALL`, `DA3-LARGE`, `DA3-GIANT`). Für v1 reicht **DA3-BASE** — das ist das
Modell, gegen das der gesamte Rust-Plan entwickelt und mit der echten C++-Quelle abgeglichen wurde.

## 2. C++-Referenz-Dumps erzeugen

```bash
python3 scripts/dump_reference.py
```

Erzeugt `dumps/reference.gguf` + `dumps/manifest.json` (Toleranzen atol=rtol=2e-3) — das ist genau
das, was `depth-anything-rs`' Parity-Tests (`input_image`, `feat_5/7/9/11`, `head_depth`,
`extrinsics`, `intrinsics`, `pos_embed_added`, `uv_embed_64`, …) erwarten.

## 3. C++-Projekt bauen (für baseline.json + compare_e2e.sh)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Falls das ggml-Submodule leer ist: `git submodule update --init --recursive` zuerst.

## 4. Rust-Parity-Tests jetzt live laufen lassen

```bash
cd depth-anything-rs
cargo test --workspace
```

Alle bisher `[skip]`-markierten Tests (u. a. `backbone_parity`, `dpt_parity`, `pose_parity`,
`e2e_native::engine_matches_reference_depth_and_pose`, `preprocess_parity`, `pos_embed_parity`)
sollten jetzt gegen `../dumps/reference.gguf` + `../models/depth-anything-base-f32.gguf` laufen.

**Wenn hier etwas fehlschlägt**, zuerst diese beiden Stellen prüfen (im finalen Review als
größte Restrisiken dokumentiert):

- `crates/da-engine/src/engine.rs::weights_from_gguf` — die 2D-Gewichts-Transposition (Fix in
  Commit `5e32def`) wurde nur gegen das Python-Export-Skript verifiziert, nie gegen echte
  Inferenz-Zahlen.
- `crates/da-engine/src/backbone.rs` — `alt_start`/`cat_token`-Werte wurden aus Skript-Text
  erschlossen (Commit `9b7500a`), nie direkt aus einer echten GGUF-Datei gelesen. Falls die
  Backbone-Parity abweicht: `python3 -c "..."` oder ein kleines Rust-Snippet, das
  `depthanything3.vit.alt_start`/`depthanything3.vit.cat_token` aus
  `models/depth-anything-base-f32.gguf` ausliest, ist der erste Debugging-Schritt.

## 5. AVX-512/VNNI-Kernel numerisch verifizieren

```bash
cargo test -p da-kernels
```

Auf x86-64 mit AVX-512F/BW/VNNI wählt `Kernels::detect()` jetzt automatisch `Isa::Avx512` statt
`Isa::Scalar` — die bisher nur compile-geprüften Kernel (`simd_avx512.rs`: `gelu_avx512`,
`add_avx512`, `gemm_q8_0_avx512`) laufen jetzt wirklich und werden gegen den skalaren Oracle
verglichen (Toleranz siehe jeweiligen Test). Ergebnis unbedingt in
`depth-anything-rs/docs/optimization-log.md` unter einem neuen Abschnitt festhalten (Muster: siehe
bestehende Einträge zu Task 8/9).

## 6. Benchmarks + baseline.json + E2E-Vergleich

```bash
# GEMM-Benchmark (Milestone-1-Entscheidung faer vs. scalar, jetzt auf echter Hardware)
cargo bench -p da-kernels --bench gemm_bench

# C++-seitige baseline.json erzeugen (Task 23 hat das Tooling gebaut, aber nie laufen lassen können)
bash scripts/gen_baseline.sh

# Rust-CLI bauen und E2E-Latenz gegen die C++-CLI vergleichen (Task 22's Tooling)
cargo build --release -p da-cli
bash scripts/compare_e2e.sh --model ../models/depth-anything-base-f32.gguf --image <ein-testbild.png>
```

`compare_e2e.sh` erwartet die C++-Binary unter `../build/examples/cli/da3-cli` (oder
`../build/da3-cli` — beide Pfade werden probiert).

## 7. Weiter optimieren — Zwei-Iterationen-Regel

Der Plan (`docs/plans/2026-07-18-rust-engine-v1.md`, Spec §6.3) verlangt: sobald eine Komponente
schneller als die C++-Baseline ist, zwei benannte Optimierungshypothesen ausprobieren, bevor man
weiterzieht (Ausnahme: nachweislich am Roofline-Limit — dann Begründung ins Log statt Zeit zu
verschwenden). Offene Hebel, die im finalen Review + im Optimierungs-Log bereits identifiziert
sind:

- **AVX2-Zwischenstufe fehlt** — `Kernels::detect()` kennt `Isa::Avx2`, aber `gelu`/`add` haben nur
  einen AVX-512-Fast-Path und fallen sonst direkt auf skalar zurück (Task 8's Minor-Finding).
- **`vit_block.rs`/`dpt_head.rs` bauen pro Op einen frischen Mini-`Graph`/`Plan`** statt einen
  einzigen kompilierten Plan mit vorgeplanter Arena für den ganzen Forward-Pass zu nutzen — verstößt
  gegen Spec §4 ("null Forward-Allokationen"), ist aber korrekt. Größter Architektur-Hebel für v2
  (siehe finale Review, Punkt I1) — braucht neue `da-graph`-Ops (`ConvTranspose2d`,
  `ResizeBilinearAc`) und eine einmalig kompilierte `Plan` für Backbone+Head zusammen.
- **q8_0/VNNI-GEMM wird von keinem Forward-Pfad genutzt** — `vit_block.rs`/`dpt_head.rs`/`pose.rs`
  nutzen ausschließlich `Weights::get_f32`; `Kernels::gemm_q8_0` (Task 9) und
  `Weights::get_q8_0`/`insert_q8_0` existieren, sind aber toter Code bis eine Komponente sie
  tatsächlich aufruft. `QuantPref` in `Engine::load` ist aktuell ein dokumentierter No-Op.
- **`baseline.json`-Zahlen fehlen noch im Optimierungs-Log** — Task 6/9/17/18/22 haben alle
  "Baseline noch nicht verfügbar" vermerkt; nach Schritt 6 oben die realen Zahlen nachtragen und
  die Go/No-Go-Entscheidungen (faer vs. tract-linalg, etc.) mit echten Zahlen bestätigen oder
  revidieren.

---

## Copy-paste-Prompt für Claude Code auf dem x86-Rechner

```
Ich habe gerade das depth-anything.cpp-master-Repo (inkl. depth-anything-rs/) von
https://github.com/hoppworks/depth-anything.cpp-master geklont. Lies zuerst
depth-anything-rs/docs/NEXT_STEPS_X86.md komplett, dann depth-anything-rs/.superpowers/sdd/progress.md
(das Ledger der bisherigen Subagent-Driven-Development-Session) für vollen Kontext, was bereits
gebaut und wie es verifiziert wurde. Führe dann die Schritte 1-6 aus NEXT_STEPS_X86.md aus:
Modell laden, GGUF konvertieren, C++-Referenz-Dumps erzeugen, C++-Projekt bauen, Rust-Parity-Tests
laufen lassen (bisher übersprungene Tests sollten jetzt live gehen), AVX-512/VNNI-Kernel gegen
echte Hardware verifizieren, Benchmarks + baseline.json + E2E-Vergleich erzeugen. Bei jedem Fehler:
nicht raten, sondern gegen die echte C++-Quelle unter ../src/ gegenchecken (Muster aus der bisherigen
Session: fast jede Unklarheit wurde durch Lesen des passenden ../src/*.cpp-Files aufgelöst, nicht
durch Annahmen). Danach: die "Zwei-Iterationen-Regel" (Spec §6.3, docs/plans/2026-07-18-rust-engine-v1.md)
auf die in NEXT_STEPS_X86.md Abschnitt 7 gelisteten offenen Hebel anwenden, docs/optimization-log.md
mit echten Zahlen aktualisieren. Nutze Subagent-Driven Development (frischer Implementer-Subagent pro
Schritt + unabhängiger Reviewer) wie in der Vorgänger-Session, und hol dir bei echten Unklarheiten
einen Opus-Agenten mit hohem Reasoning-Aufwand als Advisor dazu statt zu raten.
```
