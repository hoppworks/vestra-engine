# Design: Rust-Rebuild der depth-anything Engine (v1)

**Datum:** 2026-07-18
**Status:** Freigegeben (Brainstorming abgeschlossen)
**Vorarbeit:** Deep-Research-Report zur Rust-Rebuild-Performance-Strategie (`scratchpad/rust-rebuild-performance-research.md`)

**Verortung:** Das gesamte Rust-Projekt lebt self-contained unter `depth-anything-rs/` im Repo-Root
(eigenes Cargo-Workspace, eigene `docs/`). Es hält sich bewusst **nicht** an die Ordner-Konventionen
des C++-Repos und ist so geschnitten, dass es mit einem Handgriff (ein Ordner) herausgetrennt werden
kann. Die einzige Verbindung nach außen ist **lesend**: der parity-getriebene Dev-Flow konsumiert die
vom C++-Repo erzeugten Dump-Dateien unter `../dumps/` (relativ zum Rust-Root).

---

## 1. Ziel & Kontext

Die bestehende `depth-anything.cpp` (C++17/ggml, from-scratch-Port von ByteDance Depth Anything 3)
wird komplett in Rust neu gebaut. Der Neubau ist kein Selbstzweck, sondern muss die C++/ggml-Engine
messbar schlagen — der Sprachwechsel allein bringt ~0 %, alle Gewinne kommen aus Dingen, die eine
maßgeschneiderte Engine tun kann und ggml als General-Purpose-Framework nicht tut.

### Performance-Ziele (aus dem Research-Report)

| Ziel | Erwartung | Herkunft des Gewinns |
|------|-----------|----------------------|
| CPU-Einzelbild-Latenz | +10–30 % ggü. C++/ggml | Statischer Graph, Speicherplanung, Kernel-Fusion |
| Durchsatz / Multi-Image | 2–4× | Batching, Pipelining, persistente Threadpools (v2+) |
| GPU | Parität (~47 ms) | Erst v2 — CubeCL/CUDA-Backend |
| Speicher & Binärgröße | klar kleiner | Kein Framework-Ballast, statische Planung |

Referenz-Baseline (C++/ggml, AMD Ryzen 9 9950X3D, 16 Threads, 504×336, aus `benchmarks/BENCHMARK.md`):

| quant | model MB | load ms | infer ms | peak RAM MB |
|-------|---------:|--------:|---------:|------------:|
| f32   | 393      | 112     | 346.4    | 614         |
| q8_0  | 142      | 40      | 319.4    | 363         |
| q4_k  | 99       | 25      | 395.2    | 320         |

Schlüsselerkenntnis: Bei einem ViT-Forward (compute-bound, große GEMMs) ist q8_0 das einzige Quant-Format,
das *Geschwindigkeit* bringt; q4_k ist auf CPU langsamer als f32 und rein ein *Deployment*-Format
(Größe/RAM). Das prägt den Quant-Scope unten.

---

## 2. Scope v1

**Enthalten:**
- Modelle: DA3-SMALL (ViT-S), DA3-BASE (ViT-B), DA3-LARGE (ViT-L)
- Ausgaben: metrische/relative Depth-Map, per-Pixel-Confidence, Kamera-Pose (Extrinsics 3×4, Intrinsics 3×3)
- Ziel-CPU: x86-64 mit AVX-512 (direkte Vergleichbarkeit mit den veröffentlichten ggml-Zahlen);
  AVX2- und skalarer Fallback vorhanden
- Gewichts-Formate: f32, f16, q8_0
- Auslieferung: Rust-Crate (Workspace) + CLI

**Bewusst NICHT in v1 (Roadmap):**
- q4_k — fest als v2 eingeplant (99-MB-Edge/App-Deployment); k-Quant-Kernel-Port wird budgetiert,
  blockiert v1 aber nicht
- q5_k / q6_k — **ersatzlos gestrichen** (kein eigener Job: kaum kleiner als q8_0, kaum schneller
  als q4_k; q4_k ist near-lossless). Nur wieder aufnehmen, falls q4_k ein Qualitätsproblem zeigt.
- GPU-Backend (CUDA/Metal/Vulkan via CubeCL) — v2; Backend-Trait wird aber ab v1 offengehalten
- GIANT (ViT-g, GS-/Gaussian-Head), NESTED (Zwei-Branch-Alignment), DA2, Mono-/Metric-Sonderköpfe
- Server-/Service-Modus mit Batching-Queue (v2 — hier wird das 2–4×-Durchsatzziel sichtbar)
- C-API-Drop-in, Python-Bindings

Der Loader liest ab v1 die **vollständigen** GGUF-Metadaten und lehnt unbekannte/nicht unterstützte
Modelltypen mit sauberer Fehlermeldung ab, statt zu raten.

---

## 3. Gewählter Ansatz

**Ansatz A: Schlanke Engine mit geliehenem GEMM** (mit Ansatz B — eigener Mikrokernel — als
gezielte Eskalation pro Kernel, datenbasiert entschieden durch Meilenstein 1).

Eigener GGUF-Loader, eigener statischer Graph-Executor, eigene Modell-Implementierung. Selbst gebaut
wird nur, was es fertig nicht performant gibt. f32-GEMM kommt anfangs aus einer bestehenden Crate
(`faer` primär, `tract-linalg` als Vergleichskandidat).

Verworfen:
- **Ansatz B als Startpunkt** (auch GEMM from scratch): GEMM-Mikrokernel auf ~90 % von MKL/faer
  zu bringen ist Monate Arbeit vor dem ersten E2E-Vergleich; maximiert das „Rewrite ist erstmal
  langsamer"-Risiko. Bleibt als Evolution pro Kernel jederzeit möglich.
- **Ansatz C** (candle als Chassis): Genau die Schicht, aus der die Performance-These kommt
  (statischer Graph, Speicherplanung, Fusion), gehörte dann candle. Zudem hat die zentrale Frage
  „candle-CPU vs. ggml" in der Research keinen belastbaren Benchmark überlebt.

---

## 4. Architektur & Crate-Struktur

Cargo-Workspace, fünf Crates, geschnitten entlang „was muss sich unabhängig ändern können":

```
depth-anything-rs/
├── crates/
│   ├── da-gguf       # GGUF-Loader: Datei → Metadaten + Tensor-Views (mmap)
│   ├── da-kernels    # SIMD-Kernel + Dispatch: GEMM, q8_0-VecDot, Attention,
│   │                 # fusionierte Elementwise-Ops. Kennt nur Slices, keine Tensoren.
│   ├── da-graph      # Statischer Graph-Executor: Planung, Buffer-Arena, Threadpool,
│   │                 # Backend-Trait (CPU v1, GPU v2). Dumm und statisch.
│   ├── da-engine     # Das Modell: DINO-Backbone, RoPE2D, DPT-Head, Pose-Head,
│   │                 # Preprocessing, Pos-Embed-Caching.
│   └── da-cli        # CLI: infer + bench
└── da-parity         # Dev-Crate: Parity- & Benchmark-Tests gegen C++/PyTorch (nicht ausgeliefert)
```

Abhängigkeiten strikt gerichtet: `da-cli → da-engine → da-graph → da-kernels`; `da-gguf` wird nur
von `da-engine` konsumiert. `da-parity` hängt von allem ab, gehört aber nicht zur Auslieferung.

### Zwei tragende Schnitte

**`da-kernels` kennt keine Tensoren, nur Slices.** Jeder Kernel ist eine freie Funktion über rohe
`&[f32]` / `&[BlockQ8_0]` mit expliziten Dimensionen — kein Tensor-Typ, kein Graph-Wissen. Macht
Kernel einzeln microbenchmark- und parity-testbar und erlaubt, geliehenes GEMM später gegen eigenes
zu tauschen, ohne dass Code oberhalb es merkt.

**`da-graph` ist dumm und statisch.** Kein dynamischer Graph-Aufbau pro Forward (wie ggml). Die
Engine deklariert den Graphen einmal beim Laden (Architektur steht in den GGUF-Metadaten fest); der
Executor plant einmalig die Buffer-Arena (Lebenszeiten der Zwischentensoren → Buffer-Wiederverwendung,
**null Allokationen im Forward**) und die Thread-Aufteilung. Jeder Forward spult nur den Plan ab.
Der Backend-Trait sitzt hier: v1 = genau eine CPU-Implementierung, aber die Grenze existiert ab Tag 1,
damit CubeCL/CUDA in v2 andockt, ohne `da-engine` anzufassen.

`da-gguf` bleibt eigenes Crate: mmap-basiert (Load-Zeit ist ein Verkaufsargument — 40 ms bei q8_0),
liest die komplette Metadaten-Selbstbeschreibung („nothing is hardcoded"), standalone publizierbar.

---

## 5. Komponenten im Detail

### 5.1 `da-kernels` (nach Anteil an der Forward-Zeit)

1. **f32-GEMM** — geliehen von `faer` (primär; unterstützt Custom-Epiloge nativ) mit `tract-linalg`
   als Vergleichskandidat in Meilenstein 1. Nicht selbst geschrieben.
2. **q8_0-Pfad** — Aktivierungs-Quantisierung nach int8 + Block-Dot-Product. Direkter Port von
   ggmls AVX-512/VNNI-Kernel (`vpdpbusd`). Teuerster Eigenbau-Posten in v1. Kein fertiges Rust-Äquivalent.
3. **Attention** — fusionierte Flash-Attention-artige CPU-Variante (tiled, online-Softmax). RoPE2D
   wird in den Q/K-Ladepfad fusioniert, kein eigener Pass.
4. **Fusionierte Elementwise-Kette** — LayerNorm, GELU, Bias-Add, Residual als GEMM-Epiloge bzw. ein
   fusionierter Pass statt vier memory-bound Einzelpässen. Kern der Fusion-These.
5. **Conv2D** (Patch-Embedding + DPT-Head) — im2col + GEMM in v1. Winograd (wie im Original) nur,
   falls Profiling die Convs als relevant zeigt (YAGNI: bei ViT typischerweise <10 %).
6. **Kleinkram** — Bilinear/Bicubic-Upsampling (DPT), Softmax, Sigmoid/ReLU-Ausgänge.

**Dispatch:** einmal beim Start per `is_x86_feature_detected!` (AVX-512 → AVX2 → skalar). Der skalare
Pfad ist die Referenzimplementierung, gegen die jeder SIMD-Kernel getestet wird.

### 5.2 `da-engine`

Spiegelt die C++-Struktur (dino_backbone, vit_block, rope2d, dpt_head, ray_pose/cam_pose, preprocess),
mit zwei Engine-Level-Optimierungen als Grundprinzip statt Nachrüstung:

- **Geometrie-Caching:** Alles, was nur von der Eingabegeometrie abhängt, wird beim ersten Forward
  berechnet und gecacht — die beiden Positional-Embeddings (die ~95-ms-Lektion aus dem C++-Projekt),
  im2col-Indizes, Attention-Masken. Cache-Key = Auflösung.
- **Preprocessing gehört zur Engine:** Resize/Normalisierung SIMD-parallel, direkt ins Eingabe-Layout
  des ersten Kernels geschrieben (kein Zwischenformat).

### 5.3 `da-cli`

- `infer` — Bild → Depth (PNG/PFM) + Pose (JSON)
- `bench` — repeat=N, Warmup, Median/p95; dasselbe Protokoll wie `benchmarks/BENCHMARK.md`

---

## 6. Dev-Flow: Parity-getrieben, „C++ als Sparringspartner"

Jede Rust-Komponente hat zwei Gegner: den **Korrektheits-Oracle** (Referenz-Dumps) und den
**Geschwindigkeits-Gegner** (die C++-Komponente). Zyklus pro Komponente:
*rot → korrekt → schneller als C++ → (Zwei-Iterationen-Regel) → nächste Komponente.*

### 6.1 Korrektheit — dieselben Dumps, dieselben Toleranzen

Das C++-Repo bringt die Infrastruktur bereits mit: `scripts/dump_*.py` erzeugen Referenz-Tensor-Dumps
pro Komponente; `tests/test_*.cpp` + `tests/parity.hpp` definieren den Vergleich und die Toleranzen.

`da-parity` liest exakt dieselben Dump-Dateien: Eingabe-Dump laden → Rust-Komponente ausführen →
gegen Ausgabe-Dump vergleichen, mit denselben Schwellwerten wie `parity.hpp`. Ist
`cargo test -p da-parity` grün, ist die Rust-Engine per Konstruktion genauso „gegen DA3 verifiziert"
wie die C++-Engine — dieselbe Beweiskette, ein Glied weiter.

**Kein Bit-für-Bit-Vergleich mit C++:** anderes GEMM und andere Summationsreihenfolge ⇒ andere letzte
Bits. Maßstab ist derselbe wie im Original: Korrelation/Toleranz gegen die PyTorch-Referenz.

### 6.2 Geschwindigkeit — Baseline-Datei statt Bauchgefühl

Einmal pro Maschine baut ein Skript die C++-Seite und misst pro Komponente die Zeiten (Test-Executables
existieren; wir ergänzen ein `--bench`-Flag / Mini-Timing-Target). Ergebnis: eingecheckte `baseline.json`
(`{vit_block: 4.1ms, dpt_head: 22ms, …}`). Rust benchmarkt mit `criterion` gegen diese Datei; `cargo bench`
druckt pro Komponente:

```
Komponente     C++ (Baseline)   Rust     Faktor
vit_block         4.10 ms      3.65 ms   1.12x  ✓
dpt_head         22.0  ms     24.8  ms   0.89x  ✗ ← hier arbeiten
```

Kein Big-Bang-Vergleich am Schluss: Jede Komponente wird einzeln geschlagen, Fortschritt Richtung
346 ms ist jederzeit als Summe ablesbar. Meilenstein 1 (ViT-Block-Benchmark aus dem Report: faer vs.
tract-linalg vs. ggml, inkl. Epilogen-Fusions-Test) ist die erste Zeile dieser Tabelle und der
Go/No-Go für den GEMM-Baustein.

### 6.3 Die Zwei-Iterationen-Regel

Sobald eine Komponente die C++-Baseline schlägt, ist sie **nicht** fertig:

1. Grün → committen (Sicherungspunkt).
2. **Iteration +1:** nächste benannte Optimierungshypothese umsetzen (Fusion erweitern, Blocking-Faktor
   tunen, Prefetch, besseres Tiling). Messen.
3. **Iteration +2:** dasselbe mit der nächstbesten Hypothese.
4. Weiter zur nächsten Komponente.

**Ausnahme (die wichtige Hälfte):** Eine Iteration wird nur versucht, wenn es eine *konkrete, benannte
Hypothese* gibt. Ist die Komponente nachweislich am Limit (GEMM >90 % der theoretischen FLOP/s, oder
messbar memory-bandwidth-bound am Roofline-Punkt), wird nicht weiter iteriert, sondern im
Komponenten-Log als „am Limit, Grund: …" festgehalten. Kein Drücken gegen Wände.

Jede Komponente führt ein kurzes **Optimierungs-Log** (versuchte Hypothesen + Ergebnis, auch
verworfene) — zugleich die Roofline-Begründung, warum sie als ausoptimiert gilt.

### 6.4 E2E-Gate

Über allem: Gesamtlatenz via `da-cli bench` gegen die C++-CLI (gleiches Bild, gleiche Auflösung),
plus die vorhandenen `e2e_verify.py`-Skripte gegen die Rust-Ausgaben.

---

## 7. Fehlerbehandlung

- `Result`-basiert an den Rändern: GGUF laden, Bild-I/O, unbekannter Modelltyp → saubere Fehlermeldung
  statt Raten.
- **Kein `Result` im heißen Forward-Pfad:** Der Graph ist beim Laden validiert, der Forward *kann*
  nicht fehlschlagen.
- `unsafe` nur in SIMD-Kerneln, gekapselt hinter safe Funktionen mit Slice-Längen-Debug-Asserts. Jeder
  `unsafe`-Kernel hat seinen skalaren safe-Zwilling als Test-Oracle.

---

## 8. Testing (Gesamtbild)

1. **Korrektheit** — `da-parity` Komponententests gegen die C++/PyTorch-Dumps (Toleranzen aus `parity.hpp`).
2. **Geschwindigkeit** — `criterion` gegen `baseline.json`, mit der Zwei-Iterationen-Regel.
3. **E2E** — `da-cli bench` + `e2e_verify.py` gegen Rust-Ausgaben (Gesamtlatenz + Endausgabe).
4. **SIMD-Kernel** zusätzlich gegen ihren skalaren Zwilling.

---

## 9. Offene Punkte / Risiken

- **Epilogen-Fusion durch die GEMM-API:** faer erlaubt Custom-Epiloge, tract-linalg nur bedingt.
  Meilenstein 1 muss verifizieren, dass die LayerNorm/GELU/Bias-Fusion durch die gewählte GEMM-Crate
  überhaupt möglich ist — sonst kippt die Kernel-Strategie Richtung Ansatz B für die betroffenen Ops.
- **q8_0-VNNI-Port:** teuerster Einzelposten; Aufwand hängt daran, wie direkt sich ggmls Intrinsics
  übersetzen lassen.
- **„Rewrite ist erstmal langsamer":** durch den komponentenweisen Flow entschärft, aber real bis die
  Kernel sitzen. Erwartungsmanagement: v1-Zwischenstände dürfen unter der Baseline liegen.
- **`std::simd` ist nightly-only:** für stable Rust heißt SIMD Intrinsics oder `pulp`/`faer`.
