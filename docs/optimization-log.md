# Optimierungs-Log

Jede Kernel-/Komponenten-Task trägt hier nach der Zwei-Iterationen-Regel (Spec §6.3) ein.

## 2026-08-12 — CPU-F32-Regression, Iteration 1: RefineNet-Gewichtskopien entfernen

- **Unveränderter Messvertrag:** Ryzen 9 9950X, 16 CPU-Threads,
  `depth-anything-base-f32.gguf`, `mountains.jpg` bei 504×336, ein Warm-up
  und zehn gemessene Inferenzdurchläufe je Prozess. Die archivierte Baseline
  bleibt unverändert: C++/ggml F32 `246.683 ms`, Rust F32 `2954.256 ms`
  (Mittel der zehn Trial-Mediane). Ein neuer Vollbenchmark wurde in dieser
  Iteration bewusst noch nicht ausgeführt.
- **Profilbefund / Hypothese:** Der Head benötigte ungefähr `0.78 s` des
  Rust-Laufs. `dpt_head_debug` kopierte für jede Inferenz alle RefineNet-
  Gewichte mit `to_vec()`, obwohl diese unveränderlich in `Weights` liegen.
  Wenn diese Kopien entfernt werden, müssen die F32-Ausgaben bitgleich zum
  bisherigen Rust-Pfad bleiben und der Head darf keinen langsameren
  Smoke-Wert liefern.
- **Änderung:** `crates/da-engine/src/dpt_head.rs` leiht nun die
  `head.scratch.rn*`-Gewichte direkt aus `Weights`; Rechenreihenfolge,
  Shapes, Kernel und Ausgabepuffer bleiben unverändert.
- **Lokale Verifikation:** `cargo test -p da-engine dpt_head` PASS (5
  Unit-Tests und der dump-gesteuerte DPT-Test). Der echte C++-F32-
  Vier-Bilder-Paritätstest ist für diese Iteration noch offen, weil die
  Workhorse-Verbindung nach der Artefakt-Wiederherstellung nicht mehr
  auflösbar war. Es wird kein Paritäts- oder Performance-Sieg behauptet,
  bevor dieser Lauf mit den vier echten Bildern ausgeführt wurde.
- **Infrastrukturstatus:** Auf dem Workhorse wurden der exakte C++-Stand
  `2028b47ac75a8659c6a9aa617baf09be193eb55f` einschließlich ggml-Submodul
  `eced84c86f8b012c752c016f7fe789adea168e1e` sowie der F32-Checkpoint
  wiederhergestellt. Checkpoint-SHA-256:
  `1b13b166e8a8b4f2c862f42d36edb2f9aab995a18cc527a52b9f160b99c6b8da`.
  Die vier Eingabebilder befinden sich im gepinnten C++-Baum. Der nächste
  Schritt nach Wiederherstellung von SSH/Tailscale ist ein 1×10-Smoke mit
  anschließendem Vier-Bilder-Paritätslauf; erst dann folgt bei positiver
  Tendenz der randomisierte 10×-Vollbenchmark.

## 2026-08-12 — CPU-F32-Regression, Iteration 2: Mini-Graphen aus dem ViT-Hotpath entfernen

- **Profilbefund / Hypothese:** Der beobachtete Backbone-Anteil von etwa
  `2.29 s` enthält pro Bild 12 Blöcke mit jeweils LayerNorm, vier linearen
  Projektionen und Attention. Der bisherige Pfad erzeugte dafür ungefähr 84
  `Graph`/`Plan`/`Arena`-Instanzen und kopierte Aktivierungen sowie alle
  beteiligten Gewichte in deren Arena. Wenn LayerNorm, GEMM, Bias, GELU,
  LayerScale, QK-Norm, RoPE und Attention direkt mit denselben bestehenden
  F32-Kernels ausgeführt werden, entfallen Plan-Erzeugung und Gewichtskopien,
  ohne die mathematische Reihenfolge zu verändern.
- **Änderung:** `crates/da-engine/src/vit_block.rs` verwendet für diesen
  Hotpath nun die bestehenden direkten `da_kernels`-Kernels und `FaerGemm`
  mit geliehenen `Weights`. Der generische Graph-Backend bleibt für seine
  anderen Nutzer unverändert.
- **Regressionstest:** Neu:
  `direct_layernorm_and_linear_match_the_retired_graph_path`. Er vergleicht
  die direkte F32-Ausführung gegen den vorherigen Graph-Pfad bitweise für
  LayerNorm sowie GEMM + Bias + GELU + LayerScale. Zusätzlich bleiben die
  Tests für LayerScale-, QK-Norm- und RoPE-Gating aktiv.
- **Lokale Verifikation:** `cargo test -p da-engine vit_block` PASS (5
  Tests). Die Vier-Bilder-Parität und der 1×10-CPU-Smoke sind als nächste
  Pflichtgates offen; bis dahin existiert bewusst kein neuer Millisekunden-
  oder Speedup-Wert.

## Meilenstein 1 — GEMM-Baustein (vit_block-GEMMs, DA3-BASE @256 Tokens)

- faer vs. scalar (criterion, `cargo bench -p da-kernels --bench gemm_bench`, Apple Silicon, `Parallelism::None`, single-threaded, release/LTO-thin, 100 samples je Shape):

  | Shape (m×n×k)      | GEMM-Rolle              | faer (median) | scalar (median) | Speedup |
  |---------------------|--------------------------|---------------:|-----------------:|--------:|
  | 256×2304×768        | QKV-Projektion           | 7.565 ms       | 25.709 ms         | 3.40×   |
  | 256×768×768         | Attn-Output-Projektion   | 3.325 ms       | 16.332 ms         | 4.91×   |
  | 256×3072×768        | MLP-fc1                  | 15.530 ms      | 35.069 ms         | 2.26×   |
  | 256×768×3072        | MLP-fc2                  | 10.223 ms      | 35.154 ms         | 3.44×   |

  faer schlägt den naiven skalaren Dreifach-Loop auf allen vier repräsentativen ViT-Block-Shapes deutlich, mit 2.3×–4.9× Speedup (Median über 100 Samples je Shape) — und das bereits mit `Parallelism::None` (kein Multi-Threading, nur faers interne SIMD-Microkernel-Optimierung).

- faer vs. ggml/tinyBLAS (C++-Baseline, gleiche Shapes): not yet available (see Task 23).

- Epilogen-Fusion durch faer::matmul möglich? **Bedingt.** `faer::linalg::matmul::matmul` nimmt zwar einen `Option<E>`-Parameter (`alpha`) für eine Akkumulations-Option (C = alpha*C + beta*A*B), das ist aber kein generischer in-Kernel-Epilog-Hook für beliebige Elementweise-Operationen wie Bias-Add oder GELU — es gibt keine Callback-/Closure-Schnittstelle, die pro Ausgabekachel vor dem Store aufgerufen würde. `GemmWithEpilogue` in `gemm.rs` implementiert deshalb bewusst *post-hoc*-Fusion (Bias/GELU als separater, aber cache-warmer Pass direkt nach dem GEMM-Aufruf), nicht echte in-Kernel-Fusion. Für v1 ist das ausreichend; ein echter fused Epilog würde einen eigenen Mikrokernel (Ansatz B) oder eine niedrigere Ebene der faer-API (Kachel-Callbacks, falls vorhanden) erfordern.

- **Go/No-Go:** faer erreicht (baseline noch nicht verfügbar, siehe oben) — aber bereits gegen den naiven Skalar-Referenzkernel 2.3×–4.9× schneller, ohne jegliches manuelles Tuning. Das ist ein klares Signal, dass eine ausgereifte SIMD/BLAS-artige Bibliothek hier substanziellen Mehrwert bringt. **Entscheidung: faer behalten.** Eskalation auf tract-linalg oder einen eigenen Mikrokernel (Ansatz B) ist für diesen Baustein aktuell nicht gerechtfertigt; die Entscheidung wird in Task 23 anhand der echten ggml/tinyBLAS-Baseline-Zahlen erneut geprüft (insbesondere ob Parallelism aktiviert werden sollte, um näher an eine Multi-Thread-C++-Baseline heranzukommen).

## Task 7 — tract-linalg als dritter Vergleichskandidat: **verworfen**

- **Grund: API-Integrationsaufwand sprengt den Rahmen eines reinen Vergleichs-Benchmarks.** `tract-linalg = "0.21"` löst als Dev-Dependency von crates.io auf (nach Pin `kstring@2.0.0` wegen einer zu neuen transitiven MSRV-Anforderung von `kstring@2.0.4`, siehe unten) und kompiliert sauber gegen `da-kernels` — die Crate selbst ist also erreichbar und baubar. Die tatsächliche öffentliche API weicht aber von der im Brief angenommenen Form `tract_linalg::ops().mmm(m, n, k)` ab und lässt sich nicht auf die bestehende `Gemm`-Trait-Signatur `(m, n, k, &[f32], &[f32], &mut [f32])` abbilden, ohne einen eigenständigen Adapter zu schreiben:
  - `tract_linalg::ops().mmm(accumulator: DatumType, m: Option<usize>, k: Option<usize>, n: Option<usize>) -> Option<Box<dyn mmm::MatMatMul>>` liefert nur den Kernel-Handle, keine direkte "rechne A×B nach C"-Funktion.
  - Eingaben müssen zuerst in `tract_data::Tensor` verpackt und über `MMMInputFormat::prepare_tensor(&tensor, k_axis, mn_axis) -> Box<dyn MMMInputValue>` gepackt werden (Panel-Packing, kein simpler Slice-Wrapper).
  - Der eigentliche Aufruf läuft über `unsafe fn MatMatMul::run(&self, m, n, non_linear: &[FusedSpec]) -> TractResult<()>`, wobei `FusedSpec::AddMatMul { a: AsInputValue, b: AsInputValue, packing: usize }` und `FusedSpec::Store(OutputStore)` (aus `OutputStoreSpec::c_from_data_and_strides`, ebenfalls `unsafe`) explizit zusammengesetzt werden müssen, inklusive optionalem Scratch-Space-Management (`allocate_scratch_space`/`run_with_scratch_space`).
  - Das ist die vom Brief selbst benannte Ausschlussbedingung: "API ist um Packed-Buffer-Vorbereitung mit einer sehr anderen Aufrufkonvention herum designt, die sich nicht sauber auf die bestehende (m,n,k,a,b,c)-Form von `Gemm` abbilden lässt." Ein korrekter, nicht-fabrizierter dritter Benchmark-Arm hätte einen eigenen Pack-/Unsafe-Adapter-Layer erfordert, der über den Umfang eines reinen Vergleichspunkts (Task 7 laut Interface-Beschreibung: "kein Auslieferungsteil") deutlich hinausgeht.
  - Nebenbefund beim Dependency-Check: `tract-linalg 0.21.10` zieht `kstring 2.0.4` transitiv, was `rustc >= 1.96` verlangt; das lokale Toolchain (`rustc 1.93.0`) kompiliert das nur nach einem `cargo update -p kstring --precise 2.0.0`-Downgrade. Das ist ein weiteres Reibungssignal (Dependency-Baum ist auf neuere Toolchains getrimmt), aber nicht der Hauptgrund für die Verwerfung.
- **Vorgehen:** `tract-linalg` probeweise als Dev-Dependency ergänzt, `cargo check -p da-kernels --benches` erfolgreich gegen die reale Crate laufen lassen, deren tatsächliche API im heruntergeladenen Quellcode (`~/.cargo/registry/src/.../tract-linalg-0.21.10/src/lib.rs`, `frame/mmm/mod.rs`, `frame/mmm/fuse.rs`, `frame/mmm/input_store.rs`) verifiziert (keine Annahmen aus dem Brief übernommen), Aufwand/Nutzen bewertet und die Dependency danach wieder entfernt. Workspace baut wieder sauber ohne `tract-linalg`.
- **Entscheidung bleibt: faer als GEMM-Default.** Kein dritter Benchmark-Arm wurde hinzugefügt; die Task-6-Entscheidung (faer, siehe oben) ist final für diesen Meilenstein. Eine erneute Prüfung von tract-linalg lohnt sich nur, falls ein zukünftiger Bedarf (z.B. quantisiertes i32-GEMM oder explizite Epilogen-Fusion über `FusedSpec`) den Adapter-Aufwand rechtfertigt — das ist aktuell nicht der Fall.

## Task 9 — q8_0-Vektor-Dot-Product-GEMM (VNNI-Port)

- **Implementiert:** `quantize_row_q8_0` (symmetrische Pro-Block-Quantisierung, `d = amax/127`, ggml-kompatibel) in `crates/da-kernels/src/q8_0_dot.rs`; `scalar::gemm_q8_0` als Oracle (i32-Akkumulation pro 32er-Block, Skalierung mit `d_a*d_b`, danach f32-Summe) in `crates/da-kernels/src/scalar.rs`; `Kernels::gemm_q8_0` Dispatch-Methode in `dispatch.rs`; AVX-512-VNNI-Pfad `simd_avx512::gemm_q8_0_avx512` mit `_mm512_dpbusd_epi32`.
- **`Isa::Avx512`-Gate erweitert:** `Kernels::detect()` verlangt jetzt AVX-512F **und** AVX-512BW **und** AVX-512VNNI zusammen für die `Avx512`-Stufe (vorher nur F für Tasks 8's gelu/add). Bewusst *ein* vereinheitlichter Tier statt getrennter Feature-Flags pro Kernel — entspricht der im Plan genannten "AVX-512F/BW/VNNI primär"-Rahmung als einen Block, und hält `Kernels`-Dispatch pro Methode auf einem einfachen Zwei-Wege-Match. gelu/add (Task 8) brauchen nur F, laufen aber auf jeder realen AVX-512F-CPU auch mit BW (BW ist seit Skylake-X immer mit F mitgeliefert) — keine erwartete Regression.
- **Zwei-Iterationen-Regel — Hypothesen (Brief Step 5):**
  1. *2 Blöcke/Iteration entrollen* (64 int8-Lanes via `_mm512_dpbusd_epi32` statt 32): **direkt implementiert**, nicht als späteres Stretch-Goal. `gemm_q8_0_avx512` packt zwei 32-Byte-`qs`-Blöcke (die wegen `BlockQ8_0`s `repr(C)`-Layout `{d: f16, qs: [i8;32]}` = 34 Byte/Block *nicht* zusammenhängend im Speicher liegen — dazwischen sitzt jeweils `d`) in einen lokalen 64-Byte-Puffer, lädt ihn als ein `__m512i` und nutzt aus, dass `dpbusd_epi32`s 16 i32-Ausgabe-Lanes pro 4-Byte-Gruppe unabhängig sind (Lanes 0-7 = Block 0, Lanes 8-15 = Block 1) — ein Aufruf liefert also zwei unabhängige Block-Summen, die anschließend mit ihren jeweils eigenen `d_a*d_b`-Skalen gewichtet werden (Blöcke dürfen *nicht* vor der Skalierung gemischt werden, da jeder Block seinen eigenen Skalenfaktor `d` trägt). Ein ungerader Rest-Block fällt auf den skalaren Einzelblock-Pfad zurück.
  2. *A-Quantisierung cachen statt pro GEMM-Aufruf neu* (Aktivierungen ändern sich pro Vorwärtsdurchlauf einmal, nicht pro GEMM): **nicht implementiert** — außerhalb des Scopes von Task 9, die nur `gemm_q8_0(a_q: &[BlockQ8_0], ...)` als bereits-quantisierte Eingabe spezifiziert (Quantisierung liegt beim Aufrufer). Als Optimierung für später vorgemerkt, sobald die Modell-Vorwärtsdurchlauf-Schicht existiert, die q8_0-GEMM tatsächlich mehrfach pro Aktivierung aufruft (z. B. QKV + Attn-Output + MLP im selben Block).
- **Speed-Vergleich (scalar vs. VNNI): DEFERRED auf x86-64-Hardware**, aus demselben Grund wie in Task 8 dokumentiert (siehe Hinweis unten) — AVX-512/VNNI kann auf dieser aarch64-Entwicklungsmaschine nicht ausgeführt oder gebencht werden. Kein "am Limit"-Urteil möglich, solange keine echte Ausführung/Messung vorliegt; folgt in x86-64-CI zusammen mit den restlichen AVX-512-Kerneln.
- **Verifikation:** Gate A (`cargo test -p da-kernels --test q8_0_dot_matches_f32`, scalar-Oracle-Pfad auf diesem Host) PASS; Gate B (`cargo check -p da-kernels --target x86_64-unknown-linux-gnu`, VNNI-Intrinsic-Typkorrektheit) PASS; Gate C (`cargo check -p da-kernels`, normaler Host-Build) PASS.

## Hinweis: AVX-512-Verifikation auf arm64-Entwicklungsmaschine

Die AVX-512F/BW/VNNI-Kernelpfade (Tasks 8–10) werden auf einer aarch64-Maschine (Apple Silicon) entwickelt und sind daher nur verifiziert durch (a) `cargo check --target x86_64-unknown-linux-gnu` für Intrinsic-Typ-/Syntaxkorrektheit und (b) den skalaren Oracle-Pfad, der auf dieser Maschine tatsächlich läuft. Ihre numerische Korrektheit und alle Roofline-/Latenzmessungen sind auf echter x86-64-Hardware **unverifiziert** und müssen vor jeder Performance-Aussage oder einem Release in x86-64-CI validiert werden.

## Task 17 — ViT-Block + Backbone-Forward: Speed-Benchmarking DEFERRED

- **Implementiert:** `da_kernels::scalar::layerscale` (In-Place-Pro-Spalten-Skalierung, spiegelt `add_bias_rows`); `ModelConfig.ffn_type` (GGUF-Key `depthanything3.vit.ffn_type`, Default `"mlp"`); `Op::LayerScale` und ein revidierter `Op::Attention` (optionale QK-LayerNorm mit eigenem `qk_norm_eps`, optionales RoPE) in `da-graph`; `da_engine::vit_block` (LN1 → Attention[+QK-Norm][+RoPE] → [ls1] → Residual → LN2 → MLP(GELU) → [ls2] → Residual) und `da_engine::Backbone::forward` (12-Layer-Stack, Capture an `out_layers`).
- **Speed-Benchmarking gegen `baseline.json` (Brief Step 5, "Zwei-Iterationen-Regel"): bewusst DEFERRED**, aus zwei zusammenhängenden Gründen, nicht nur einem:
  1. `baseline.json` (die reale C++/ggml-Referenzmessung) existiert in diesem Environment noch nicht — sie ist laut Plan erst Task 23's Auslieferungsteil. Ohne sie gibt es keine Zielgröße, gegen die ein "Go/No-Go" (wie bei Task 6/9 oben) sinnvoll bewertet werden könnte.
  2. Es gibt kein reales Modell und keine Dumps in diesem Environment (`../models/da3-base-f16.gguf`, `../dumps/reference.gguf` fehlen beide — siehe `tests/backbone_parity.rs`, das deshalb sauber überspringt). Ein Benchmark von `vit_block`/`Backbone::forward` gegen synthetische Zufallsgewichte (wie in den Unit-Tests dieser Task verwendet) hätte zwar eine Zahl geliefert, aber keine, die irgendetwas über reale Performance-Constraints (echte Tensor-Shapes stimmen zwar überein, aber Cache-Verhalten/Allokationsmuster bei wiederholten Mini-`Plan`-Kompilierungen pro Block-Aufruf — siehe `vit_block.rs`'s `run_layernorm`/`run_linear`/`run_attention`-Helfer — wären nicht repräsentativ für eine spätere, tatsächlich für Wiederverwendung optimierte Fassung) aussagt. Eine fabrizierte Zahl ist schlechter als keine Zahl.
- **Bekannter, dokumentierter Performance-Fluchtpunkt für später:** Die aktuelle `vit_block`-Implementierung kompiliert pro Aufruf mehrere kleine `da_graph::Graph`/`Plan`/`Arena`-Instanzen neu (eine je Rechenschritt: LN1, QKV-Gemm, Attention, Proj+LS1, LN2, FC1+GELU, FC2+LS2) statt eine einzige, über alle 12 Layer wiederverwendete `Arena` zu pflegen. Das ist für Task 17 (Korrektheit der drei Fallen: QK-Norm-Eps, RoPE-Gating, LayerScale-Presence-Gating) bewusst in Kauf genommen — Wiederverwendung/Fusion einer einzigen Arena über den gesamten Backbone-Forward ist ein offensichtlicher erster Optimierungskandidat, sobald eine reale Baseline zum Vergleich existiert.
- **Verifikation ohne Baseline/Dumps:** alle mechanisch prüfbaren Teile (siehe Task-17-Report) sind grün; End-to-End-Zahlen-Parität gegen `feat_{5,7,9,11}` ist UNVERIFIZIERT.

## Task 22 — `da bench` + `compare_e2e.sh`: E2E-Latenzmessung ist Infrastruktur, keine Messung

- **Implementiert:** `da bench --model <gguf> --image <png> --repeat N --warmup W`
  (`crates/da-cli/src/bench.rs`) — lädt das Modell einmal (`Engine::load`), führt
  `warmup` ungemessene `Engine::infer`-Aufrufe aus, misst `repeat` weitere Aufrufe
  mit `std::time::Instant` und druckt Median/p95 (lineare Interpolations-Perzentile,
  dieselbe Konvention wie `numpy.percentile`'s `linear`-Methode) als
  `median_ms=...`/`p95_ms=...`-Zeilen sowie eine `iter[i]_ms=...`-Zeile pro Sample.
  Terminologie und Format ("N warmup + median over N timed iterations") sind bewusst
  an dieses Dokument angelehnt, damit `da bench`-Zahlen direkt mit den oben
  dokumentierten C++/PyTorch-Zahlen vergleichbar sind. `scripts/compare_e2e.sh`
  läuft die reale C++-CLI (`da3-cli depth --model M --input I --repeat N`, deren
  eingebauter Bench-Hook `examples/cli/main.cpp::cmd_depth_bench`) und `da bench` auf
  demselben Bild/Modell mit demselben `--repeat`/`--warmup`-Protokoll und druckt beide
  Mediane nebeneinander plus den Faktor `rust_median_ms / cpp_median_ms`.

- **Schritt 6 (E2E-Latenz messen + Gesamtfaktor gegen die 346-ms-Baseline eintragen):
  NICHT durchführbar in diesem Environment — ehrlich offen gelassen, keine Zahl
  fabriziert.** Zwei harte Voraussetzungen fehlen beide:
  1. Ein echtes DA3-BASE-GGUF-Modell (`../models/*.gguf`) — existiert in diesem
     Environment nicht (bestätigt: `models/` enthält nur `MODEL_CARD.md`/`SHA256SUMS`,
     keine `.gguf`-Datei), genau wie bei jedem anderen modell-gated Test in diesem
     Workspace.
  2. Eine gebaute C++-Referenzbinary — `../build/` existiert in diesem Environment
     nicht; das C++-Projekt wurde hier nie mit `cmake --build ../build` kompiliert.
     `scripts/compare_e2e.sh` wurde tatsächlich ausgeführt (`--model /tmp/nope.gguf
     --image /tmp/nope.png`) und bestätigt den erwarteten sauberen Skip-Pfad: druckt
     "C++ CLI binary not found ... run 'cmake --build ../build' first" und beendet
     sich mit Exit-Code 0 (kein Crash, keine irreführende Fehlermeldung).
  Die `da bench`/`compare_e2e.sh`-Tooling ist fertig gebaut und einsatzbereit, sobald
  beide Voraussetzungen erfüllt sind — das ist **Infrastruktur, keine abgeschlossene
  Messung**. Die Timing-/Perzentil-/Ausgabe-Logik selbst ist real getestet (siehe
  Verifikation unten), nur die *Zahl* gegen die 346-ms-Baseline fehlt.

- **Offene Komponenten-Hebel, die für eine spätere echte E2E-Messung relevant sein
  dürften** (evidenzbasiert aus den bisherigen Tasks, nicht spekulativ neu erfunden):
  1. **faer als GEMM-Backend** (Task 6-Entscheidung) — noch nie gegen die reale
     ggml/tinyBLAS-C++-Baseline verglichen (das o.g. "faer vs. ggml/tinyBLAS:
     not yet available"-Item ist selbst noch offen); erster Kandidat für einen
     Go/No-Go-Check, sobald `da bench` gegen ein echtes Modell läuft.
  2. **q8_0/VNNI-Kernel** (Tasks 8-9) — auf dieser aarch64-Entwicklungsmaschine
     nur über den skalaren Oracle-Pfad verifiziert; numerische Korrektheit und
     jede Performance-Aussage zu AVX-512/VNNI sind laut dem Hinweis oben in
     diesem Dokument auf echter x86-64-Hardware weiterhin unverifiziert.
  3. **`vit_block`'s Mini-Graph/Plan-pro-Op-pro-Layer-Muster** (Task 17s
     dokumentierter "Performance-Fluchtpunkt") — kompiliert aktuell mehrere
     kleine `da_graph::Graph`/`Plan`/`Arena`-Instanzen pro Block-Aufruf statt
     eine einzige über den gesamten Backbone-Forward wiederverwendete Arena;
     laut Task-17-Log der naheliegendste erste Optimierungskandidat, sobald eine
     reale Baseline zum Vergleich existiert.
  4. Die C++-Seite selbst hat bereits Winograd-Conv (#4/#5) und fused
     Flash-Attention (#6) für den Backbone/Head-Pfad; die Rust-Seite (Tasks
     14-21) hat noch keine dieser Optimierungen — ein direkter Vergleich wird
     also nicht nur GEMM-Backend-Unterschiede zeigen, sondern auch, welche der
     oben dokumentierten C++-Optimierungen (Winograd 3x3, ggml_flash_attn_ext,
     pos-embed-Caching) auf der Rust-Seite noch fehlen.

- **Verifikation (ohne echtes Modell/C++-Binary):**
  - `cargo test -p da-cli` — Unit-Tests für `compute_stats`/`percentile` (feste
    Sample-Listen, u.a. das im Brief genannte `[10.0, 20.0, 15.0, ...]`-Muster),
    `clap`-Parsing-Tests für `Command::Bench`, plus ein neuer Integrationstest
    (`tests/bench_native.rs`) der `da bench --repeat 2 --warmup 1` gegen ein
    selbstgebautes synthetisches GGUF (echte Binär-GGUF-Bytes, dasselbe Muster wie
    `da-engine/tests/e2e_native.rs`) laufen lässt und eine parsebare, endliche,
    nicht-negative `median_ms=`/`p95_ms=`-Zeile verifiziert — reale,
    modell-unabhängige Abdeckung der Timing-/Parsing-Logik selbst.
  - `scripts/compare_e2e.sh` mit einem nicht-existenten Modell/Bild ausgeführt:
    bestätigt den Graceful-Skip-Pfad (Exit 0, klare Meldung) in genau der Umgebung,
    in der das C++-Binary tatsächlich fehlt.
  - `cargo test --workspace`: alle Tests grün, keine Regression.

## CPU-F32-Optimierungslauf — Workhorse, 2026-08-12

Messvertrag unverändert: Ryzen 9 9950X, 16 Threads, identisches
`depth-anything-base-f32.gguf`, `mountains.jpg`, 504×336, Modell-Laden und
Bild-Dekodierung ausgeschlossen. Referenz-Baseline: C++/ggml 246,683 ms,
Rust 2.954,256 ms (je Mittelwert von zehn Trial-Medianen).

| Iteration | Hypothese und isolierte Änderung | Smoke (1 Warm-up + 1 Messung) | Parität | Entscheidung |
|---|---|---:|---|---|
| 1 | 1×1-Convs kopieren NCHW unnötig durch im2col. Direkte GEMM-Ansicht statt Kopie. Commit `c3f4d89`. | noch nicht separat gemessen | Kernel-Test bitgleich | behalten |
| 2 | Unabhängige Attention-Zeilen und nicht überlappende Transpose-Convs können kanal-/zeilenparallel laufen. Commits `8eaff27`, `07efefd`. | 878,934 ms | Kernel-Tests bitgleich | behalten |
| 3 | Dichte Faer-GEMMs könnten die Online-Attention schlagen. Commit `3f4b88a`. | **1.029,594 ms** | nur Toleranztest | verworfen; Rücknahme in `d364bbf` |
| 4 | Im2col-Zeilen sind unabhängig; die bei 3×3-Hochauflösung entstehenden hunderte MiB können parallel materialisiert werden. Commit `d364bbf`. | **686,772 ms** | siehe unten | behalten |
| 5 | Streaming-Attention ruft `expf` millionenfach skalar auf. AVX-512 verarbeitet die unabhängigen Softmax-Werte je 64er-Tile vektorisiert; Reduktionsreihenfolge bleibt erhalten. Commit `d4398e7`. | **635,788 ms** (Median, 3 Messungen) | siehe unten | behalten |
| 6 | Q·K-Dotproducts mit 64 Dimensionen werden noch skalar reduziert. AVX-512 multipliziert 16 Lanes gleichzeitig, finale F32-Reduktion bleibt explizit. Commit `cff287b`. | **573,557 ms** (Median, 3 Messungen) | siehe unten | behalten |
| 7 | Die DPT-3×3-Convs materialisieren im2col und führen deutlich mehr Multiplikationen als nötig aus. Winograd F(2×2,3×3) mit additions-/halbierungsbasierten Transformen. Commit `2996447`. | **438,758 ms** (Median, 3 Messungen) | siehe unten | behalten |

Iteration 4 hat im Profil den DPT-Head von etwa 400 ms auf 192 ms reduziert;
die schnellere Iteration ist kein Wechsel der Modell-, Auflösungs- oder
Thread-Parameter. Der Kernel-Test
`parallel_im2col_is_bitwise_serial` vergleicht die parallele gegen die frühere
serielle Materialisierung bitweise. Zusätzlich bestanden Attention- und
Conv-Orakeltests.

Frisch gegen die auf dem Workhorse gebaute C++/ggml-Referenz gemessene
Vier-Bilder-F32-Parität:

| Bild | Pearson r | MAE | Ergebnis |
|---|---:|---:|---|
| canyon | 0,9999936 | 0,001813 | PASS |
| desk | 0,9999783 | 0,001773 | PASS |
| mountains | 0,9999856 | 0,003675 | PASS |
| street | 0,9999721 | 0,000821 | PASS |

Alle Bilder erfüllen r ≥ 0,9999 und MAE ≤ 0,005. Das Ziel ist damit noch
**nicht** erreicht: Ein voller randomisierter 10×-Vergleich ist erst für
einen Kandidaten nahe bzw. unter 222,015 ms gerechtfertigt. Das aktuelle
Operator-Profil zeigt als nächsten dominanten Schritt Attention (ca. 17–21 ms
pro ViT-Block) und anschließend die beiden MLP-GEMMs (ca. 8 bzw. 3 ms pro
Block); ein fehlendes `perf` auf dem Workhorse wurde dokumentiert und durch
Zeitmessung pro Operator ersetzt.

Nach Iteration 5 wurde die Vier-Bilder-Prüfung erneut ausgeführt. Die Werte
blieben innerhalb der oben angegebenen Schwellen (schlechtester Wert:
street r=0,9999721; mountains MAE=0,0036752). Die leicht abweichenden
Rohwerte gegenüber der skalaren Exponentialfunktion sind erwartbar und liegen
deutlich innerhalb des verbindlichen F32-Paritätsvertrags.

Nach Iteration 6 wurde sie noch einmal wiederholt. Schlechteste gemessene
Werte: street r=0,9999721 und mountains MAE=0,0036751; damit weiter PASS.

Iteration 7 wurde ebenfalls gegen den Vier-Bilder-Korpus geprüft. Schlechteste
Werte: street r=0,9999721 und mountains MAE=0,0036753; damit weiter PASS.
Der neue Kernel hat zusätzlich den isolierten Oracle-Test
`winograd_f2_matches_direct_3x3_oracle` (maximal 2e-5 auf zufälligen 3×3-Fällen).

**Verworfene Hypothesen:** Dichte Faer-Attention-GEMMs (1.029,594 ms),
OpenBLAS für den gesamten GEMM-Pfad (1.308,291 ms), parallele LayerNorm
(476,383 ms), ein dauerhaft gehaltener im2col-Workspace (528,263 ms) und
eine Register-only-Dotreduktion (500,831 ms) verschlechterten den Smoke-Test
oder lieferten keinen belastbaren Gewinn. Sie wurden nicht übernommen.

| 8 | Die unabhängigen Kanäle der bilinearen Resize-Operatoren können ohne Änderung der Pixelarithmetik parallel verarbeitet werden. Commit `e3110cf`. | **441,690 ms** (Median, 7 Messungen) | Vier Bilder PASS (identische Werte zur Tabelle oben) | behalten |
| 9 | Winograd F(2×2,3×3) allokiert einen temporären Input-Tile-Vektor pro Tile. Worker-lokaler Scratch eliminiert diese Allokationen. Commit `5ab6adb`. | **412,742 ms** (Median, 5 Messungen) | Vier Bilder PASS (identische Werte zur Tabelle oben) | behalten |

Nach Iteration 9 blieben alle vier End-to-End-Ergebnisse gegenüber C++ F32
innerhalb des Vertrags: schlechtestes `r=0,9999721` (street), höchste
`MAE=0,0036753` (mountains). Die Veränderung betrifft ausschließlich die
Lebensdauer temporärer Speicher und ist zusätzlich durch den bestehenden
Winograd-gegen-direkte-Conv-Oracle-Test abgedeckt.

**Weitere verworfene Hypothesen (nicht übernommen):** Q-blockierte Online-
Attention (464 ms), AVX-512-Winograd mit pro Inferenz umgeordneten Gewichten
(497 ms), direkte parallele Tile-Ausgabe (531 ms), parallele QKV-Transposes
(426 ms) und AVX-512-Residual-Adds (428 ms). Die verworfenen Implementierungen
wurden jeweils vollständig entfernt; weder Messvertrag noch Modell-, Bild-,
Auflösungs- oder Threadparameter wurden geändert.

| 10 | Ein AVX-512-F32-Mikrokernel (4 Tokenzeilen × 96 Ausgabekanäle) könnte Faer für die vier DA3-BASE-Projektionen ersetzen. Die A/B-Messung erfolgt im selben Binary; nur der GEMM-Pfad wird umgeschaltet. | Spezialkernel 337,251 ms, Faer 326,949 ms (je 1 Warm-up + Median aus 5 Messungen) | nicht nötig; Kandidat verworfen, kein Produktionspfad geändert | verworfen |

Der Kandidat las die zeilenorientierten Gewichtstiles für jede Vierergruppe
von Token erneut und war damit trotz AVX-512 schlechter als der bestehende
gepackte Faer-Pfad. Er wurde vor einer Paritätsprüfung vollständig entfernt.
Weitere Kernel-Experimente werden nicht im Hauptrepository vermischt: Das
isolierte Repository `../da3-kernels` (Start-Commit `8b2f56f`) besitzt die
spätere importierbare Schnittstelle, Shape-Vertrag und eigene Benchmarks.
Die nächste Hypothese muss daher die Gewichtslayout-/Prepacking-Frage lösen,
statt lediglich die innere Schleife zu vektorisieren.

| 11 | C++/ggml packt Winograd-Filter als `Position × Eingang × Ausgang` und verarbeitet acht Tiles je Microkernel. Rust rechnet bisher Tile für Tile und lädt die Filter wiederholt. | **239,571 ms** (Smoke: 1 Warm-up + Median aus 10 Messungen, Workhorse) | Vier Bilder PASS; r ≥ 0,9999721, MAE ≤ 0,0036751 | behalten |

Iteration 11 portiert ausschließlich diese Datenführung in das getrennte
Repository `../da3-kernels` (Commit `9b1aa87`) und bindet sie im Runtime-Repo
über Commit `2365c42` ein. Der externe Kernel akzeptiert nur die explizite
F(2×2)-Winograd-Blockform (maximal acht Tiles, Ausgangskanäle in 16er-Gruppen)
und fällt außerhalb davon auf den bestehenden Pfad zurück. Sein Unit-Test
vergleicht die FMA-Akkumulation bitgenau mit derselben skalaren Reihenfolge.
Der Messvertrag blieb unverändert. Das Ergebnis ist ein belastbarer
Smoke-Kandidat, aber noch kein vollständiger 10×-Sieg und noch **17,556 ms**
über dem harten Ziel von 222,015 ms.

**Verworfen (nach Iteration 11):** Ein externer AVX-512-Kernel für die
64-wertige Q/K-LayerNorm. A/B im selben Release-Binary ergab 238,087 ms mit
dem Kandidaten gegenüber 237,771 ms ohne ihn (je 1 Warm-up, Median aus 10).
Der Kandidat wurde vollständig entfernt und nicht committet.

| 12 | FC1 schreibt das 865×3072-Ergebnis zunächst aus, bevor Bias und GELU in zwei separaten Speicherdurchläufen folgen. Ein spezialisierter Kernel kann exakt dieselbe K-major-FMA-Akkumulation beibehalten und Bias plus GELU beim finalen Store anwenden. | **220,609 ms** gegen **224,149 ms** mit `DA3_KERNELS_DISABLE_FC1_EPILOGUE=1` (selbes native Release-Binary, 1 Warm-up + Median aus 5 Messungen) | Vier Bilder PASS: Canyon r=0,9999936 / MAE=0,0018125; Desk r=0,9999783 / 0,0017725; Mountains r=0,9999856 / 0,0036750; Street r=0,9999721 / 0,0008209 | behalten |

Iteration 12 liegt im eigenständigen Repository `../da3-kernels`, Commit
`5b9ff33`; die Runtime-Anbindung ist in `17bb9ed` und `1e8adc4` festgehalten.
Der Kandidat ist ein Smoke-Ergebnis, kein finaler 10×-Vergleich. Er verwendet
weiterhin das identische F32-Modell, Bild, 504×336-Auflösung und 16 Threads;
die volle Studienserie bleibt dem Zielkandidaten vorbehalten.

| 13 | Die großen Q/K/V- und Attention-Aktivierungen werden zwischen den zwölf Transformer-Blöcken neu alloziert. Ein gemeinsamer Workspace könnte ausschließlich Allocator- und Initialisierungskosten entfernen. | Wiederverwendung p95 241,201 / 236,624 ms, bisheriger Pfad p95 238,951 / 235,435 ms (alternierend im selben Release-Binary, 1 Warm-up + 10 Messungen) | Lokale ViT- und Backbone-Tests PASS; keine End-to-End-Parität nötig, weil verworfen | verworfen |

Iteration 13 wurde vollständig zurückgenommen. Die Messmaschine war während
dieses A/B zudem deutlich langsamer als der vorherige 218,6-ms-Smoke, deshalb
wird daraus weder ein Fortschritt noch ein Rückschritt für die Rangliste
abgeleitet. Die Arbeitsbereichs-Variante ist aber in beiden alternierenden
Paaren schlechter als der alte Pfad und wird nicht weiter verfolgt. Nächster
Schritt bleibt ein Operator-für-Operator-Vergleich gegen C++ mit stabiler
Maschinenauslastung, bevor ein weiterer Kernel gebaut wird.

| 14 | Der FC1-Bias/GELU-Fusionskernel sollte zwei Speicherpasses sparen. Das alterierende A/B muss bestätigen, dass sein einfacher AVX-512-GEMM nicht Faers Packing- und Threadingvorteil verliert. | Fusion 232,616 / 227,165 / 224,107 ms; bestehender Faer- + Epilogpfad 208,743 / 210,029 / 209,761 ms (jeweils 1 Warm-up + Median aus 10) | Vier Bilder PASS: canyon r=0,9999936 / MAE=0,0018125; desk r=0,9999783 / 0,0017725; mountains r=0,9999856 / 0,0036750; street r=0,9999721 / 0,0008209 | Fusion entfernt |

Iteration 14 nimmt ausschließlich die Runtime-Anbindung des FC1-Kernels
zurück. Der Kernel bleibt im getrennten Repository für spätere, anders
gepackte Varianten erhalten, wird aber nicht mehr ausgeführt. Der schnellere
Fallback hat im direkten Workhorse-Smoke 212,836 ms erreicht. Die Ausgabe ist
gegenüber der C++-F32-Referenz innerhalb des unveränderten Vertrags auf allen
vier Bildern verifiziert.

| 15 | Für die 865×768/2304/3072-Transformer-Projektionen wird derselbe Gewichtspanel bislang pro 6-Token-Zeilengruppe gescannt. Spaltenpanels erlauben einen K-major-FMA-identischen Pfad mit deutlich weniger geteiltem L3-Gewichtsverkehr. | Spaltenpfad 197,801 / 195,855 / 201,115 ms; Zeilenpfad 204,507 / 205,495 / 202,705 ms (alternierend, 1 Warm-up + Median aus 10) | lokaler und Workhorse-Bitvergleich der beiden Kernelformen PASS; Vier Bilder gegen C++ PASS | behalten |

Die Kerneländerung liegt im separaten Repository `../da3-kernels`, Commit
`f81732e`. Sie betrifft nur exakt definierte DA3-BASE-F32-Projektionsshapes;
alle übrigen GEMMs bleiben auf Faer. Die vier F32-Metriken blieben exakt auf
dem bisherigen Stand: niedrigstes r=0,9999721, höchste MAE=0,0036751.

| 16 | Der globale Winograd-Cache prüfte bei jedem warmen Bild erneut den vollständigen Filterinhalt per FNV-Hash. Ein modellgebundener DPT-Cache kann die vorbereiteten Filter sicher über die Lebensdauer desselben geladenen Modells halten. | **194,302 ms** (1 Warm-up + Median aus 10) | Vier Bilder PASS: canyon r=0,9999936 / MAE=0,0018125; desk r=0,9999783 / 0,0017725; mountains r=0,9999856 / 0,0036750; street r=0,9999721 / 0,0008209 | behalten |

Die Instrumentierung zeigte vor der Änderung etwa 3,3 ms kumulierte Hashzeit
pro warmem Bild (darunter 0,77 ms allein für den 768×128-Filter und rund
1,8 ms für die 128×128-Filter). Die neue Cache-Instanz gehört zu `Engine`;
sie verwendet keine globale Pointer-Identität und kann deshalb keine Filter
zwischen separat geladenen Modellen verwechseln. Die erste ungemessene
Warm-up-Inferenz bereitet die Filter vor, die gemessenen Durchläufe verwenden
sie direkt.

**Verworfen nach Iteration 16:** Spaltenaufteilung des FC2-/Projektions-
Epilogs (209–220 ms gegenüber 194–206 ms für den bestehenden Zeilenpfad),
Spaltenaufteilung von QKV (kein stabiler Vorteil) sowie ein AVX-512-
Spezialkernel für die 128→128-1×1-Fusion (nicht stabil besser). Sie wurden
nicht in den Produktionspfad übernommen. Ein `target-cpu=native`-Build wurde
wegen stark schwankender, nicht aussagekräftiger Einzelmessungen ebenfalls
nicht als Benchmark-Konfiguration übernommen.

| 17 | Die beiden nichtüberlappenden Head-Upsampling-Convolutions packen ihre unveränderlichen IOHW-Gewichte pro Bild um. Ein modellgebundener Cache könnte diese Vorbereitung in die ungemessene Warm-up-Phase verschieben. | Cache 201,023 / 196,730 ms; erneutes Packen 198,730 / 190,313 ms (alternierend, je 1 Warm-up + Median aus 10) | nicht nötig; Kandidat verworfen | verworfen |

Iteration 17 erweitert den separaten Kernelvertrag um eine vorbereitete
Transpose-Filterform, aber der modellgebundene Cache selbst war in beiden
direkten Paaren langsamer. Die hypothetische Einsparung beim Umpacken ist
kleiner als die zusätzliche Cache-Verwaltung; der Produktionspfad verwendet
daher weiterhin das bewährte, pro Aufruf vorbereitete Format. Die aktuelle
Vier-Bilder-F32-Parität wurde danach erneut bestätigt: niedrigstes
`r=0,9999721`, höchste `MAE=0,0036751`.

**Verworfen nach Iteration 17:** Ein Winograd-Block aus 16 statt acht Tiles.
Der isolierte Smoke lag bei 208,488 ms und damit klar außerhalb der bisherigen
Varianz des acht-Tile-Pfads. Die Änderung wurde unmittelbar zurückgenommen;
größere Arbeitsblöcke erhöhen hier Register- und Cache-Druck stärker als sie
Scheduler-Overhead sparen.

| 18 | Das feste DA3-BASE-Attention-Layout enthält 865 Schlüssel; der letzte 64er SIMD-Block hatte bislang 31 physische Padding-Spalten je Dimension. Ein dichter K-Pack mit maskierten Tail-Ladevorgängen kann die ungenutzten Daten entfernen, ohne gültige Score-Lanes zu ändern. | Dicht/maskiert 213,071 / 215,223 ms; gepaddete Kontrolle 215,792 / 215,451 ms (alternierend, 1 Warm-up + Median aus 10) | Vier Bilder PASS: niedrigstes `r=0,9999721`, höchste `MAE=0,0036751` | behalten |

Iteration 18 ersetzt ausschließlich die physische Tail-Behandlung des
persistenten Flash-Attention-K-Packs. Ein erster dichter Kandidat wurde vor
der Annahme verworfen, weil dessen letzter 33-Key-Block out-of-bounds las und
bei der Paritätsausführung segmentierte. Der endgültige Kandidat maskiert die
fehlenden AVX-512-Lanes ausdrücklich auf Null und wurde anschließend auf dem
gesamten Vier-Bilder-Korpus verifiziert. Damit bleibt die FMA- und
Online-Softmax-Reihenfolge jeder gültigen Score-Lane unverändert.

| 19 | Vollständiger, randomisierter CPU-F32-Vergleich des bereinigten Kandidaten gegen die auf demselben Workhorse frisch gebaute C++/ggml-Referenz. | Rust **208,889 ms** [206,109; 211,668], C++ **241,203 ms** [240,300; 242,107] (je 10 unabhängige Trial-Mediane, 1 Warm-up + 10 Messungen) | Vier Bilder erneut PASS, niedrigstes `r=0,9999721`, höchste MAE `0,0036751` | 10%-Ziel bestanden: Rust ist **13,40 %** schneller |

Die Rohdaten dieses Durchlaufs liegen auf dem Workhorse unter
`/tmp/da3-cpu-f32-final-candidate/raw-results.json`; `RESULTS.md` enthält
die Statistik, den festen Seed `20260812`, Befehle, Modell- und Bild-Hashes.
Der Build war auf Ryzen 9 9950X fest auf `target-cpu=znver5` spezialisiert;
beide Arme verwendeten 16 Benchmark-Threads und dieselbe DA3-BASE-F32-GGUF
bei 504×336. Der Abstand der 95%-Intervalle ist klar getrennt. Das strengere
25%-Ziel bleibt offen.

| 20 | Ein flacher Attention-Scheduler sollte verschachtelte Rayon-Aufgaben vermeiden: zunächst alle K-Packs, danach eine gemeinsame Query-Tile-Menge über alle Heads. | Flach p95 213,728 / 216,150 / 222,930 ms; bestehender verschachtelter Pfad 218,188 / 211,365 / 212,939 ms (alternierend, 1 Warm-up + 10 Messungen) | Kernel-Tests 7/7 PASS; End-to-End nicht nötig, da langsamer | verworfen |

Der Kandidat bleibt ausschließlich als expliziter A/B-Schalter
`DA3_KERNELS_FLAT_FLASH_TILES=1` im separaten Kernel-Repository. Er verändert
die F32-Operationen nicht, materialisiert aber rund 2,65 MiB K-Packs vor der
Berechnung und führt eine Phasenschranke ein. Das bringt auf dieser Maschine
keinen stabilen Netto-Vorteil und ist daher nicht der Standardpfad.

| 21 | Die AVX-512-Projektionskernel verarbeiten standardmäßig sechs Tokenzeilen je Registerblock. Vier oder acht Zeilen könnten Registerdruck bzw. Schleifen-Overhead günstiger ausbalancieren. | 4 Zeilen: 209,962 / 214,463 / 211,241 ms; 6 Zeilen: **200,863 / 204,999 / 199,135 ms**; 8 Zeilen: 212,631 / 204,529 / 210,445 ms (je 1 Warm-up + Median aus 10) | AVX-512-Projektions-Orakel für 4/6/8 Zeilen bitgleich, externer Kernel-Test 8/8 PASS | Standard bei 6 Zeilen belassen |

Der neue Schalter `DA3_KERNELS_LINEAR_ROWS=4|8` dient nur reproduzierbaren
Folgemessungen; nicht gesetzte oder ungültige Werte wählen weiterhin sechs
Zeilen. Der Nutzen einer abweichenden Tile-Größe ist klar nicht vorhanden.

| 22 | Die Winograd-F(2×2,3×3)-Arbeitseinheiten könnten mit weniger gleichzeitig gehaltenen Transform- und Produkttiles besser in den Zen-5-Worker-Cache passen. | 4 Tiles: 192,840 / 198,327 / 199,255 ms; 8 Tiles: 212,588 / 218,316 / 209,997 ms; 12 Tiles: 572,286 / 570,150 / 566,332 ms; 16 Tiles: 593,596 / 598,638 / 595,014 ms (je 1 Warm-up + Median aus 10) | Vier Bilder PASS, identische C++-PFM-Metriken | **4 Tiles als Standard behalten** |

Ein alternierender 4-vs-8-Lauf war durch starke Host-Varianz langsamer, hielt
aber in allen vier Paaren den Vorteil für vier Tiles (264,036 < 270,071;
262,271 < 275,103; 215,943 < 231,123; 218,873 < 221,053 ms). Der neue
Standard berührt ausschließlich die Anzahl parallel geplanter, disjunkter
Tiles; Transforms, FMA-Reihenfolge und Ausgaben sind identisch.

| 23 | Die für Attention-Output und FC2 fusionierten Bias+LayerScale-Projektionen sollten den bei rohen Projektionen erfolgreichen 64-Spalten-Scheduler übernehmen. | Bestehender Zeilenpfad 195,663 / 195,368 / 200,765 ms; Spaltenpfad 220,866 / 223,776 / 218,343 ms (alternierend, 1 Warm-up + Median aus 10) | AVX-512-Orakel bitgleich; End-to-End nicht nötig, weil klar langsamer | verworfen |

Der Schalter `DA3_KERNELS_BIAS_SCALE_COLUMN_SPLIT=1` bleibt nur für
reproduzierbare Gegenmessungen vorhanden. Die Fusion verändert den
Speicherzugriff gegenüber dem rohen GEMM genug, dass die Zeilenaufteilung auf
dieser Hardware die bessere Wahl ist.

| 24 | Der finale 64→32-3×3-Head-Conv auf 504×336 ist ein großer F(2×2)-Winograd-Pfad. F(4×4) reduziert die Transform-Domain-Produkte pro Ausgabepixel, darf aber wegen anderer Transform-Koeffizienten nur mit der bestehenden End-to-End-F32-Schranke akzeptiert werden. | F(4): **200,898 ms**; fusioniertes F(2): **203,980 ms** (je 5 alternierende Läufe, 1 Warm-up + Median aus 10) | Vier Bilder PASS: Canyon r=0,9999936 / MAE=0,0018125; Desk r=0,9999783 / 0,0017725; Mountains r=0,9999856 / 0,0036750; Street r=0,9999721 / 0,0008209 | opt-in behalten; erst mit AVX-512-Produktkernel für eine Finalstudie qualifizieren |

Iteration 24 ist absichtlich durch `DA3_WINO_F4_FINAL=1` opt-in und auf die
feste DA3-BASE-Endgeometrie `(64,288,192) → (64,504,336)` begrenzt. Sie hält
den bilinearen Resize materialisiert, damit der Vergleich nur den
Convolution-Algorithmus ändert; der F(2)-Resize-Fusionspfad bleibt die
Kontrolle. Der direkte F4-gegen-naive-3×3-Orakeltest liegt innerhalb 0,003
maximaler F32-Abweichung. Der aktuelle F4-Produktkern ist skalar: Das
Smoke-Ergebnis rechtfertigt deshalb einen separaten AVX-512-Kandidaten, aber
noch keinen Anspruch auf einen finalen Benchmark-Sieg.

| 25 | Die Zeitaufteilung muss C++ und Rust nach denselben logischen Grenzen vergleichen, bevor weitere Kernel priorisiert werden. | C++ unfused Median: Preprocess **3,7 ms**, Backbone **141,7 ms**, Head **101,1 ms**; Rust warmes Profil: Preprocess ca. **5 ms**, Backbone ca. **135 ms**, Head ca. **60 ms** | nicht anwendbar (nur Instrumentierung/Profilierung) | Architekturpriorität: Fusion der Rust-Backbone/Head-Ausführung untersuchen |

Die C++-CLI-Referenz verwendet im eigentlichen fairen Benchmark standardmäßig
einen einzigen fusionierten ggml-Graphen; sie kann deshalb nur die kombinierte
Graphzeit ausgeben. Für die Segmentmessung wurde ausschließlich
`DA_FUSED=0` verwendet. Das misst den C++-Backbone und -Head exakt, ist aber
nicht mit der 240,928-ms-Fusionsreferenz zu verwechseln. Die Messung zeigt
dennoch den entscheidenden Unterschied: Rust ist in beiden entkoppelten
Rechensegmenten bereits konkurrenzfähig, materialisiert aber die vier
Backbone-Features zwischen getrennten Ausführungsphasen. Ein Rust-Plan, der
diese Datenübergabe in einem persistenten Plan/Workspace hält, ist damit der
größte noch offene Architekturhebel.

| 26 | Die DPT-Stufen kopieren jeweils ein 865×1536-Feature für LayerNorm und erzeugen danach nochmals CHW. Ein direkter, zeilenparalleler LayerNorm→CHW-Pass sollte vier Clone-Buffer (zusammen ca. 20,3 MiB) entfernen. | Erster A/B-Smoke: 201,682 vs. 206,114 ms und 194,346 vs. 201,976 ms; stärkere 5-Paar-Wiederholung: Kontrolle **195,796 ms**, Kandidat **198,755 ms** im Mittel | Direkter handoff-Orakeltest bitgleich; Vier Bilder gegen C++ PASS (niedrigstes r=0,9999721; höchste MAE=0,0036750) | verworfen – **2,959 ms langsamer** in der Wiederholung |

Iteration 26 bleibt ausschließlich als Schalter `DA3_FUSE_HEAD_TOKEN_NORM_CHW=1`
für reproduzierbare Gegenmessungen erhalten. Sie demonstriert, warum ein
Speichertraffic-Argument ohne einen stabilen alternierenden Messwert nicht
ausreicht: Die beibehaltene CHW-Ausgabe lässt den nachfolgenden 1×1-GEMM-Pfad
unverändert, während die neue Task- und Schreibverteilung auf Zen 5 den
vermuteten Kopiervorteil überkompensiert. Keine Vollstudie wurde durchgeführt.

| 27 | Die MLP-Projektionen FC1 und FC2 dominieren nach Attention weiterhin die Backbone-Zeit. Andere Zeilenkacheln könnten den Register-/Scheduler-Kompromiss verbessern. | Profil über alle 12 Blöcke: FC2 Standard-6-Zeilen **33,129 ms**; 4 Zeilen 38,566 ms (+16,4 %); 8 Zeilen 36,046 ms (+8,8 %) | Kernel-Orakel/Build PASS; keine End-to-End-Parität nötig, da klar langsamer | verworfen |

FC1 und FC2 verwenden bereits den spezialisierten externen AVX-512-Pfad;
sie sind keine Faer- oder generische-Rust-GEMM-Regressionsquelle mehr. Ein
dauerhaftes Gewichts-Prepack wurde ebenfalls verworfen: Die K-major-
64-Spaltenpanels sind bereits passend angeordnet und L2-resident. Weitere
Mikrovarianten werden erst nach einem tatsächlich anderen GEMM-Backend
untersucht.

| 28 | OpenBLAS/OpenMP könnte die DA3-MLP-Formen auf dem Ryzen schneller ausführen als der eigene AVX-512-Pfad. | Eigenes Shape-Mikrobenchmark-Signal war gut; End-to-End über 5 alternierende Paare: Standard **198,174 ms**, OpenBLAS **197,296 ms** (nur 0,878 ms / 0,44 %) | Rust-gegen-Rust: r=0,9999999999997624, MAE=0,0000003902, max=0,000005245; damit deutlich innerhalb der F32-Grenzen | verworfen |

Die optionale Anbindung (`DA3_OPENBLAS=1` beim Build,
`DA3_KERNELS_OPENBLAS_FC=1` zur Laufzeit) bleibt deaktiviert. Sie setzt die
OpenMP-Gruppe ausdrücklich auf 16 Threads und wird nur außerhalb von
Rayon-parallelen Bereichen auf die exakten FC1/FC2-Formen angewandt. Obwohl
die Einzel-GEMMs schnell sind, heben Thread-Team-Overhead und die bereits
effiziente integrierte Kachelstrategie den erwarteten End-to-End-Vorteil
nahezu auf. Ein solcher Bibliothekswechsel wird nicht als Performance-Sieg
gezählt.

| 29 | F(4×4)-Winograd kann für die acht großen 128→128-ReLU-Residual-Convolutions des DPT-Heads die Transform-Domain-Produkte um 43,75 % pro Ausgabepixel senken. | 5 Paar-Smoke auf dem Workhorse: kein stabiler Vorteil; abgesehen vom ersten gestörten Paar im Mittel etwa **+0,6 ms** langsamer, ohne dieses Paar etwa +2,5 ms | Direkter F4-Orakeltest innerhalb der F32-Hülle; neuer ReLU-Eingangsübergang bitgleich | verworfen |

Die F4-Ausführung wurde nur für die großen rn1/rn2-Residual-Convs als
`DA3_WINO_F4_HEAD=1` getestet. Anders als der isolierte finale F4-Conv
dominiert hier die 6×6-Eingangs-/Ausgangstransformation gegenüber der
reduzierten Produktarbeit. Der breite Runtime-Patch wird zurückgenommen;
der bereits separat vorhandene finale F4-Proof bleibt davon unberührt.

| 30 | Der reale ggml-Attention-Kern verwendet echte 64×64 Query/KV-Tiles und dynamische Verteilung über etwa 168 Head/Query-Jobs; Rust verwendete bisher Query-Tiles bis 20. | GGML64-Kern bitgleich, aber 5 Paar-Smoke: Standard **197,811 ms**, GGML64 **201,564 ms** (+3,753 ms / +1,90 %) | AVX-512-Kernorakel gegen acht QT8-Tiles bitgleich PASS | verworfen |

Der opt-in Schalter `DA3_KERNELS_FLASH_GGML64=1` bleibt nur für spätere
Gegenmessungen erhalten. Der Quellvergleich korrigierte eine falsche alte
Annahme über die ggml-Kachelgröße, aber der isoliert portierte Kern beweist
auch: Der ggml-Vorteil kommt nicht allein aus 64×64-Tiling. Er muss im
modellweiten Graph-/Arena-/Threadplan oder dessen Operator-Zusammenspiel
liegen, weshalb die nächste Untersuchung diesen Umfang adressiert.

| 31 | Attention-Ausgang liegt Head-major vor, wird aber vor `attn_proj` in Token-major entpackt und anschließend nochmals als Projektionsergebnis plus separatem Residual materialisiert. Ein direkter HND→Projection→Residual-Kern entfernt diese drei Übergaben. | 5 Paar-Smoke: Standard **197,631 ms**, direkter HND-Pfad **199,128 ms** (+1,497 ms / +0,76 %) | AVX-512-Oracle gegen Entpacken + bestehende Projektion/Epilog + Residual bitgleich PASS | verworfen |

Der Schalter `DA3_KERNELS_HND_PROJ_RESIDUAL=1` bleibt deaktiviert. Das
Head-major Aktivierungslayout verschlechtert die Gewicht-/Aktivierungslokalität
der anschließenden Projektion stärker als die eingesparten zwei 865×768
Puffer. Diese Messung setzt zugleich eine enge Obergrenze für reine
Backbone-Plan-/Buffer-Fusionen ohne andere Rechenform.

| 32 | LLVM-PGO kann die Ausführung auf dem festen Ryzen/DA3-BASE-Workload besser anordnen, ohne Rechenarbeit oder Benchmarkvertrag zu ändern. | Strenger alternierender Vergleich: Normal **198,628 ms**, PGO **285,726 ms** (+87,098 ms / +43,85 %) | nicht anwendbar; Build-Optimierung, kein Algorithmuswechsel | verworfen |

Der PGO-Korpus enthielt Canyon, Desk, Mountains und Street (je 1 Warm-up,
4 Wiederholungen); `rustc 1.97.1`/LLVM 22.1.6, das zusammengeführte Profil
und beide Binärhashes liegen isoliert unter `/tmp/da3-pgo` auf dem Workhorse.
Der Rückschritt ist weit außerhalb jeder Messvarianz. PGO wird weder als
Benchmarkkonfiguration noch als Kompensationsargument verwendet.

| 33 | QKV kann Q/K-LayerNorm und RoPE direkt beim 64-Spalten-Panel-Store ausführen und dadurch den separaten epilogischen Pass entfernen. | 5 Paar-Smoke: Standard **200,629 ms**, fusioniert **210,795 ms** (+10,166 ms / +5,07 %) | AVX-512-Oracle gegen QKV + bisherige QK-Norm/RoPE bitgleich PASS | verworfen |

Der Kandidat verlängert die QKV-Kritische-Route: Q/K-Zeilenarbeit verteilt
sich nur über 36 Projektionspanels, während der bestehende nachgelagerte Pass
seine vielen kurzen Zeilen effizient parallelisiert und nur etwa 1–2 ms
Gesamtbudget hat. Die Runtime-Anbindung wird vollständig zurückgenommen;
keine Fusionsbehauptung wird ohne eine stabile End-to-End-Messung behalten.

| 34 | Vollmaterialisierte Attention-Scores könnten die beiden Matrixprodukte als Faer-GEMMs schneller ausführen als der Online-Flash-Pfad. | Tatsächliche 12×865×64-Mikromessung mit 16 Threads, 3 Warm-ups + 30 alternierende Aufrufe: Online-AVX-512 Median **2,338 ms**, Full-Score Median **3,041 ms** (+30,10 %) | MAE=0,000000198, max. absolute Abweichung=0,000001967 zur Online-Referenz | verworfen |

Der Full-Score-Pfad verwendete eine wiederverwendete 35,9-MiB-Scorearena,
12 äußere Head-Jobs und Faer-GEMMs ohne verschachtelte Parallelität. Seine
Numerik wäre ausreichend, aber die zusätzliche Score-Bandbreite und
Softmax-Phase überwiegen jeden GEMM-Vorteil klar. Der Online-Flash-Algorithmus
bleibt der korrekte Ausgangspunkt für weitere Kernelarbeit.

| 35 | Die weitere Arbeit braucht einen nachweisbar identischen Quell-/Binärstand und Zähler erst nach Modell-Laden und Warm-up. | Neuer `scripts/run_hardware_profile.py` hält Quellen-, Kernel-, Modell-, Bild- und Binär-Hashes fest und startet `perf stat` verzögert. Smoke: Rust 50 Iterationen: **1.153.585** Page Faults / **203.095** Kontextwechsel / **12.240** Migrationen; C++: **4** / **2.672** / **788**. | Kein numerischer Pfad geändert. Die Rust-Cycle-Stichprobe liegt hauptsächlich in Bias+LayerScale-Projektionen (23,08 %), Flash (20,04 %), rohen Projektionen (13,20 %), QKV (9,39 %) und Winograd (16,37 % inklusive Scheduler). | Profilbasis behalten; noch keine Leistungsbehauptung |

Die Profilierung erfasst die Prozess-Lebensdauer erst ab drei Sekunden, damit
GGUF-Laden, Bilddekodierung und der ungemessene Warm-up-Durchlauf keine
Hardware-Zähler verfälschen. Die Seite-Fault-Stichprobe ordnet 57,37 % der
Rust-Faults den bilinearen DPT-Resizes und weitere 20,40 % den Winograd-
Arbeitseinheiten zu. Das ist ein konkreter Hinweis für eine spätere echte
Head-Pipeline, aber noch keine Aussage über den Anteil an der Wandzeit.

| 36 | Ein alternativer Prozess-Allocator könnte die extremen Rust-Page-Faults reduzieren. | TCMalloc reduzierte Rust bei 50 warmen Iterationen auf **6.786** Page Faults und war in fünf Paaren im Mittel etwa **5,55 ms** schneller (197,418 → 191,866 ms). C++ profitierte im selben Test jedoch stärker: 238,600 → 229,700 ms im Mittel (ca. **8,9 ms**). | Allokator ändert keine F32-Arithmetik, wurde aber nicht als Paritäts- oder End-to-End-Kandidat qualifiziert. | verworfen als Rust-spezifischer Vergleichsvorteil |

Der Test bleibt als Diagnosesignal bestehen: die persistenten Head-Puffer
verdienen Untersuchung. Für einen fairen Runtime-Vergleich kann ein allgemeiner
System-Allocator aber nicht nur für Rust als Sieg gerechnet werden, weil C++
denselben Vorteil mindestens ebenso nutzen kann.

| 37 | Die dynamischen Zeilen- und Blockschleifen im heißen 6×64-Bias+LayerScale-AVX-512-Kern könnten bei 144 vollen DA3-Tokenkacheln durch statisches Entrollen entfallen. | AVX-512-Bitorakel PASS. Fünf alternierende 1+10-Smokepaare: Kontrolle 204,331 / 200,859 / 198,991 / 203,131 / 204,378 ms, statisch 204,932 / 201,198 / 204,039 / 198,836 / 203,207 ms; Mittel **202,338 → 202,442 ms**. | Gleiche FMA-Reihenfolge nachgewiesen; vier-Bild-Gate nicht nötig, weil kein Geschwindigkeitsgewinn. | verworfen und vollständig entfernt |

Iteration 37 bestätigt den Disassembly-/Profiler-Grundsatz praktisch: Der
Compiler erzeugt für diesen Pfad bereits ausreichend gute Vollkachel-Codes;
manuelles Entrollen ist kein produktiver Ersatz für einen neuen Zen-5-
Makrokernel oder eine arithmetisch integrierte Head-Pipeline.

| 38 | Wiederholte Inferenzläufe sollen die vollständig überschriebenen DPT-Fusion-, Stage-, Resize- und Winograd-Aktivierungen wiederverwenden, statt bei jedem Video-Frame neue virtuelle Seiten zu faulten. | Vollständige 10×-Studie: Workspace Rust **195,986 ms** [194,540; 197,431], C++ **238,840 ms** [237,337; 240,343]. Direkte 10×-Kontrolle ohne Workspace: Rust **200,788 ms** [199,538; 202,039], C++ **238,173 ms** [236,871; 239,474]. Damit beträgt der isolierte Rust-Gewinn **4,803 ms** / **2,39 %**. | Vier C++-F32-PFM-Gates PASS: Canyon r=0,9999936280 / MAE=0,0018124937; Desk r=0,9999782659 / 0,0017725352; Mountains r=0,9999855793 / 0,0036750214; Street r=0,9999721277 / 0,0008208673. Synthetischer Wiederverwendungs-Orakeltest bitgleich PASS. | behalten |

Der Workspace ist bewusst kein allgemeiner Graph-/Arena-Ersatz: Er enthält
nur private Puffer, die der unmittelbar folgende Kernel vollständig
überschreibt. Debug-Captures sowie Tiefen- und Konfidenzausgabe behalten ihre
normale Ownership. Die Seite-Faults im 50-Iterationen-Profil sinken dadurch
gegenüber dem ursprünglichen Profil von 1.153.585 auf rund 0,4 Mio.; der
verbleibende Unterschied zeigt aber, dass reine Pufferwiederverwendung nicht
mehr die fehlenden rund 25 ms bis zum 40%-Ziel liefern kann.

**Provenienz Iteration 38:** Vollstudien liegen auf dem Workhorse unter
`/tmp/da3-cpu-f32-expanded-workspace-20260813/raw-results.json`
(SHA-256 `39211236419caa5275d48875f7a20ec7b419f63a614c4e9bfbf3fcd4a458afdf`)
und `/tmp/da3-cpu-f32-no-workspace-20260813/raw-results.json`
(`33933766946a1533691c42e5078969bba4e789fd44f5178093d61d78f5d38017`).
Der angenommene Kandidat wurde als `da`-Binary
`b77df0afb7192c0556ebd308a32cc16dba5961bea9ff8bf8451cafdc1e00686b`
gegen C++ `eba42df633ebc5f4f6c178e0c39e80054124a3591a49e4e7f8da1d73e81aece5`
gemessen; Modell und Bild behalten die gesperrten SHA-256-Werte
`1b13b166…c6b8da` und `936d60f4…b8c969`. Der Remote-Sync ist kein Git-
Checkout; die hierzu gehörigen Dateihashes sind daher als Benchmark-
Provenienz maßgeblich, nicht ein lokaler Git-HEAD.

| 40 | Ein Zen-orientiertes BLIS/AOCL-artiges SGEMM-Backend könnte die festen Transformer-Projektionen besser auslasten als die lokalen 6×64-AVX-512-Kerne. | Shape-Mikrobenchmarks zeigen deutliche Vorteile vor allem für 865×768×3072 und 865×768×768; der qualifizierte 10×-Gesamtlauf ergibt Rust **188,005 ms** [181,437; 194,574] gegenüber C++ **238,809 ms** [237,230; 240,387]. Das sind gegenüber der Workspace-Baseline 195,986 ms **−7,981 ms / −4,07 %**. | Vier C++-F32-PFM-Gates PASS: Canyon r=0,9999936278 / MAE=0,0018125444; Desk r=0,9999782570 / 0,0017729213; Mountains r=0,9999855778 / 0,0036751658; Street r=0,9999721236 / 0,0008210414. Zusätzlich validiert ein BLIS-Row-major-View-Oracle die transpositionsfreie Speicherinterpretation gegen naive SGEMM. | qualifizierter Zwischenstand; sportliches -30-ms-GEMM-Ziel nicht erreicht |

Der Kandidat bindet BLIS nur in einem expliziten Build (`da3_blis`) und nur
bei `DA3_KERNELS_BLIS_LINEAR=1` ein. Der normale Binarypfad bleibt unverändert.
Die Bibliothek war ausschließlich temporär unter `/tmp` auf dem Workhorse
gebaut; alle Hashes, Umgebung und Rohdaten sind in
`docs/benchmarks/2026-08-workhorse/iteration40-blis/` festgehalten. Wegen
der noch breiten Rust-Varianz wird dieser Gewinn nicht als Ende der GEMM-Wette
oder als 40%-Sieg ausgegeben. Die nächste Architekturwette folgt wie geplant:
eine echte DPT-Head-Pipeline mit einem sportlichen Ziel von mindestens 20 ms
End-to-End-Gewinn.

| 41 | Die großen 1×1-Projektionen des DPT-Heads laufen seriell und können deshalb einen BLIS-16-Thread-Teamaufruf nutzen, ohne verschachtelte Rayon-/Faer-Parallelität. | Vollständige 10×-Studie: Rust **181,138 ms** [175,623; 186,653], C++ **238,513 ms** [236,701; 240,325]. Gegenüber Iteration 38 sind das **−14,848 ms / −7,58 %**; Rust ist hier **31,67 % schneller** als C++. | Vier C++-F32-PFM-Gates PASS: Canyon r=0,9999936279 / MAE=0,0018125425; Desk r=0,9999782569 / 0,0017729257; Mountains r=0,9999855778 / 0,0036751673; Street r=0,9999721236 / 0,0008210428. | behalten als qualifizierter Zwischenstand; 40%-Ziel noch nicht erreicht |

Der warme Head-Profiler reduziert mit der Brücke den Head von ungefähr
58,4 ms auf 53,9 ms, hauptsächlich über die Stage-Projektionen; Fusion und
der bereits fusionierte finale Resize+F2-Operator bleiben dominant. Der
Kandidat verwendet ausschließlich eine explizite Runtime-Umgebung und einen
temporär gebauten BLIS-Link. Die vollständige Provenienz liegt unter
`docs/benchmarks/2026-08-workhorse/iteration41-blis-head/`. Beim aktuellen
C++-Mittel von 238,513 ms entspricht das 40%-Ziel 170,367 ms; es fehlen noch
10,771 ms. Daher bleibt die Head-Pipeline-Wette aktiv, statt diesen
Backend-Gewinn als Abschluss zu behandeln.

| 39 | Bilineares Resize und eine räumlich geteilte 1×1-Projektion sind algebraisch vertauschbar. Die Projektion vor dem Resize hätte insbesondere in den großen RefineNet-Stufen deutlich weniger MACs. | Fünf alternierende 1+10-Smokepaare: Kontrolle **197,107 / 198,431 / 204,047 / 198,134 / 195,222 ms**, Kandidat **201,284 / 200,197 / 194,812 / 200,114 / 190,648 ms**; Mittel **198,588 → 197,411 ms** (−1,177 ms / −0,59 %). | Synthetischer Head-Test innerhalb des F32-MAE-Limits PASS. Wegen des kleinen, stark streuenden Smoke-Vorteils kein Vier-Bild- oder Vollbenchmark. | verworfen und vollständig entfernt |

Die algebraische Reduktion der 1×1-Arbeit überträgt sich nicht zuverlässig auf
die Modelllatenz: Die zusätzliche Reihenfolge der Zwischenaktivierungen und
die Resize-Arbeit dominieren. Der Test bleibt nicht als Runtime-Schalter im
Quellstand, damit der akzeptierte Workspace-Baselinepfad eindeutig bleibt.

| 42 | Nach BLIS-SGEMM könnte ein separater AVX-512-/Rayon-Bias+LayerScale-Epilog die noch serielle Nacharbeit der Transformer-Projektionen beschleunigen. | Fünf alternierende 1+10-Smokepaare auf freiem Workhorse: Baseline **182,215 / 172,937 / 171,520 / 171,686 / 194,686 ms**, Vektor-Epilog **176,936 / 175,505 / 177,828 / 184,974 / 178,608 ms**; Mittel **178,609 → 178,770 ms** (**+0,161 ms / +0,09 %**). | Der direkte Element-Oracle war bitgleich PASS; die Messreihe zeigt dennoch keinen Gewinn. | verworfen und vollständig entfernt |

Der Epilog ist zwar arithmetisch trivial und numerisch sicher, liegt aber
außerhalb des dominanten BLIS-Teils und erzeugt eine zusätzliche Rayon-Phase.
Ohne belastbaren End-to-End-Gewinn darf er nicht als Produktionspfad bleiben.

| 43 | Das materialisierte finale bilineare Resize könnte auf Zen 5 trotz zusätzlichem 41-MiB-Zwischenpuffer speicherlokaler als das eingebettete Resize im F(2)-Inputtransform sein. | Vollständige 10×-Studie: Rust **185,855 ms** [181,940; 189,771], C++ **238,912 ms** [237,365; 240,458]. Der vorherige fünf-Paar-Smoke war irreführend; gegenüber der akzeptierten Iteration 41 (**181,138 ms**) ist der Kandidat **+4,718 ms** langsamer. | Vier-Bild-C++-F32-Parität PASS, aber die End-to-End-Studie widerlegt den vermeintlichen Smoke-Gewinn. | verworfen; fusionierter Produktionspfad bleibt aktiv |

Die Iteration ist ein bewusstes Gegenbeispiel gegen das Optimieren auf kurze
Smokes: Die materialisierte Variante gewann dort alle fünf Paare, verlor aber
im voll randomisierten 10×-Protokoll klar gegen den bestehenden Fusionspfad.

| 44 | Ein 6×64-AVX-512-Mikrokernel könnte die QK- und PV-Produkte im Flash-Pfad mit mehr Query-Zeilen pro Registerblock beschleunigen. | Fünf alternierende 1+10-Smokepaare: 4-Zeilen-Kontrolle **174,035 / 177,960 / 179,211 / 175,598 / 174,628 ms**, 6-Zeilen-Kandidat **190,111 / 177,020 / 184,513 / 179,032 / 193,368 ms**; Mittel **176,286 → 184,809 ms** (**+8,523 ms / +4,84 %**). | Die Variante behält FMA- und Softmax-Reihenfolge bei, ist aber durch höheren Registerdruck deutlich langsamer. | verworfen; Standard bleibt 4×64 |

| 45 | Wiederverwendete AVX-512-Exponentialkonstanten pro Flash-Query-Tile könnten den Softmax-Overhead reduzieren. | Bitorakel PASS. Fünf alternierende 1+10-Smokepaare: Kontrolle **193,808 / 185,465 / 173,892 / 195,299 / 172,491 ms**, Kandidat **186,243 / 180,653 / 181,036 / 181,120 / 186,405 ms**; Mittel **184,191 → 183,091 ms** (**−1,100 ms / −0,60 %**). | Zu klein und zu streuend für das Flash-Ziel von mindestens 11 ms; kein Vier-Bild- oder Vollbenchmark. | verworfen und vollständig entfernt |

| 46 | Ein 8×32-AVX-512-Flash-Mikrokernel könnte die 4×64-Kachel ohne 6×64-Registerdruck ersetzen und K/V-Panels über acht Queries teilen. | AVX-512-Bitorakel und vier C++-F32-PFM-Gates PASS. Vollstudie als Opt-in: Rust **170,903 ms** [165,207; 176,599], C++ **239,364 ms** [238,301; 240,427]; Punktwert **40,06 %** schneller. | Der Kandidat ist arithmetisch bitgleich zum 4×64-Kern; die Studie zeigt jedoch hohe Rust-Varianz. | als Standardpfad übernommen, anschließend separat bestätigt |

| 47 | Der 8×32-Kern muss als Standardpfad ohne Kandidatenschalter dieselbe vollständige Messung bestehen. | Vollstudie: Rust **175,116 ms** [171,285; 178,946], C++ **237,590 ms** [235,915; 239,264], Punktwert **35,68 %** schneller. Vier-Bild-C++-F32-Parität PASS. | Der Unterschied zur Opt-in-Studie liegt innerhalb der deutlich breiteren Rust-Varianz; für das neue 50%-Ziel gilt konservativ diese Studie als Ausgangswert. | behalten, aber nicht als 40%-Sieg ausgeben |

| 48 | BLIS könnte die 865×768×2304-QKV-Projektion schneller ausführen, falls ein persistent wiederverwendeter Token-major-Handoff die Layoutkosten begrenzt. | Direkter Layouttest innerhalb F32-Hülle PASS. Fünf alternierende 1+10-Smokepaare: Kontrolle **174,160 / 177,109 / 173,003 / 175,750 / 173,843 ms**, BLIS-QKV **175,207 / 180,214 / 185,218 / 178,146 / 179,559 ms**; Mittel **174,773 → 179,669 ms** (**+4,896 ms / +2,80 %**). | Das 8-MiB-Staging und die Head-major-Transposition überwiegen den SGEMM-Vorteil. | verworfen und vollständig entfernt |

| 49 | Der direkte QKV-6×64-Kern könnte wie Flash von einem 8×32-Tile mit geringerer Registerlast profitieren. | AVX-512-Bitorakel PASS. Fünf alternierende 1+10-Smokepaare: Kontrolle **179,838 / 163,806 / 168,891 / 165,001 / 181,440 ms**, 8×32 **186,335 / 169,402 / 175,240 / 176,239 / 186,767 ms**; Mittel **171,795 → 178,797 ms** (**+7,002 ms / +4,08 %**). | Bei der 768-langen QKV-Reduktion kostet die halbierte Output-Panelbreite mehr als sie an Registerdruck spart. | verworfen und vollständig entfernt |

| 50 | Ein einmalig gepackter BLIS-Gewichtspfad könnte die wiederholte Gewichtspackung der unveränderlichen Modellmatrizen eliminieren. | Die vorhandene temporäre BLIS-Bibliothek exportiert zwar `sgemm_pack*` und `sgemm_compute*`, aber weder ihre deklarierte CBLAS-Pack-API noch eine native Pack/Compute-Kombination erzeugte bei den echten DA3-Formen numerisch korrekte Ergebnisse (native Probe: maximale Abweichung 15,54). Der normale BLIS-Row-major-Aufruf bleibt korrekt. | Keine Runtime-Integration und kein Paritäts-Gate, weil schon der isolierte numerische Oracle fehlschlug. | verworfen; erst mit einer verifizierbaren BLIS/AOCL-Pack-API neu bewerten |

| 51 | Die 16 F(2,3×3)-Transformpositionen der finalen 64→32-Winograd-Convolution könnten als große BLIS-GEMMs über alle 42.336 Tiles schneller als der bestehende Vektorpfad laufen. | Der isolierte Probe enthält nur die unvermeidliche position-major→tile-major-Aktivierungsanordnung und die 16 BLIS-Produkte, nicht einmal die räumlichen Vor- oder Rücktransformationen: **21,863 / 21,071 / 20,917 ms** nach Warm-up. Der aktuelle fusionierte finale Resize+F2-Ausgabeabschnitt liegt bereits bei rund 11 ms inklusive zusätzlicher Arbeit. | Kein Runtime-Kandidat; die reine Untermenge ist schon deutlich langsamer. | verworfen und Probe entfernt |

| 52 | Größere F(2)-Winograd-Tilegruppen könnten die dominanten `rn1`-Residual-Convolutions durch weniger Scheduler-Overhead beschleunigen. | Sweep 4/6/12/16: 12 und 16 degenerieren auf **704–896 ms**. Fünf alternierende 4-vs-6-Paare: 4-Tile **174,734 / 168,896 / 165,915 / 175,013 / 166,753 ms**, 6-Tile **165,420 / 164,998 / 179,267 / 181,044 / 168,481 ms**; Mittel **170,262 → 171,842 ms**. | Arithmetik unverändert; keine Paritätsprüfung nötig, weil die robuste A/B-Reihe keinen Gewinn zeigt. | verworfen; Standard bleibt vier Tiles |

| 53 | Die bestehende Vier-Tile-Jobbündelung im aktuellen Head-Pfad muss als frische Gesamtbasis gegen C++ und auf allen vier Bildern reproduziert werden. | Vollstudie: Rust **170,288 ms** [165,776; 174,800], C++ **238,353 ms** [236,449; 240,257]. Das entspricht **28,55 %** weniger Latenz; für einen 50%-Speedup fehlen noch **11,386 ms** bis 158,902 ms. Die Vier-Bilder-PFM-Gates PASS: Canyon r=0,9999936279/MAE=0,0018125425; Desk 0,9999782569/0,0017729257; Mountains 0,9999855778/0,0036751673; Street 0,9999721236/0,0008210428. | Die aktuelle Standardbündelung bleibt aktiv; sie ist keine neue verdeckte Kandidatenumgebung. Rohdaten: Workhorse `/tmp/da3-cpu-f32-winograd-min4-20260819/raw-results.json`. | qualifizierte aktuelle Basis, Ziel offen |

| 54 | Das Falten der Residualaddition in den inverse Winograd-Store könnte pro RefineNet-Residualblock einen vollständigen CHW-Add-Pass entfernen. | Direkter Orakeltest bitgleich. Fünf alternierende 1+10-Paare: Kontrolle **169,118 / 165,054 / 162,681 / 183,759 / 168,174 ms**, Fusion **173,834 / 166,522 / 174,265 / 183,082 / 184,783 ms**; Mittel **169,757 → 176,497 ms**. | Die inner-loop-Residual-Lesezugriffe überwiegen den entfernten Vector-Add klar; keine Vier-Bild- oder Vollstudie gerechtfertigt. | verworfen und vollständig zurückgebaut |

| 55 | Der normale QT8-Flash-Kernel reserviert auch im Produktionspfad Stack für diagnostische K-/V-Fallback-Puffer. Ein separater persistent-gepackter QT8-Kern könnte diese unbenutzte Last entfernen, ohne die Flash-Arithmetik zu ändern. | Vollständige 10×-Studie: Rust **165,751 ms** [161,038; 170,464], C++ **238,647 ms** [236,902; 240,392]. Das sind **1,440× bzw. 43,98 %** höhere Rust-Geschwindigkeit und gegenüber Iteration 53 **−4,537 ms / −2,66 %** Rust-Latenz. | AVX-512-Orakel gegen den bisherigen QT8-Kern ist bitgleich, auch für das 33-Key-Schlusspanel. Vier C++-F32-PFM-Gates PASS: Canyon r=0,9999936279 / MAE=0,0018125425; Desk 0,9999782569 / 0,0017729257; Mountains 0,9999855778 / 0,0036751673; Street 0,9999721236 / 0,0008210428. | behalten; 50%-Ziel noch offen |

Der neue Kern läuft ausschließlich für vollständige Acht-Query-Tiles mit
persistenter, dimension-major gepackter K-Matrix. Er behält die aufsteigende
Key-/K-Dimension-FMA-Reihenfolge, die Online-Softmax-Reihenfolge sowie den
64-Schritt-Zero-Tail für V bei. Das abschließende 1-Query-Tile bleibt im
generischen, bereits bewährten Pfad. Der alternative Pfad bleibt für
gleichbinäre A/B-Messungen über `DA3_KERNELS_DISABLE_FLASH_PACKED_QT8=1`
abschaltbar.

Beim aktuellen C++-Mittel entspricht das sportliche 50%-Speedup-Ziel einer
Rust-Latenz von höchstens **159,098 ms**. Es fehlen damit noch **6,653 ms**;
der breite Rust-Konfidenzbereich ist ausdrücklich kein Zielerfolg. Die
vollständigen Rohdaten liegen auf dem Workhorse unter
`/tmp/da3-cpu-f32-packed-flash-20260819/raw-results.json`.
