# Optimierungs-Log

Jede Kernel-/Komponenten-Task trägt hier nach der Zwei-Iterationen-Regel (Spec §6.3) ein.

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
