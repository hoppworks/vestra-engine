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
