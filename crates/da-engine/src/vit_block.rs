//! A single DINOv2/DA3 ViT transformer block (`vit_block`) and its weight
//! tensor naming convention.
//!
//! Block structure (verified against the real C++ reference,
//! `../src/vit_block.cpp`/`../src/attention.cpp` — see the module-level
//! doc comments on `da_graph::graph::Op::Attention` for the two traps this
//! implements):
//!
//! ```text
//! x -> LN1(ln_eps) -> Attention[+qk_norm if i>=qknorm_start][+RoPE if i>=rope_start]
//!   -> [scale by ls1 if present] -> x += (that)
//!   -> LN2(ln_eps) -> Linear(fc1) -> GELU(erf) -> Linear(fc2)
//!   -> [scale by ls2 if present] -> x += (that)
//! ```
//!
//! ## Weight tensor names
//!
//! Confirmed against two independent real sources (not guessed): the GGUF
//! converter's renaming table (`../scripts/gguf_keys.py::rename_backbone`)
//! and the C++ reference's loaders (`../src/vit_block.cpp::load_block`,
//! `../src/attention.cpp::load_attn`). For layer `i`, under the
//! `vit.blk.{i}.` prefix:
//!
//! - `norm1.weight` / `norm1.bias`, `norm2.weight` / `norm2.bias` — block LN.
//! - `attn_qkv.weight` / `attn_qkv.bias` — fused QKV linear (`out_features =
//!   3*embed_dim`, column order `[Q(embed) | K(embed) | V(embed)]`, each
//!   `embed`-wide block itself `[heads, head_dim]` head-major with
//!   `head_dim` contiguous per head — confirmed from `attention.cpp`'s
//!   `ggml_reshape_4d(qkv, D, H, 3, tok)`).
//! - `attn_proj.weight` / `attn_proj.bias` — attention output projection.
//! - `attn_qnorm.weight` / `attn_qnorm.bias`, `attn_knorm.weight` /
//!   `attn_knorm.bias` — per-head QK-LayerNorm, present only on models that
//!   use it (absent tensor => qk-norm skipped regardless of `qknorm_start`).
//! - `ls1`, `ls2` — LayerScale gammas, length `embed_dim`. Presence-gated:
//!   if absent, that residual branch is added unscaled.
//! - `mlp_fc1.weight` / `mlp_fc1.bias`, `mlp_fc2.weight` / `mlp_fc2.bias` —
//!   the classic MLP FFN (`ffn_type == "mlp"`, DA3-BASE). `ffn_type ==
//!   "swiglu"` (giant models, `mlp_w12`/`mlp_w3` tensors) is a deliberate,
//!   honest not-yet-supported hard error here — see [`vit_block`].
//!
//! ## Linear-weight orientation: a documented convention, not yet real data
//!
//! `da_graph::Op::Gemm` computes `out[m,n] = a[m,k] @ b[k,n]` (see
//! `da-graph/tests/graph_runs_linear.rs`, the only existing exerciser of
//! this op before this task). This module's `run_linear` helper always
//! passes the token activations as `a` (`[n_tok, in_features]`) and the
//! named weight tensor as `b`, which therefore must already be laid out
//! `[in_features, out_features]` — i.e. the **transpose** of the raw
//! PyTorch/GGUF `nn.Linear.weight` layout (`[out_features, in_features]`,
//! confirmed unmodified by `../scripts/convert_da3_to_gguf.py`, which saves
//! tensors via `np.ascontiguousarray` with no transpose). Real GGUF weight
//! loading is Task 20's job (not this task's — no dumps/real weights exist
//! in this environment); whichever code populates a real `Weights` map for
//! this module's linear tensors (`attn_qkv`, `attn_proj`, `mlp_fc1`,
//! `mlp_fc2`) **must transpose them from GGUF's `[out,in]` layout into this
//! module's expected `[in,out]` layout first** — this is not merely this
//! module's private assumption but the same orientation `Op::Gemm` already
//! requires everywhere else in `da-graph`.
use crate::ModelConfig;
use da_graph::{Backend, Weights};
use vestra_kernels::gemm::{Da3ProjectionGemm, Gemm};

/// Execution seam for the two transformer residual additions. The normal
/// production path supplies `None` and keeps the optimized CPU add. A CUDA
/// implementation is intentionally an opt-in parity slice until all adjacent
/// operators can remain device-resident.
pub trait ResidualAddExecutor: Send + Sync {
    fn add_in_place(&self, destination: &mut [f32], source: &[f32]);
}

#[cfg(feature = "cuda-residual-oracle")]
impl ResidualAddExecutor for vestra_kernels::cuda::CudaRuntime {
    fn add_in_place(&self, destination: &mut [f32], source: &[f32]) {
        let mut destination_device = self
            .upload_f32(destination)
            .expect("CUDA residual destination upload must succeed");
        let source_device = self
            .upload_f32(source)
            .expect("CUDA residual source upload must succeed");
        self.add_f32_in_place(&mut destination_device, &source_device)
            .expect("CUDA residual kernel must succeed");
        let result = self
            .download_f32(&destination_device)
            .expect("CUDA residual result download must succeed");
        destination.copy_from_slice(&result);
    }
}

fn add_residual(
    destination: &mut [f32],
    source: &[f32],
    executor: Option<&dyn ResidualAddExecutor>,
) {
    if let Some(executor) = executor {
        executor.add_in_place(destination, source);
    } else {
        vestra_kernels::Kernels::detect().add(destination, source);
    }
}

/// The per-head QK-LayerNorm epsilon (trap #1 from the Task 17 brief):
/// torch's *default* `nn.LayerNorm` eps, **not** the block's `ln_eps`. See
/// `da_graph::graph::Op::Attention`'s doc comment and
/// `../src/vit_block.cpp`'s "reference parity note" comment
/// (`QK_NORM_EPS = 1e-5f`) for the full provenance.
pub const QK_NORM_EPS: f32 = 1e-5;

fn wname(layer_idx: usize, suffix: &str) -> String {
    format!("vit.blk.{layer_idx}.{suffix}")
}

/// Runs `x[rows,cols] -> LayerNorm(x, gamma, beta, eps)` directly on an
/// activation copy.  The model parameters are immutable and borrowed from
/// `Weights`; copying them into a per-operation graph arena was pure
/// overhead on the inference path.
fn run_layernorm(
    x_in: &[f32],
    rows: usize,
    cols: usize,
    gamma_name: &str,
    beta_name: &str,
    eps: f32,
    weights: &Weights,
) -> Vec<f32> {
    let mut out = x_in.to_vec();
    let gamma = weights
        .get_f32(gamma_name)
        .unwrap_or_else(|| panic!("Weights missing f32 entry {gamma_name:?}"));
    let beta = weights
        .get_f32(beta_name)
        .unwrap_or_else(|| panic!("Weights missing f32 entry {beta_name:?}"));
    vestra_kernels::scalar::layernorm(&mut out, rows, cols, gamma, beta, eps);
    out
}

/// Runs `y = x[m,k] @ w[k,n] + bias[n]`, optionally followed by GELU and/or
/// an in-place LayerScale (`y *= ls_gamma[n]`, only if `ls_name` is `Some`
/// *and* that tensor is actually present in `weights` — presence-gated,
/// trap #3).
#[allow(clippy::too_many_arguments)]
fn run_linear(
    x_in: &[f32],
    m: usize,
    k: usize,
    n: usize,
    w_name: &str,
    b_name: &str,
    gelu: bool,
    ls_name: Option<&str>,
    weights: &Weights,
) -> Vec<f32> {
    let weight = weights
        .get_f32(w_name)
        .unwrap_or_else(|| panic!("Weights missing f32 entry {w_name:?}"));
    let bias = weights
        .get_f32(b_name)
        .unwrap_or_else(|| panic!("Weights missing f32 entry {b_name:?}"));
    let mut out = vec![0.0; m * n];
    if !gelu {
        if let Some(name) = ls_name {
            if let Some(gamma) = weights.get_f32(name) {
                if vestra_kernels::linear_bias_scale_f32_da3_base(
                    m, n, k, x_in, weight, bias, gamma, &mut out,
                ) {
                    return out;
                }
            }
        }
    }
    Da3ProjectionGemm.gemm(m, n, k, x_in, weight, &mut out);
    vestra_kernels::scalar::add_bias_rows(&mut out, m, n, bias);
    if gelu {
        vestra_kernels::Kernels::detect().gelu(&mut out);
    }
    if let Some(name) = ls_name {
        if let Some(gamma) = weights.get_f32(name) {
            vestra_kernels::scalar::layerscale(&mut out, m, n, gamma);
        }
    }
    out
}

/// Runs the attention sub-block: fused QKV linear -> split/transpose into
/// per-head layout -> `Op::Attention` (optional qk-norm, optional RoPE,
/// scaled-dot-product core) -> output projection (+ `ls1` if present). Returns
/// the projected token-major buffer ready for the residual addition.
///
/// `gh`/`gw` (the patch-grid resolution) are only used to derive RoPE
/// positions when `rope` is active; they're ignored otherwise (including
/// whenever `cfg.rope_start < 0`, i.e. RoPE never used by this model).
#[allow(clippy::too_many_arguments)]
fn run_attention(
    ln1_out: &[f32],
    n: usize,
    gh: usize,
    gw: usize,
    global: bool,
    view_count: usize,
    cfg: &ModelConfig,
    layer_idx: usize,
    weights: &Weights,
) -> Vec<f32> {
    let phase_profile = std::env::var_os("DA_PHASE_PROFILE").is_some();
    let embed = cfg.embed_dim as usize;
    let heads = cfg.num_heads as usize;
    let head_dim = cfg.head_dim as usize;

    // Split the fused per-token [Q(embed)|K(embed)|V(embed)] row (each
    // embed-wide block itself [heads,head_dim] head-major) and transpose
    // token-major [n, heads, head_dim] -> head-major [heads, n, head_dim],
    // the layout `Op::Attention` requires. Pure data movement, done in
    // host Rust rather than as a graph op (see module doc: reshape/permute
    // isn't part of the approved Op set this task touches).
    let mut q = vec![0f32; heads * n * head_dim];
    let mut k = vec![0f32; heads * n * head_dim];
    let mut v = vec![0f32; heads * n * head_dim];
    let qkv_started = std::time::Instant::now();
    let qkv_weight = weights
        .get_f32(&wname(layer_idx, "attn_qkv.weight"))
        .unwrap();
    let qkv_bias = weights.get_f32(&wname(layer_idx, "attn_qkv.bias")).unwrap();
    let direct_qkv =
        vestra_kernels::qkv_f32_da3_base(ln1_out, qkv_weight, qkv_bias, &mut q, &mut k, &mut v);
    let qkv_elapsed = qkv_started.elapsed();
    let pack_elapsed;
    if direct_qkv {
        pack_elapsed = std::time::Duration::ZERO;
    } else {
        let qkv = run_linear(
            ln1_out,
            n,
            embed,
            3 * embed,
            &wname(layer_idx, "attn_qkv.weight"),
            &wname(layer_idx, "attn_qkv.bias"),
            false,
            None,
            weights,
        );
        let pack_started = std::time::Instant::now();
        for t in 0..n {
            let row = &qkv[t * 3 * embed..(t + 1) * 3 * embed];
            for h in 0..heads {
                let dst = (h * n + t) * head_dim;
                let src = h * head_dim;
                q[dst..dst + head_dim].copy_from_slice(&row[src..src + head_dim]);
                k[dst..dst + head_dim].copy_from_slice(&row[embed + src..embed + src + head_dim]);
                v[dst..dst + head_dim]
                    .copy_from_slice(&row[2 * embed + src..2 * embed + src + head_dim]);
            }
        }
        pack_elapsed = pack_started.elapsed();
    }

    let qn_w = wname(layer_idx, "attn_qnorm.weight");
    let use_qknorm = cfg.qknorm_start >= 0
        && (layer_idx as i32) >= cfg.qknorm_start
        && weights.get_f32(&qn_w).is_some();
    let use_rope = cfg.rope_start >= 0 && (layer_idx as i32) >= cfg.rope_start;

    // RoPE positions: special tokens (CLS + registers) always get (0,0).
    // Patch token `idx` (row-major over the (gh,gw) grid) gets:
    //   - `global == false` ("local" set): (row+1, col+1), 1-indexed,
    //     reserving (0,0) for the special tokens.
    //   - `global == true` ("nodiff" set): (1,1) for EVERY patch — every
    //     position collapses to the same value, i.e. RoPE contributes no
    //     positional differentiation for global/cross-view attention layers.
    // Confirmed against `../src/dino_backbone.cpp`'s `pos_local`/`pos_nodiff`
    // construction. The caller (`Backbone::forward`) decides `global` per
    // layer (`cfg.alt_start>=0 && i>=cfg.alt_start && i%2==1`); this function
    // never inspects `cfg.alt_start` itself.
    let n_special = 1 + cfg.num_register as usize;
    let pos_yx: Vec<f32> = if use_rope {
        let mut p = vec![0f32; n * 2];
        let tokens_per_view = n_special + gh * gw;
        assert!(view_count > 0, "view_count must be non-zero");
        assert_eq!(
            n,
            tokens_per_view * view_count,
            "token count must equal tokens_per_view * view_count"
        );
        for view in 0..view_count {
            let base = view * tokens_per_view;
            for local_t in n_special..tokens_per_view {
                let t = base + local_t;
                if global {
                    p[2 * t] = 1.0;
                    p[2 * t + 1] = 1.0;
                } else {
                    let idx = local_t - n_special;
                    let row = idx / gw;
                    let col = idx % gw;
                    p[2 * t] = (row + 1) as f32;
                    p[2 * t + 1] = (col + 1) as f32;
                }
            }
        }
        p
    } else {
        Vec::new()
    };

    let position_started = std::time::Instant::now();
    let mut used_fused_qk_norm_rope = false;
    if use_qknorm {
        let q_gamma = weights
            .get_f32(&qn_w)
            .unwrap_or_else(|| panic!("Weights missing f32 entry {qn_w:?}"));
        let q_beta_name = wname(layer_idx, "attn_qnorm.bias");
        let q_beta = weights
            .get_f32(&q_beta_name)
            .unwrap_or_else(|| panic!("Weights missing f32 entry {q_beta_name:?}"));
        let k_gamma_name = wname(layer_idx, "attn_knorm.weight");
        let k_gamma = weights
            .get_f32(&k_gamma_name)
            .unwrap_or_else(|| panic!("Weights missing f32 entry {k_gamma_name:?}"));
        let k_beta_name = wname(layer_idx, "attn_knorm.bias");
        let k_beta = weights
            .get_f32(&k_beta_name)
            .unwrap_or_else(|| panic!("Weights missing f32 entry {k_beta_name:?}"));
        if use_rope {
            let positions: Vec<i64> = pos_yx.iter().map(|&value| value as i64).collect();
            used_fused_qk_norm_rope = vestra_kernels::qk_norm_rope_f32_da3_base(
                &mut q,
                &mut k,
                q_gamma,
                q_beta,
                k_gamma,
                k_beta,
                &positions,
                cfg.rope_freq,
                QK_NORM_EPS,
            );
        }
        if !used_fused_qk_norm_rope {
            vestra_kernels::scalar::layernorm(
                &mut q,
                heads * n,
                head_dim,
                q_gamma,
                q_beta,
                QK_NORM_EPS,
            );
            vestra_kernels::scalar::layernorm(
                &mut k,
                heads * n,
                head_dim,
                k_gamma,
                k_beta,
                QK_NORM_EPS,
            );
        }
    }
    if use_rope && !used_fused_qk_norm_rope {
        let positions: Vec<i64> = pos_yx.iter().map(|&value| value as i64).collect();
        vestra_kernels::rope2d(&mut q, heads, n, head_dim, &positions, cfg.rope_freq);
        vestra_kernels::rope2d(&mut k, heads, n, head_dim, &positions, cfg.rope_freq);
    }
    let position_elapsed = position_started.elapsed();
    let core_started = std::time::Instant::now();
    let mut attn_hnd = vec![0.0; heads * n * head_dim];
    vestra_kernels::attention(&q, &k, &v, heads, n, head_dim, &mut attn_hnd);
    let core_elapsed = core_started.elapsed();

    // Transpose head-major [heads, n, head_dim] back to token-major
    // [n, embed] before the output projection.
    let unpack_started = std::time::Instant::now();
    let mut attn_tok = vec![0f32; n * embed];
    for t in 0..n {
        for h in 0..heads {
            for d in 0..head_dim {
                attn_tok[t * embed + h * head_dim + d] = attn_hnd[(h * n + t) * head_dim + d];
            }
        }
    }
    let unpack_elapsed = unpack_started.elapsed();

    let projection_started = std::time::Instant::now();
    let output = run_linear(
        &attn_tok,
        n,
        embed,
        embed,
        &wname(layer_idx, "attn_proj.weight"),
        &wname(layer_idx, "attn_proj.bias"),
        false,
        Some(&wname(layer_idx, "ls1")),
        weights,
    );
    if phase_profile {
        eprintln!(
            "phase: attention[{layer_idx}] qkv={:.3}ms pack={:.3}ms qk_norm_rope={:.3}ms core={:.3}ms unpack={:.3}ms proj={:.3}ms",
            qkv_elapsed.as_secs_f64() * 1e3,
            pack_elapsed.as_secs_f64() * 1e3,
            position_elapsed.as_secs_f64() * 1e3,
            core_elapsed.as_secs_f64() * 1e3,
            unpack_elapsed.as_secs_f64() * 1e3,
            projection_started.elapsed().as_secs_f64() * 1e3,
        );
    }
    output
}

/// Runs one DINOv2/DA3 ViT transformer block over `tokens` in place:
/// `LN1 -> Attention(+qk-norm, +RoPE) -> [ls1] -> residual -> LN2 -> MLP(GELU)
/// -> [ls2] -> residual`. See the module doc comment for weight tensor
/// names and the linear-weight orientation convention.
///
/// `n` is the token count (`1 + num_register + gh*gw`); `gh`/`gw` (the
/// patch-grid resolution) are only consulted when this layer uses RoPE
/// (`layer_idx >= cfg.rope_start`).
///
/// `global` selects which of the two RoPE position sets this layer's
/// attention uses when RoPE is active (see `run_attention`'s doc comment):
/// `false` = "local" (per-patch positions), `true` = "nodiff" (every patch
/// treated as position `(1,1)`). The CALLER decides `global` (typically
/// `cfg.alt_start>=0 && layer_idx>=cfg.alt_start && layer_idx%2==1`,
/// matching `../src/dino_backbone.cpp`'s alternation) — this function does
/// not consult `cfg.alt_start` itself, keeping the block math oblivious to
/// which layer index it's running at.
///
/// # Panics
/// - If `tokens.len() != n * cfg.embed_dim`.
/// - If `cfg.ffn_type == "swiglu"`: a deliberate, honest "not yet
///   supported" hard error (see the module/crate-level docs) rather than
///   silently running the wrong FFN math. Only `"mlp"` (DA3-BASE) is
///   implemented by this function.
#[allow(clippy::too_many_arguments)]
pub(crate) fn vit_block_with_views(
    tokens: &mut [f32],
    n: usize,
    gh: usize,
    gw: usize,
    global: bool,
    view_count: usize,
    cfg: &ModelConfig,
    layer_idx: usize,
    weights: &Weights,
    _backend: &dyn Backend,
    residual_executor: Option<&dyn ResidualAddExecutor>,
) {
    let phase_profile = std::env::var_os("DA_PHASE_PROFILE").is_some();
    assert_eq!(
        tokens.len(),
        n * cfg.embed_dim as usize,
        "tokens length must be n * embed_dim"
    );
    if cfg.ffn_type == "swiglu" {
        unimplemented!(
            "vit_block: ffn_type=\"swiglu\" (giant DA3 models, mlp_w12/mlp_w3) is not \
             implemented — only the classic MLP (fc1/fc2) path used by DA3-BASE is \
             supported (Task 17 explicitly defers SwiGLU; see module doc comment)"
        );
    }

    let embed = cfg.embed_dim as usize;
    let mlp_hidden = cfg.mlp_hidden as usize;
    let eps = cfg.ln_eps;

    // --- Attention sub-block ---
    // `tokens` is only read by `run_layernorm`/`run_attention` (both
    // operate on fresh Vec<f32> copies via their mini-graphs' own arenas),
    // so it still holds the pre-attention residual right up until the
    // in-place `vestra_kernels::scalar::add` below.
    let ln1_started = std::time::Instant::now();
    let ln1 = run_layernorm(
        tokens,
        n,
        embed,
        &wname(layer_idx, "norm1.weight"),
        &wname(layer_idx, "norm1.bias"),
        eps,
        weights,
    );
    let ln1_elapsed = ln1_started.elapsed();
    let attention_started = std::time::Instant::now();
    let attn_out = run_attention(&ln1, n, gh, gw, global, view_count, cfg, layer_idx, weights);
    add_residual(tokens, &attn_out, residual_executor);
    let attention_elapsed = attention_started.elapsed();

    // --- MLP sub-block --- (same "tokens still holds the residual" trick)
    let ln2_started = std::time::Instant::now();
    let ln2 = run_layernorm(
        tokens,
        n,
        embed,
        &wname(layer_idx, "norm2.weight"),
        &wname(layer_idx, "norm2.bias"),
        eps,
        weights,
    );
    let ln2_elapsed = ln2_started.elapsed();
    let fc1_started = std::time::Instant::now();
    let h = run_linear(
        &ln2,
        n,
        embed,
        mlp_hidden,
        &wname(layer_idx, "mlp_fc1.weight"),
        &wname(layer_idx, "mlp_fc1.bias"),
        true,
        None,
        weights,
    );
    let fc1_elapsed = fc1_started.elapsed();
    let fc2_started = std::time::Instant::now();
    let m = run_linear(
        &h,
        n,
        mlp_hidden,
        embed,
        &wname(layer_idx, "mlp_fc2.weight"),
        &wname(layer_idx, "mlp_fc2.bias"),
        false,
        Some(&wname(layer_idx, "ls2")),
        weights,
    );
    add_residual(tokens, &m, residual_executor);
    if phase_profile {
        eprintln!(
            "phase: block[{layer_idx}] ln1={:.3}ms attention={:.3}ms ln2={:.3}ms fc1_gelu={:.3}ms fc2_residual={:.3}ms",
            ln1_elapsed.as_secs_f64() * 1e3,
            attention_elapsed.as_secs_f64() * 1e3,
            ln2_elapsed.as_secs_f64() * 1e3,
            fc1_elapsed.as_secs_f64() * 1e3,
            fc2_started.elapsed().as_secs_f64() * 1e3,
        );
    }
}

/// Runs a transformer block for one view.
///
/// Multi-view global attention uses the internal [`vit_block_with_views`]
/// entry point so RoPE special-token boundaries repeat for every view.
#[allow(clippy::too_many_arguments)]
pub(crate) fn vit_block_with_residual(
    tokens: &mut [f32],
    n: usize,
    gh: usize,
    gw: usize,
    global: bool,
    cfg: &ModelConfig,
    layer_idx: usize,
    weights: &Weights,
    backend: &dyn Backend,
    residual_executor: Option<&dyn ResidualAddExecutor>,
) {
    vit_block_with_views(
        tokens,
        n,
        gh,
        gw,
        global,
        1,
        cfg,
        layer_idx,
        weights,
        backend,
        residual_executor,
    );
}

pub fn vit_block(
    tokens: &mut [f32],
    n: usize,
    gh: usize,
    gw: usize,
    global: bool,
    cfg: &ModelConfig,
    layer_idx: usize,
    weights: &Weights,
    backend: &dyn Backend,
) {
    vit_block_with_residual(
        tokens, n, gh, gw, global, cfg, layer_idx, weights, backend, None,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use da_graph::{CpuBackend, Graph};

    #[test]
    fn direct_layernorm_and_linear_match_the_retired_graph_path() {
        let backend = CpuBackend::new();
        let mut rng = Xorshift32(0x51A7_C0DE);
        let input = random_vec(&mut rng, 3 * 4);
        let mut weights = Weights::new();
        weights.insert_f32("norm.g", random_vec(&mut rng, 4));
        weights.insert_f32("norm.b", random_vec(&mut rng, 4));
        weights.insert_f32("linear.w", random_vec(&mut rng, 4 * 5));
        weights.insert_f32("linear.b", random_vec(&mut rng, 5));
        weights.insert_f32("linear.ls", random_vec(&mut rng, 5));

        let direct_norm = run_layernorm(&input, 3, 4, "norm.g", "norm.b", 1e-6, &weights);
        let mut norm_graph = Graph::builder();
        let norm_x = norm_graph.input(3 * 4);
        let norm_g = norm_graph.weight("norm.g", 4);
        let norm_b = norm_graph.weight("norm.b", 4);
        norm_graph.layer_norm(norm_x, norm_g, norm_b, 3, 4, 1e-6);
        norm_graph.output(norm_x);
        let graph_norm = norm_graph
            .build()
            .compile()
            .run(&backend, &[&input], &weights)
            .remove(0);

        let direct_linear = run_linear(
            &input,
            3,
            4,
            5,
            "linear.w",
            "linear.b",
            true,
            Some("linear.ls"),
            &weights,
        );
        let mut linear_graph = Graph::builder();
        let linear_x = linear_graph.input(3 * 4);
        let linear_w = linear_graph.weight("linear.w", 4 * 5);
        let linear_b = linear_graph.weight("linear.b", 5);
        let linear_out = linear_graph.gemm(linear_x, linear_w, 3, 5, 4);
        linear_graph.add_bias(linear_out, linear_b, 3, 5);
        linear_graph.gelu(linear_out);
        let linear_ls = linear_graph.weight("linear.ls", 5);
        linear_graph.layer_scale(linear_out, linear_ls, 3, 5);
        linear_graph.output(linear_out);
        let graph_linear = linear_graph
            .build()
            .compile()
            .run(&backend, &[&input], &weights)
            .remove(0);

        for (direct, graph) in direct_norm.iter().zip(graph_norm.iter()) {
            assert_eq!(direct.to_bits(), graph.to_bits());
        }
        for (direct, graph) in direct_linear.iter().zip(graph_linear.iter()) {
            assert_eq!(direct.to_bits(), graph.to_bits());
        }
    }

    fn test_cfg(embed: u32, heads: u32, head_dim: u32, mlp_hidden: u32) -> ModelConfig {
        ModelConfig {
            arch: "depthanything3".to_string(),
            patch_size: 14,
            image_size: 28,
            embed_dim: embed,
            depth: 1,
            num_heads: heads,
            head_dim,
            mlp_hidden,
            num_register: 0,
            rope_start: -1,
            qknorm_start: -1,
            rope_freq: 100.0,
            ln_eps: 1e-6,
            out_layers: vec![0],
            ffn_type: "mlp".to_string(),
            head_features: 1,
            head_max_depth: 1.0,
            img_mean: [0.0, 0.0, 0.0],
            img_std: [1.0, 1.0, 1.0],
            img_resize_mode: "bilinear".to_string(),
            alt_start: -1,
            cat_token: true,
            cam_dim_in: 1,
            head_pos_embed: true,
        }
    }

    /// A deterministic PRNG-filled weight set covering every tensor
    /// `vit_block` needs for one layer, sized to `cfg`. `with_ls`/`with_qkn`
    /// control whether ls1/ls2 and qnorm/knorm tensors are inserted
    /// (presence-gating, trap #3 / traps #1-2).
    fn synthetic_weights(
        cfg: &ModelConfig,
        layer_idx: usize,
        with_ls: bool,
        with_qkn: bool,
    ) -> Weights {
        let embed = cfg.embed_dim as usize;
        let head_dim = cfg.head_dim as usize;
        let mlp_hidden = cfg.mlp_hidden as usize;
        let mut rng = Xorshift32(0xB16B_00B5 ^ (layer_idx as u32));
        let mut w = Weights::new();
        let mut put = |name: String, len: usize, w: &mut Weights| {
            w.insert_f32(name, random_vec(&mut rng, len));
        };
        put(wname(layer_idx, "norm1.weight"), embed, &mut w);
        put(wname(layer_idx, "norm1.bias"), embed, &mut w);
        put(wname(layer_idx, "norm2.weight"), embed, &mut w);
        put(wname(layer_idx, "norm2.bias"), embed, &mut w);
        put(
            wname(layer_idx, "attn_qkv.weight"),
            embed * 3 * embed,
            &mut w,
        );
        put(wname(layer_idx, "attn_qkv.bias"), 3 * embed, &mut w);
        put(wname(layer_idx, "attn_proj.weight"), embed * embed, &mut w);
        put(wname(layer_idx, "attn_proj.bias"), embed, &mut w);
        put(
            wname(layer_idx, "mlp_fc1.weight"),
            embed * mlp_hidden,
            &mut w,
        );
        put(wname(layer_idx, "mlp_fc1.bias"), mlp_hidden, &mut w);
        put(
            wname(layer_idx, "mlp_fc2.weight"),
            mlp_hidden * embed,
            &mut w,
        );
        put(wname(layer_idx, "mlp_fc2.bias"), embed, &mut w);
        if with_ls {
            put(wname(layer_idx, "ls1"), embed, &mut w);
            put(wname(layer_idx, "ls2"), embed, &mut w);
        }
        if with_qkn {
            put(wname(layer_idx, "attn_qnorm.weight"), head_dim, &mut w);
            put(wname(layer_idx, "attn_qnorm.bias"), head_dim, &mut w);
            put(wname(layer_idx, "attn_knorm.weight"), head_dim, &mut w);
            put(wname(layer_idx, "attn_knorm.bias"), head_dim, &mut w);
        }
        w
    }

    #[test]
    fn vit_block_preserves_token_shape_and_changes_values() {
        let cfg = test_cfg(8, 2, 4, 16);
        let weights = synthetic_weights(&cfg, 0, true, false);
        let backend = CpuBackend::new();
        let n = 5usize; // 1 CLS + 2x2 patch grid
        let mut rng = Xorshift32(0xC0FF_EE00);
        let mut tokens = random_vec(&mut rng, n * cfg.embed_dim as usize);
        let before = tokens.clone();

        vit_block(&mut tokens, n, 2, 2, false, &cfg, 0, &weights, &backend);

        assert_eq!(tokens.len(), before.len());
        assert_ne!(
            tokens, before,
            "a real forward pass should change token values"
        );
        assert!(
            tokens.iter().all(|v| v.is_finite()),
            "output must not contain NaN/Inf"
        );
    }

    #[test]
    fn vit_block_layerscale_presence_gating_changes_output() {
        // Same random tokens/base weights, only ls1/ls2 presence differs —
        // output must differ (trap #3: LayerScale must actually be applied
        // when present, and its absence must be a real code path too).
        let cfg = test_cfg(8, 2, 4, 16);
        let backend = CpuBackend::new();
        let n = 5usize;
        let mut rng = Xorshift32(0xFACE_FEED);
        let tokens0 = random_vec(&mut rng, n * cfg.embed_dim as usize);

        let w_with_ls = synthetic_weights(&cfg, 0, true, false);
        let w_without_ls = synthetic_weights(&cfg, 0, false, false);
        // synthetic_weights re-seeds its own RNG per call (keyed only by
        // layer_idx), so the shared (non-ls) tensors are byte-identical
        // between the two calls — only ls1/ls2 presence differs.

        let mut t_with = tokens0.clone();
        vit_block(&mut t_with, n, 2, 2, false, &cfg, 0, &w_with_ls, &backend);
        let mut t_without = tokens0.clone();
        vit_block(
            &mut t_without,
            n,
            2,
            2,
            false,
            &cfg,
            0,
            &w_without_ls,
            &backend,
        );

        assert_ne!(t_with, t_without);
    }

    #[test]
    fn vit_block_qknorm_and_rope_gated_by_layer_idx() {
        // qknorm_start=1, rope_start=1: layer 0 must NOT apply either
        // (even though the qn/kn weight tensors are present), layer 1 must.
        let mut cfg = test_cfg(8, 2, 4, 16);
        cfg.qknorm_start = 1;
        cfg.rope_start = 1;
        let backend = CpuBackend::new();
        let n = 5usize;
        let mut rng = Xorshift32(0x5EED_1234);
        let tokens0 = random_vec(&mut rng, n * cfg.embed_dim as usize);

        let weights = synthetic_weights(&cfg, 0, true, true);
        let weights1 = synthetic_weights(&cfg, 1, true, true);

        let mut t_layer0 = tokens0.clone();
        vit_block(&mut t_layer0, n, 2, 2, false, &cfg, 0, &weights, &backend);

        // Manually force qknorm_start/rope_start to "always on" and rerun
        // layer 0 with the SAME weights/tokens: since layer 0 < 1, the
        // gated run above should differ from an "always on" run using
        // identical tensors — proving the gate actually suppresses the
        // qk-norm/RoPE application at layer 0, not just that they're wired
        // at all (that's covered by da-graph's own op-level tests).
        let mut cfg_always_on = cfg.clone();
        cfg_always_on.qknorm_start = 0;
        cfg_always_on.rope_start = 0;
        let mut t_layer0_forced = tokens0.clone();
        vit_block(
            &mut t_layer0_forced,
            n,
            2,
            2,
            false,
            &cfg_always_on,
            0,
            &weights,
            &backend,
        );

        assert_ne!(
            t_layer0, t_layer0_forced,
            "layer_idx < qknorm_start/rope_start must suppress qk-norm/RoPE"
        );

        // And layer 1 (>= qknorm_start/rope_start) really does run with
        // gating active (sanity: it must simply produce finite output).
        let mut t_layer1 = tokens0.clone();
        vit_block(&mut t_layer1, n, 2, 2, false, &cfg, 1, &weights1, &backend);
        assert!(t_layer1.iter().all(|v| v.is_finite()));
    }

    #[test]
    #[should_panic(expected = "swiglu")]
    fn vit_block_hard_errors_on_swiglu() {
        let mut cfg = test_cfg(8, 2, 4, 16);
        cfg.ffn_type = "swiglu".to_string();
        let weights = synthetic_weights(&cfg, 0, true, false);
        let backend = CpuBackend::new();
        let n = 5usize;
        let mut tokens = vec![0f32; n * cfg.embed_dim as usize];
        vit_block(&mut tokens, n, 2, 2, false, &cfg, 0, &weights, &backend);
    }

    /// Deterministic, dependency-free PRNG (Xorshift32) for reproducible
    /// synthetic test data (matching the convention already used in
    /// `vestra-kernels/src/conv.rs`'s tests).
    struct Xorshift32(u32);
    impl Xorshift32 {
        fn next_f32(&mut self) -> f32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            self.0 = x;
            ((x as f32) / (u32::MAX as f32)) * 2.0 - 1.0
        }
    }
    fn random_vec(rng: &mut Xorshift32, n: usize) -> Vec<f32> {
        (0..n).map(|_| rng.next_f32()).collect()
    }
}
