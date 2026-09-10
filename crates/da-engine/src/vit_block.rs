//! A single DINOv2/DA3 ViT transformer block (`vit_block`) and its weight
//! tensor naming convention.
//!
//! ## Third-party provenance
//!
//! The block topology, optional Q/K normalization, RoPE boundary, tensor
//! naming, and local/global attention semantics are structure-preserving Rust
//! ports of [`src/vit_block.cpp`], [`src/attention.cpp`], and
//! [`src/dino_backbone.cpp`] at pinned `depth-anything.cpp` revision
//! `2028b47ac75a8659c6a9aa617baf09be193eb55f` (MIT). Vestra supplies different
//! graph ownership, dispatch, and CPU/CUDA kernels. See the repository-root
//! `THIRD_PARTY_NOTICES.md`.
//!
//! [`src/vit_block.cpp`]: https://github.com/localai-org/depth-anything.cpp/blob/2028b47ac75a8659c6a9aa617baf09be193eb55f/src/vit_block.cpp
//! [`src/attention.cpp`]: https://github.com/localai-org/depth-anything.cpp/blob/2028b47ac75a8659c6a9aa617baf09be193eb55f/src/attention.cpp
//! [`src/dino_backbone.cpp`]: https://github.com/localai-org/depth-anything.cpp/blob/2028b47ac75a8659c6a9aa617baf09be193eb55f/src/dino_backbone.cpp
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
use rayon::prelude::*;
use vestra_kernels::gemm::{Da3ProjectionGemm, Gemm};
use vestra_kernels::packed_gemm::PreparedLinearF32;

/// Execution seam for the two transformer residual additions. The normal
/// production path supplies `None` and keeps the optimized CPU add. A CUDA
/// implementation is intentionally an opt-in parity slice until all adjacent
/// operators can remain device-resident.
pub trait ResidualAddExecutor: Send + Sync {
    fn add_in_place(&self, destination: &mut [f32], source: &[f32]);
}

/// Execution seam for a complete DA3 MLP branch. Implementations own the
/// LayerNorm/FC1/FC2 parameter lifetime and return the token-major FC2
/// result; the residual addition remains an explicitly separate boundary.
pub trait MlpExecutor {
    fn run_mlp(&self, layer_idx: usize, input: &[f32], rows: usize) -> Vec<f32>;
}

/// Experimental CPU MLP executor backed by model-load-time panel-packed F32
/// weights. It is selected only through `DA3_PACKED_MLP=1`; normal inference
/// remains on the qualified BLIS route until this candidate wins end-to-end.
pub struct PackedMlpExecutor {
    layers: Vec<PackedMlpLayer>,
    ln_eps: f32,
}

struct PackedMlpLayer {
    norm_gamma: Vec<f32>,
    norm_beta: Vec<f32>,
    fc1: PreparedLinearF32,
    fc1_bias: Vec<f32>,
    fc2: PreparedLinearF32,
    fc2_bias: Vec<f32>,
    ls2: Option<Vec<f32>>,
}

impl PackedMlpExecutor {
    pub fn new(cfg: &ModelConfig, weights: &Weights) -> Option<Self> {
        if cfg.embed_dim != 768 || cfg.mlp_hidden != 3072 || cfg.ffn_type != "mlp" {
            return None;
        }
        let mut layers = Vec::with_capacity(cfg.depth as usize);
        for layer_idx in 0..cfg.depth as usize {
            let get = |suffix: &str| {
                weights
                    .get_f32(&wname(layer_idx, suffix))
                    .map(ToOwned::to_owned)
            };
            layers.push(PackedMlpLayer {
                norm_gamma: get("norm2.weight")?,
                norm_beta: get("norm2.bias")?,
                fc1: PreparedLinearF32::try_new(get("mlp_fc1.weight")?.as_slice(), 768, 3072)?,
                fc1_bias: get("mlp_fc1.bias")?,
                fc2: PreparedLinearF32::try_new(get("mlp_fc2.weight")?.as_slice(), 3072, 768)?,
                fc2_bias: get("mlp_fc2.bias")?,
                ls2: get("ls2"),
            });
        }
        Some(Self {
            layers,
            ln_eps: cfg.ln_eps,
        })
    }
}

/// Execution seam for the complete QKV → Q/K normalization/RoPE → attention
/// → output-projection branch. Implementations return `None` when the model
/// shape or operator configuration is not their qualified CUDA subset.
pub trait AttentionExecutor {
    fn run_attention(
        &self,
        layer_idx: usize,
        input: &[f32],
        rows: usize,
        heads: usize,
        head_dim: usize,
        positions_yx: Option<&[f32]>,
    ) -> Option<Vec<f32>>;
}

/// A device-resident transformer tail beginning after CPU-order LN1. The
/// executor owns attention, both residual additions, LN2, and the complete
/// MLP branch. `None` means the block is outside its validated subset.
pub trait TransformerTailExecutor {
    #[cfg(feature = "cuda-residual-oracle")]
    fn persistent_cuda_tail(&self) -> Option<&CudaTransformerTailExecutor> {
        None
    }

    #[allow(clippy::too_many_arguments)]
    fn run_tail(
        &self,
        layer_idx: usize,
        tokens: &[f32],
        ln1: &[f32],
        rows: usize,
        gh: usize,
        gw: usize,
        global: bool,
        view_count: usize,
        cfg: &ModelConfig,
    ) -> Option<Vec<f32>>;
}

/// Cached device MLP parameters for every classic DA3-BASE transformer
/// block. It is a parity-only vertical slice until layer normalization,
/// attention, and both residual branches can also remain device-resident.
#[cfg(feature = "cuda-residual-oracle")]
pub struct CudaMlpExecutor {
    runtime: vestra_kernels::cuda::CudaRuntime,
    ln_eps: f32,
    layers: Vec<CudaMlpLayer>,
}

/// Cached device parameters for the DA3-BASE attention branch. This keeps
/// QKV, Q/K preparation, online attention, unpack, and output projection on
/// CUDA; only LN1 input and the projected residual branch cross the host
/// boundary while the rest of the transformer remains CPU-owned.
#[cfg(feature = "cuda-residual-oracle")]
pub struct CudaAttentionExecutor {
    runtime: vestra_kernels::cuda::CudaRuntime,
    layers: Vec<CudaAttentionLayer>,
}

/// Combines the qualified CUDA attention and MLP branches so no attention
/// result, residual, or FC1 activation crosses PCIe between them.
#[cfg(feature = "cuda-residual-oracle")]
pub struct CudaTransformerTailExecutor {
    attention: CudaAttentionExecutor,
    mlp: CudaMlpExecutor,
    camera_reference: vestra_kernels::cuda::CudaTensorF32,
    camera_source: vestra_kernels::cuda::CudaTensorF32,
}

#[cfg(feature = "cuda-residual-oracle")]
struct CudaAttentionLayer {
    norm1_gamma: vestra_kernels::cuda::CudaTensorF32,
    norm1_beta: vestra_kernels::cuda::CudaTensorF32,
    qkv: vestra_kernels::cuda::CudaLinearF32,
    qk: Option<CudaQkParameters>,
    projection: vestra_kernels::cuda::CudaLinearF32,
}

#[cfg(feature = "cuda-residual-oracle")]
struct CudaQkParameters {
    q_gamma: vestra_kernels::cuda::CudaTensorF32,
    q_beta: vestra_kernels::cuda::CudaTensorF32,
    k_gamma: vestra_kernels::cuda::CudaTensorF32,
    k_beta: vestra_kernels::cuda::CudaTensorF32,
}

#[cfg(feature = "cuda-residual-oracle")]
struct CudaMlpLayer {
    norm2_gamma: vestra_kernels::cuda::CudaTensorF32,
    norm2_beta: vestra_kernels::cuda::CudaTensorF32,
    fc1: vestra_kernels::cuda::CudaLinearF32,
    fc2: vestra_kernels::cuda::CudaLinearF32,
}

#[cfg(feature = "cuda-residual-oracle")]
impl CudaMlpExecutor {
    pub fn new(
        runtime: vestra_kernels::cuda::CudaRuntime,
        cfg: &ModelConfig,
        weights: &Weights,
    ) -> Result<Self, vestra_kernels::cuda::CudaError> {
        assert_eq!(
            cfg.ffn_type, "mlp",
            "CUDA MLP supports classic DA3 MLP only"
        );
        let embed = cfg.embed_dim as usize;
        let hidden = cfg.mlp_hidden as usize;
        let mut layers = Vec::with_capacity(cfg.depth as usize);
        for layer_idx in 0..cfg.depth as usize {
            let norm2_gamma =
                runtime.upload_f32(weights.get_f32(&wname(layer_idx, "norm2.weight")).unwrap())?;
            let norm2_beta =
                runtime.upload_f32(weights.get_f32(&wname(layer_idx, "norm2.bias")).unwrap())?;
            let fc1 = runtime.prepare_linear_f32(
                embed,
                hidden,
                weights
                    .get_f32(&wname(layer_idx, "mlp_fc1.weight"))
                    .unwrap(),
                weights.get_f32(&wname(layer_idx, "mlp_fc1.bias")).unwrap(),
                None,
            )?;
            let fc2 = runtime.prepare_linear_f32(
                hidden,
                embed,
                weights
                    .get_f32(&wname(layer_idx, "mlp_fc2.weight"))
                    .unwrap(),
                weights.get_f32(&wname(layer_idx, "mlp_fc2.bias")).unwrap(),
                weights.get_f32(&wname(layer_idx, "ls2")),
            )?;
            layers.push(CudaMlpLayer {
                norm2_gamma,
                norm2_beta,
                fc1,
                fc2,
            });
        }
        Ok(Self {
            runtime,
            ln_eps: cfg.ln_eps,
            layers,
        })
    }
}

#[cfg(feature = "cuda-residual-oracle")]
impl CudaAttentionExecutor {
    pub fn new(
        runtime: vestra_kernels::cuda::CudaRuntime,
        cfg: &ModelConfig,
        weights: &Weights,
    ) -> Result<Self, vestra_kernels::cuda::CudaError> {
        assert_eq!(
            cfg.embed_dim, 768,
            "CUDA attention supports DA3-BASE embed=768 only"
        );
        assert_eq!(
            cfg.num_heads, 12,
            "CUDA attention supports DA3-BASE heads=12 only"
        );
        assert_eq!(
            cfg.head_dim, 64,
            "CUDA attention supports DA3-BASE head_dim=64 only"
        );
        assert!(
            cfg.qknorm_start >= 0 && cfg.rope_start >= 0 && cfg.qknorm_start == cfg.rope_start,
            "CUDA attention requires paired, enabled DA3 Q/K norm and RoPE starts"
        );
        let embed = cfg.embed_dim as usize;
        let mut layers = Vec::with_capacity(cfg.depth as usize);
        for layer_idx in 0..cfg.depth as usize {
            let q_gamma_name = wname(layer_idx, "attn_qnorm.weight");
            let qk = weights
                .get_f32(&q_gamma_name)
                .map(|q_gamma| {
                    Ok(CudaQkParameters {
                        q_gamma: runtime.upload_f32(q_gamma)?,
                        q_beta: runtime.upload_f32(
                            weights
                                .get_f32(&wname(layer_idx, "attn_qnorm.bias"))
                                .unwrap(),
                        )?,
                        k_gamma: runtime.upload_f32(
                            weights
                                .get_f32(&wname(layer_idx, "attn_knorm.weight"))
                                .unwrap(),
                        )?,
                        k_beta: runtime.upload_f32(
                            weights
                                .get_f32(&wname(layer_idx, "attn_knorm.bias"))
                                .unwrap(),
                        )?,
                    })
                })
                .transpose()?;
            layers.push(CudaAttentionLayer {
                norm1_gamma: runtime
                    .upload_f32(weights.get_f32(&wname(layer_idx, "norm1.weight")).unwrap())?,
                norm1_beta: runtime
                    .upload_f32(weights.get_f32(&wname(layer_idx, "norm1.bias")).unwrap())?,
                qkv: runtime.prepare_linear_f32(
                    embed,
                    3 * embed,
                    weights
                        .get_f32(&wname(layer_idx, "attn_qkv.weight"))
                        .unwrap(),
                    weights.get_f32(&wname(layer_idx, "attn_qkv.bias")).unwrap(),
                    None,
                )?,
                qk,
                projection: runtime.prepare_linear_f32(
                    embed,
                    embed,
                    weights
                        .get_f32(&wname(layer_idx, "attn_proj.weight"))
                        .unwrap(),
                    weights
                        .get_f32(&wname(layer_idx, "attn_proj.bias"))
                        .unwrap(),
                    weights.get_f32(&wname(layer_idx, "ls1")),
                )?,
            });
        }
        Ok(Self { runtime, layers })
    }
}

#[cfg(feature = "cuda-residual-oracle")]
impl AttentionExecutor for CudaAttentionExecutor {
    fn run_attention(
        &self,
        layer_idx: usize,
        input: &[f32],
        rows: usize,
        heads: usize,
        head_dim: usize,
        positions_yx: Option<&[f32]>,
    ) -> Option<Vec<f32>> {
        if heads != 12 || head_dim != 64 || positions_yx.is_none() || input.len() != rows * 768 {
            return None;
        }
        let layer = self.layers.get(layer_idx)?;
        let input = self.runtime.upload_f32(input).ok()?;
        let qkv = layer.qkv.run(&input, rows).ok()?;
        let (mut q, mut k, v) = self
            .runtime
            .split_qkv_hnd_f32(&qkv, rows, heads, head_dim)
            .ok()?;
        if let Some(qk) = layer.qk.as_ref() {
            let positions = self.runtime.upload_f32(positions_yx?).ok()?;
            self.runtime
                .qk_norm_rope_f32_da3_base(
                    &mut q,
                    &mut k,
                    &qk.q_gamma,
                    &qk.q_beta,
                    &qk.k_gamma,
                    &qk.k_beta,
                    &positions,
                    heads,
                    rows,
                    100.0,
                    QK_NORM_EPS,
                )
                .ok()?;
        } else {
            return None;
        }
        let hnd = self
            .runtime
            .attention_online_f32(&q, &k, &v, heads, rows, head_dim)
            .ok()?;
        let token_major = self
            .runtime
            .hnd_to_token_f32(&hnd, rows, heads, head_dim)
            .ok()?;
        let output = layer.projection.run(&token_major, rows).ok()?;
        self.runtime.download_f32(&output).ok()
    }
}

#[cfg(feature = "cuda-residual-oracle")]
impl CudaTransformerTailExecutor {
    pub fn new(
        runtime: vestra_kernels::cuda::CudaRuntime,
        cfg: &ModelConfig,
        weights: &Weights,
    ) -> Result<Self, vestra_kernels::cuda::CudaError> {
        let camera = weights
            .get_f32("vit.camera_token")
            .expect("CUDA persistent tail requires vit.camera_token");
        let embed = cfg.embed_dim as usize;
        let camera_reference = runtime.upload_f32(&camera[..embed])?;
        let camera_source = runtime.upload_f32(if camera.len() >= 2 * embed {
            &camera[embed..2 * embed]
        } else {
            &camera[..embed]
        })?;
        Ok(Self {
            attention: CudaAttentionExecutor::new(runtime.clone(), cfg, weights)?,
            mlp: CudaMlpExecutor::new(runtime, cfg, weights)?,
            camera_reference,
            camera_source,
        })
    }

    /// Uploads a token-major activation for the persistent single-view
    /// backbone route. Kept crate-visible so CUDA types remain below Engine's
    /// public API boundary.
    pub(crate) fn upload_tokens(
        &self,
        tokens: &[f32],
    ) -> Option<vestra_kernels::cuda::CudaTensorF32> {
        self.attention.runtime.upload_f32(tokens).ok()
    }

    pub(crate) fn download_tokens(
        &self,
        tokens: &vestra_kernels::cuda::CudaTensorF32,
    ) -> Option<Vec<f32>> {
        self.attention.runtime.download_f32(tokens).ok()
    }

    pub(crate) fn copy_tokens(
        &self,
        tokens: &vestra_kernels::cuda::CudaTensorF32,
    ) -> Option<vestra_kernels::cuda::CudaTensorF32> {
        self.attention.runtime.copy_f32(tokens).ok()
    }

    pub(crate) fn copy_token_segment(
        &self,
        tokens: &vestra_kernels::cuda::CudaTensorF32,
        source_offset: usize,
        len: usize,
    ) -> Option<vestra_kernels::cuda::CudaTensorF32> {
        self.attention
            .runtime
            .copy_segment_f32(tokens, source_offset, len)
            .ok()
    }

    pub(crate) fn copy_token_segment_into(
        &self,
        destination: &mut vestra_kernels::cuda::CudaTensorF32,
        destination_offset: usize,
        source: &vestra_kernels::cuda::CudaTensorF32,
        source_offset: usize,
        len: usize,
    ) -> Option<()> {
        self.attention
            .runtime
            .copy_segment_into_f32(destination, destination_offset, source, source_offset, len)
            .ok()
    }

    /// Applies DA3's reference-camera token injection without a host
    /// activation handoff. Multi-view source-token injection is deferred with
    /// the multi-view persistent scheduler.
    pub(crate) fn inject_reference_camera_token(
        &self,
        tokens: &mut vestra_kernels::cuda::CudaTensorF32,
        embed: usize,
    ) -> Option<()> {
        self.attention
            .runtime
            .overwrite_prefix_f32(tokens, &self.camera_reference, embed)
            .ok()
    }

    pub(crate) fn inject_camera_token_for_view(
        &self,
        tokens: &mut vestra_kernels::cuda::CudaTensorF32,
        view_index: usize,
        embed: usize,
    ) -> Option<()> {
        let source = if view_index == 0 {
            &self.camera_reference
        } else {
            &self.camera_source
        };
        self.attention
            .runtime
            .overwrite_prefix_f32(tokens, source, embed)
            .ok()
    }

    /// Runs one qualified DA3-BASE transformer block entirely on device.
    ///
    /// The caller transfers ownership of the token buffer and receives it
    /// back after both residual branches.  This is deliberately separate
    /// from the host-facing [`TransformerTailExecutor`] adapter below: a
    /// persistent backbone can chain this method across layers without an
    /// upload/download at each block, while the existing oracle retains its
    /// narrow, independently testable boundary.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_tail_device(
        &self,
        layer_idx: usize,
        tokens: vestra_kernels::cuda::CudaTensorF32,
        rows: usize,
        gh: usize,
        gw: usize,
        global: bool,
        view_count: usize,
        cfg: &ModelConfig,
    ) -> Option<vestra_kernels::cuda::CudaTensorF32> {
        let heads = cfg.num_heads as usize;
        let head_dim = cfg.head_dim as usize;
        if heads != 12 || head_dim != 64 || tokens.len() != rows * 768 {
            return None;
        }
        let layer = self.attention.layers.get(layer_idx)?;
        let qk = layer.qk.as_ref()?;
        let n_special = 1 + cfg.num_register as usize;
        let tokens_per_view = n_special + gh * gw;
        if rows != tokens_per_view * view_count {
            return None;
        }
        let mut positions_yx = vec![0.0_f32; rows * 2];
        for view in 0..view_count {
            let base = view * tokens_per_view;
            for local_t in n_special..tokens_per_view {
                let token = base + local_t;
                if global {
                    positions_yx[2 * token] = 1.0;
                    positions_yx[2 * token + 1] = 1.0;
                } else {
                    let patch = local_t - n_special;
                    positions_yx[2 * token] = (patch / gw + 1) as f32;
                    positions_yx[2 * token + 1] = (patch % gw + 1) as f32;
                }
            }
        }
        let runtime = &self.attention.runtime;
        let ln1 = runtime
            .layernorm_f32_cpu_order(
                &tokens,
                &layer.norm1_gamma,
                &layer.norm1_beta,
                rows,
                768,
                self.mlp.ln_eps,
            )
            .ok()?;
        let qkv = layer.qkv.run(&ln1, rows).ok()?;
        let (mut q, mut k, v) = runtime
            .split_qkv_hnd_f32(&qkv, rows, heads, head_dim)
            .ok()?;
        let positions = runtime.upload_f32(&positions_yx).ok()?;
        runtime
            .qk_norm_rope_f32_da3_base(
                &mut q,
                &mut k,
                &qk.q_gamma,
                &qk.q_beta,
                &qk.k_gamma,
                &qk.k_beta,
                &positions,
                heads,
                rows,
                cfg.rope_freq,
                QK_NORM_EPS,
            )
            .ok()?;
        let hnd = runtime
            .attention_online_f32(&q, &k, &v, heads, rows, head_dim)
            .ok()?;
        let token_major = runtime.hnd_to_token_f32(&hnd, rows, heads, head_dim).ok()?;
        let attention = layer.projection.run(&token_major, rows).ok()?;
        let mut residual = tokens;
        runtime.add_f32_in_place(&mut residual, &attention).ok()?;

        let mlp_layer = self.mlp.layers.get(layer_idx)?;
        let normalized = runtime
            .layernorm_f32_cpu_order(
                &residual,
                &mlp_layer.norm2_gamma,
                &mlp_layer.norm2_beta,
                rows,
                768,
                self.mlp.ln_eps,
            )
            .ok()?;
        let mut hidden = mlp_layer.fc1.run(&normalized, rows).ok()?;
        runtime.gelu_f32_in_place(&mut hidden).ok()?;
        let mlp = mlp_layer.fc2.run(&hidden, rows).ok()?;
        runtime.add_f32_in_place(&mut residual, &mlp).ok()?;
        Some(residual)
    }
}

#[cfg(feature = "cuda-residual-oracle")]
impl TransformerTailExecutor for CudaTransformerTailExecutor {
    fn persistent_cuda_tail(&self) -> Option<&CudaTransformerTailExecutor> {
        Some(self)
    }

    fn run_tail(
        &self,
        layer_idx: usize,
        tokens: &[f32],
        ln1: &[f32],
        rows: usize,
        gh: usize,
        gw: usize,
        global: bool,
        view_count: usize,
        cfg: &ModelConfig,
    ) -> Option<Vec<f32>> {
        let runtime = &self.attention.runtime;
        // Keep the `ln1` argument in the trait contract for the legacy
        // adapter. Device persistence deliberately recomputes it from
        // `tokens`, using cached parameters, so it never depends on a host
        // intermediate. The assertion also catches accidental caller drift.
        if ln1.len() != tokens.len() {
            return None;
        }
        let tokens = runtime.upload_f32(tokens).ok()?;
        let output =
            self.run_tail_device(layer_idx, tokens, rows, gh, gw, global, view_count, cfg)?;
        runtime.download_f32(&output).ok()
    }
}

#[cfg(feature = "cuda-residual-oracle")]
impl MlpExecutor for CudaMlpExecutor {
    fn run_mlp(&self, layer_idx: usize, input: &[f32], rows: usize) -> Vec<f32> {
        let layer = &self.layers[layer_idx];
        let input = self
            .runtime
            .upload_f32(input)
            .expect("CUDA MLP token upload must succeed");
        let normalized = self
            .runtime
            .layernorm_f32_cpu_order(
                &input,
                &layer.norm2_gamma,
                &layer.norm2_beta,
                rows,
                layer.fc1.input_features(),
                self.ln_eps,
            )
            .expect("CUDA LayerNorm must succeed");
        let mut hidden = layer
            .fc1
            .run(&normalized, rows)
            .expect("CUDA FC1 must succeed");
        self.runtime
            .gelu_f32_in_place(&mut hidden)
            .expect("CUDA GELU must succeed");
        let output = layer.fc2.run(&hidden, rows).expect("CUDA FC2 must succeed");
        self.runtime
            .download_f32(&output)
            .expect("CUDA MLP result download must succeed")
    }
}

impl MlpExecutor for PackedMlpExecutor {
    fn run_mlp(&self, layer_idx: usize, input: &[f32], rows: usize) -> Vec<f32> {
        let layer = &self.layers[layer_idx];
        assert_eq!(
            rows, 865,
            "packed MLP supports the locked DA3 token shape only"
        );
        assert_eq!(input.len(), rows * 768);

        let mut output = vec![0.0; rows * 768];
        // One outer Rayon launch owns a small set of rows through LN2, FC1,
        // GELU, and FC2. This keeps the ~72 KiB normalized and ~72 KiB
        // hidden activation for each six-row tile local to the worker instead
        // of materializing full 865-row matrices and launching two separate
        // projection jobs. The serial inner projection is deliberate: nested
        // Rayon launches would defeat the cache-local schedule.
        input
            .par_chunks(6 * 768)
            .zip(output.par_chunks_mut(6 * 768))
            .for_each_init(
                || (Vec::<f32>::new(), Vec::<f32>::new()),
                |(normalized, hidden), (input_rows, output_rows)| {
                    let tile_rows = input_rows.len() / 768;
                    normalized.clear();
                    normalized.extend_from_slice(input_rows);
                    vestra_kernels::scalar::layernorm(
                        normalized,
                        tile_rows,
                        768,
                        &layer.norm_gamma,
                        &layer.norm_beta,
                        self.ln_eps,
                    );
                    hidden.resize(tile_rows * 3072, 0.0);
                    assert!(layer.fc1.run_rows_serial(normalized, hidden));
                    vestra_kernels::scalar::add_bias_rows(
                        hidden,
                        tile_rows,
                        3072,
                        &layer.fc1_bias,
                    );
                    vestra_kernels::Kernels::detect().gelu(hidden);
                    assert!(layer.fc2.run_rows_serial(hidden, output_rows));
                    vestra_kernels::scalar::add_bias_rows(
                        output_rows,
                        tile_rows,
                        768,
                        &layer.fc2_bias,
                    );
                    if let Some(scale) = layer.ls2.as_deref() {
                        vestra_kernels::scalar::layerscale(output_rows, tile_rows, 768, scale);
                    }
                },
            );
        output
    }
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

/// Reusable CPU scratch for a ViT block's attention and MLP branches.
///
/// The DA3-BASE inference path visits twelve blocks with the same activation
/// geometry. Retaining these purely transient buffers avoids allocating and
/// zero-initialising several token-sized tensors at every block. Branch
/// buffers remain owned by this workspace until their residual addition has
/// completed.
#[derive(Default)]
pub(crate) struct VitWorkspace {
    // `norm` is shared by LN1 and LN2. The attention branch has consumed
    // LN1 before LN2 overwrites it.
    norm: Vec<f32>,
    // `branch` is shared by the MLP's FC2 output and retained only until its
    // residual addition completes.
    branch: Vec<f32>,
    mlp_hidden: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attention_heads: Vec<f32>,
    attention_tokens: Vec<f32>,
    qkv_fallback: Vec<f32>,
    positions_f32: Vec<f32>,
    positions_i64: Vec<i64>,
}

fn wname(layer_idx: usize, suffix: &str) -> String {
    format!("vit.blk.{layer_idx}.{suffix}")
}

/// Runs `x[rows,cols] -> LayerNorm(x, gamma, beta, eps)` directly on an
/// activation copy.  The model parameters are immutable and borrowed from
/// `Weights`; copying them into a per-operation graph arena was pure
/// overhead on the inference path.
fn run_layernorm_into(
    x_in: &[f32],
    rows: usize,
    cols: usize,
    gamma_name: &str,
    beta_name: &str,
    eps: f32,
    weights: &Weights,
    out: &mut [f32],
) {
    assert_eq!(
        out.len(),
        x_in.len(),
        "layernorm output has the wrong length"
    );
    out.copy_from_slice(x_in);
    let gamma = weights
        .get_f32(gamma_name)
        .unwrap_or_else(|| panic!("Weights missing f32 entry {gamma_name:?}"));
    let beta = weights
        .get_f32(beta_name)
        .unwrap_or_else(|| panic!("Weights missing f32 entry {beta_name:?}"));
    vestra_kernels::scalar::layernorm(out, rows, cols, gamma, beta, eps);
}

/// Runs `y = x[m,k] @ w[k,n] + bias[n]`, optionally followed by GELU and/or
/// an in-place LayerScale (`y *= ls_gamma[n]`, only if `ls_name` is `Some`
/// *and* that tensor is actually present in `weights` — presence-gated,
/// trap #3).
#[allow(clippy::too_many_arguments)]
fn run_linear_into(
    x_in: &[f32],
    m: usize,
    k: usize,
    n: usize,
    w_name: &str,
    b_name: &str,
    gelu: bool,
    ls_name: Option<&str>,
    weights: &Weights,
    out: &mut [f32],
) {
    let weight = weights
        .get_f32(w_name)
        .unwrap_or_else(|| panic!("Weights missing f32 entry {w_name:?}"));
    let bias = weights
        .get_f32(b_name)
        .unwrap_or_else(|| panic!("Weights missing f32 entry {b_name:?}"));
    assert_eq!(out.len(), m * n, "linear output has the wrong length");
    if !gelu {
        if let Some(name) = ls_name {
            if let Some(gamma) = weights.get_f32(name) {
                if vestra_kernels::linear_bias_scale_f32_da3_base(
                    m, n, k, x_in, weight, bias, gamma, out,
                ) {
                    return;
                }
            }
        }
    }
    Da3ProjectionGemm.gemm(m, n, k, x_in, weight, out);
    vestra_kernels::scalar::add_bias_rows(out, m, n, bias);
    if gelu {
        vestra_kernels::Kernels::detect().gelu(out);
    }
    if let Some(name) = ls_name {
        if let Some(gamma) = weights.get_f32(name) {
            vestra_kernels::scalar::layerscale(out, m, n, gamma);
        }
    }
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
    attention_executor: Option<&dyn AttentionExecutor>,
    workspace: &mut VitWorkspace,
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
    let activation_len = heads * n * head_dim;
    let mut q = std::mem::take(&mut workspace.q);
    let mut k = std::mem::take(&mut workspace.k);
    let mut v = std::mem::take(&mut workspace.v);
    q.resize(activation_len, 0.0);
    k.resize(activation_len, 0.0);
    v.resize(activation_len, 0.0);
    let qkv_started = std::time::Instant::now();
    let qkv_weight = weights
        .get_f32(&wname(layer_idx, "attn_qkv.weight"))
        .unwrap();
    let qkv_bias = weights.get_f32(&wname(layer_idx, "attn_qkv.bias")).unwrap();
    let direct_qkv =
        vestra_kernels::qkv_f32_da3_base(ln1_out, qkv_weight, qkv_bias, &mut q, &mut k, &mut v);
    let qkv_elapsed = qkv_started.elapsed();
    let pack_elapsed = if direct_qkv {
        std::time::Duration::ZERO
    } else {
        let mut qkv = std::mem::take(&mut workspace.qkv_fallback);
        qkv.resize(n * 3 * embed, 0.0);
        run_linear_into(
            ln1_out,
            n,
            embed,
            3 * embed,
            &wname(layer_idx, "attn_qkv.weight"),
            &wname(layer_idx, "attn_qkv.bias"),
            false,
            None,
            weights,
            &mut qkv,
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
        workspace.qkv_fallback = qkv;
        pack_started.elapsed()
    };

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
    let mut pos_yx = std::mem::take(&mut workspace.positions_f32);
    if use_rope {
        pos_yx.resize(n * 2, 0.0);
        pos_yx.fill(0.0);
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
                    pos_yx[2 * t] = 1.0;
                    pos_yx[2 * t + 1] = 1.0;
                } else {
                    let idx = local_t - n_special;
                    let row = idx / gw;
                    let col = idx % gw;
                    pos_yx[2 * t] = (row + 1) as f32;
                    pos_yx[2 * t + 1] = (col + 1) as f32;
                }
            }
        }
    } else {
        pos_yx.clear();
    }

    if use_qknorm && use_rope {
        if let Some(executor) = attention_executor {
            if let Some(output) =
                executor.run_attention(layer_idx, ln1_out, n, heads, head_dim, Some(&pos_yx))
            {
                workspace.q = q;
                workspace.k = k;
                workspace.v = v;
                workspace.positions_f32 = pos_yx;
                return output;
            }
        }
    }

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
            let mut positions = std::mem::take(&mut workspace.positions_i64);
            positions.clear();
            positions.extend(pos_yx.iter().map(|&value| value as i64));
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
            workspace.positions_i64 = positions;
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
        let mut positions = std::mem::take(&mut workspace.positions_i64);
        positions.clear();
        positions.extend(pos_yx.iter().map(|&value| value as i64));
        vestra_kernels::rope2d(&mut q, heads, n, head_dim, &positions, cfg.rope_freq);
        vestra_kernels::rope2d(&mut k, heads, n, head_dim, &positions, cfg.rope_freq);
        workspace.positions_i64 = positions;
    }
    let position_elapsed = position_started.elapsed();
    let core_started = std::time::Instant::now();
    let mut attn_hnd = std::mem::take(&mut workspace.attention_heads);
    attn_hnd.resize(activation_len, 0.0);
    vestra_kernels::attention(&q, &k, &v, heads, n, head_dim, &mut attn_hnd);
    let core_elapsed = core_started.elapsed();

    // Transpose head-major [heads, n, head_dim] back to token-major
    // [n, embed] before the output projection.
    let unpack_started = std::time::Instant::now();
    let mut attn_tok = std::mem::take(&mut workspace.attention_tokens);
    attn_tok.resize(n * embed, 0.0);
    for t in 0..n {
        for h in 0..heads {
            for d in 0..head_dim {
                attn_tok[t * embed + h * head_dim + d] = attn_hnd[(h * n + t) * head_dim + d];
            }
        }
    }
    let unpack_elapsed = unpack_started.elapsed();

    let projection_started = std::time::Instant::now();
    let mut output = std::mem::take(&mut workspace.branch);
    output.resize(n * embed, 0.0);
    run_linear_into(
        &attn_tok,
        n,
        embed,
        embed,
        &wname(layer_idx, "attn_proj.weight"),
        &wname(layer_idx, "attn_proj.bias"),
        false,
        Some(&wname(layer_idx, "ls1")),
        weights,
        &mut output,
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
    workspace.q = q;
    workspace.k = k;
    workspace.v = v;
    workspace.attention_heads = attn_hnd;
    workspace.attention_tokens = attn_tok;
    workspace.positions_f32 = pos_yx;
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
fn vit_block_with_views_workspace(
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
    mlp_executor: Option<&dyn MlpExecutor>,
    attention_executor: Option<&dyn AttentionExecutor>,
    transformer_tail_executor: Option<&dyn TransformerTailExecutor>,
    workspace: &mut VitWorkspace,
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
    // `tokens` is read into reusable LN scratch, then remains the
    // pre-attention residual until the in-place addition below.
    let ln1_started = std::time::Instant::now();
    let mut norm = std::mem::take(&mut workspace.norm);
    norm.resize(n * embed, 0.0);
    run_layernorm_into(
        tokens,
        n,
        embed,
        &wname(layer_idx, "norm1.weight"),
        &wname(layer_idx, "norm1.bias"),
        eps,
        weights,
        &mut norm,
    );
    let ln1_elapsed = ln1_started.elapsed();
    if let Some(executor) = transformer_tail_executor {
        if let Some(result) =
            executor.run_tail(layer_idx, tokens, &norm, n, gh, gw, global, view_count, cfg)
        {
            tokens.copy_from_slice(&result);
            workspace.norm = norm;
            return;
        }
    }
    let attention_started = std::time::Instant::now();
    let attn_out = run_attention(
        &norm,
        n,
        gh,
        gw,
        global,
        view_count,
        cfg,
        layer_idx,
        weights,
        attention_executor,
        workspace,
    );
    add_residual(tokens, &attn_out, residual_executor);
    // The attention projection and FC2 have the same [n, embed] shape. Its
    // buffer is no longer read after the residual addition, so return it to
    // the shared branch slot for the following MLP and next block.
    workspace.branch = attn_out;
    let attention_elapsed = attention_started.elapsed();

    // --- MLP sub-block --- (same "tokens still holds the residual" trick)
    let mlp_started = std::time::Instant::now();
    if let Some(executor) = mlp_executor {
        let m = executor.run_mlp(layer_idx, tokens, n);
        add_residual(tokens, &m, residual_executor);
    } else {
        run_layernorm_into(
            tokens,
            n,
            embed,
            &wname(layer_idx, "norm2.weight"),
            &wname(layer_idx, "norm2.bias"),
            eps,
            weights,
            &mut norm,
        );
        let mut hidden = std::mem::take(&mut workspace.mlp_hidden);
        hidden.resize(n * mlp_hidden, 0.0);
        run_linear_into(
            &norm,
            n,
            embed,
            mlp_hidden,
            &wname(layer_idx, "mlp_fc1.weight"),
            &wname(layer_idx, "mlp_fc1.bias"),
            true,
            None,
            weights,
            &mut hidden,
        );
        let mut branch = std::mem::take(&mut workspace.branch);
        branch.resize(n * embed, 0.0);
        run_linear_into(
            &hidden,
            n,
            mlp_hidden,
            embed,
            &wname(layer_idx, "mlp_fc2.weight"),
            &wname(layer_idx, "mlp_fc2.bias"),
            false,
            Some(&wname(layer_idx, "ls2")),
            weights,
            &mut branch,
        );
        add_residual(tokens, &branch, residual_executor);
        workspace.mlp_hidden = hidden;
        workspace.branch = branch;
    }
    workspace.norm = norm;
    if phase_profile {
        eprintln!(
            "phase: block[{layer_idx}] ln1={:.3}ms attention={:.3}ms mlp_with_ln2={:.3}ms",
            ln1_elapsed.as_secs_f64() * 1e3,
            attention_elapsed.as_secs_f64() * 1e3,
            mlp_started.elapsed().as_secs_f64() * 1e3,
        );
    }
}

/// Runs a transformer block with fresh transient attention buffers.
///
/// This remains the compatibility route for the multiview adapters.  The
/// single-view inference path below keeps one workspace for its whole block
/// stack.
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
    backend: &dyn Backend,
    residual_executor: Option<&dyn ResidualAddExecutor>,
    mlp_executor: Option<&dyn MlpExecutor>,
    attention_executor: Option<&dyn AttentionExecutor>,
    transformer_tail_executor: Option<&dyn TransformerTailExecutor>,
) {
    let mut workspace = VitWorkspace::default();
    vit_block_with_views_workspace(
        tokens,
        n,
        gh,
        gw,
        global,
        view_count,
        cfg,
        layer_idx,
        weights,
        backend,
        residual_executor,
        mlp_executor,
        attention_executor,
        transformer_tail_executor,
        &mut workspace,
    );
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
    mlp_executor: Option<&dyn MlpExecutor>,
    attention_executor: Option<&dyn AttentionExecutor>,
    transformer_tail_executor: Option<&dyn TransformerTailExecutor>,
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
        mlp_executor,
        attention_executor,
        transformer_tail_executor,
    );
}

/// Single-view block route sharing an attention scratch workspace with its
/// neighbouring blocks.  It intentionally leaves the public compatibility
/// entry point above unchanged for the multiview schedulers.
#[allow(clippy::too_many_arguments)]
pub(crate) fn vit_block_with_residual_workspace(
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
    mlp_executor: Option<&dyn MlpExecutor>,
    attention_executor: Option<&dyn AttentionExecutor>,
    transformer_tail_executor: Option<&dyn TransformerTailExecutor>,
    workspace: &mut VitWorkspace,
) {
    vit_block_with_views_workspace(
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
        mlp_executor,
        attention_executor,
        transformer_tail_executor,
        workspace,
    );
}

#[allow(clippy::too_many_arguments)]
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
        tokens, n, gh, gw, global, cfg, layer_idx, weights, backend, None, None, None, None,
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

        let mut direct_norm = vec![0.0; input.len()];
        run_layernorm_into(
            &input,
            3,
            4,
            "norm.g",
            "norm.b",
            1e-6,
            &weights,
            &mut direct_norm,
        );
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

        let mut direct_linear = vec![0.0; 3 * 5];
        run_linear_into(
            &input,
            3,
            4,
            5,
            "linear.w",
            "linear.b",
            true,
            Some("linear.ls"),
            &weights,
            &mut direct_linear,
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
    fn attention_workspace_matches_fresh_route_after_reuse() {
        let cfg = test_cfg(8, 2, 4, 16);
        let weights = synthetic_weights(&cfg, 0, true, false);
        let backend = CpuBackend::new();
        let n = 5usize;
        let mut rng = Xorshift32(0xA771_0A11);
        let input = random_vec(&mut rng, n * cfg.embed_dim as usize);

        let mut fresh = input.clone();
        vit_block_with_residual(
            &mut fresh, n, 2, 2, false, &cfg, 0, &weights, &backend, None, None, None, None,
        );

        let mut workspace = VitWorkspace::default();
        let mut reused = input.clone();
        vit_block_with_residual_workspace(
            &mut reused,
            n,
            2,
            2,
            false,
            &cfg,
            0,
            &weights,
            &backend,
            None,
            None,
            None,
            None,
            &mut workspace,
        );
        let activation_capacities = (
            workspace.norm.capacity(),
            workspace.mlp_hidden.capacity(),
            workspace.branch.capacity(),
        );
        for buffer in [
            &mut workspace.norm,
            &mut workspace.branch,
            &mut workspace.mlp_hidden,
            &mut workspace.q,
            &mut workspace.k,
            &mut workspace.v,
            &mut workspace.attention_heads,
            &mut workspace.attention_tokens,
            &mut workspace.qkv_fallback,
            &mut workspace.positions_f32,
        ] {
            buffer.fill(f32::NAN);
        }
        workspace.positions_i64.fill(i64::MIN);

        let mut reused_after_poison = input;
        vit_block_with_residual_workspace(
            &mut reused_after_poison,
            n,
            2,
            2,
            false,
            &cfg,
            0,
            &weights,
            &backend,
            None,
            None,
            None,
            None,
            &mut workspace,
        );

        assert_eq!(
            activation_capacities,
            (
                workspace.norm.capacity(),
                workspace.mlp_hidden.capacity(),
                workspace.branch.capacity(),
            ),
            "fixed-geometry reuse must retain the MLP activation allocations"
        );

        for ((expected, first_reuse), second_reuse) in
            fresh.iter().zip(&reused).zip(&reused_after_poison)
        {
            assert_eq!(expected.to_bits(), first_reuse.to_bits());
            assert_eq!(expected.to_bits(), second_reuse.to_bits());
        }
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
