use std::collections::HashMap;

use crate::tensor::TensorId;
use crate::Plan;

/// A single operation in the static op graph. Every op reads zero or more
/// `TensorId`s and either writes a fresh `out: TensorId` (`Gemm`,
/// `Attention`, `Conv2d`) or mutates its `x`/input tensor in place (`AddBias`,
/// `Gelu`, `LayerNorm` — matching the in-place `da_kernels` functions they
/// wrap).
///
/// Only the variants exercised by this task's own test (`Gemm`, `AddBias`,
/// `Gelu`) are required to work end-to-end right now. `LayerNorm`,
/// `Attention` and `Conv2d` exist with sensible signatures (mirroring the
/// `da_kernels` functions they'll dispatch to) so `CpuBackend::execute` can
/// already handle them, but they're only exercised by later milestones'
/// parity tests.
/// Precomputed 2D-RoPE inputs for [`Op::Attention`]: `pos_yx` is a
/// `TensorId` of `n*2` f32-encoded `(y, x)` integer grid positions (row-major
/// per token, matching `da_kernels::rope2d`'s `pos_yx` argument once cast
/// back to `i64`), and `freq` is the rotation base.
#[derive(Debug, Clone, Copy)]
pub struct RopeParams {
    pub pos_yx: TensorId,
    pub freq: f32,
}

#[derive(Debug, Clone, Copy)]
pub enum Op {
    /// `out[m,n] = a[m,k] @ b[k,n]`.
    Gemm {
        a: TensorId,
        b: TensorId,
        out: TensorId,
        m: usize,
        n: usize,
        k: usize,
    },
    /// In-place `x[rows,cols] += bias[cols]` (broadcast per row).
    AddBias {
        x: TensorId,
        bias: TensorId,
        rows: usize,
        cols: usize,
    },
    /// In-place GELU over `x`.
    Gelu { x: TensorId },
    /// In-place LayerNorm over `x[rows,cols]` with per-column scale `g` and
    /// shift `b`.
    LayerNorm {
        x: TensorId,
        g: TensorId,
        b: TensorId,
        rows: usize,
        cols: usize,
        eps: f32,
    },
    /// In-place per-column scale (LayerScale / DINOv2 `ls1`/`ls2`):
    /// `x[r,c] *= gamma[c]`, mirroring `AddBias`'s shape convention.
    /// Wraps `da_kernels::scalar::layerscale`.
    LayerScale {
        x: TensorId,
        gamma: TensorId,
        rows: usize,
        cols: usize,
    },
    /// Scaled-dot-product attention, revised (Task 17) to also cover the two
    /// ViT-block traps that live *inside* attention rather than around it:
    /// per-head QK-LayerNorm and 2D-RoPE. `q`,`k`,`v` and `out` are
    /// `[heads, n, head_dim]` row-major, matching
    /// `da_kernels::attention::attention` — `q`/`k` are the *raw*
    /// (pre-norm, pre-RoPE) projections; `CpuBackend::execute` mutates them
    /// in place (qk-norm, then RoPE, both optional) before running the
    /// softmax(QK^T/sqrt(d))V core into `out`.
    ///
    /// `qnorm`/`knorm`, when `Some`, are `(gamma, beta)` `TensorId` pairs of
    /// length `head_dim`, applied as a LayerNorm over each `(head, token)`
    /// row of `q`/`k` (i.e. `rows = heads*n, cols = head_dim`) at
    /// `qk_norm_eps` — a distinct epsilon from any block-level LayerNorm's
    /// `eps`, because the reference model's per-head q_norm/k_norm are
    /// constructed with torch's *default* `nn.LayerNorm` eps (`1e-5`), not
    /// the block's `ln_eps` (see `src/vit_block.cpp`'s "reference parity
    /// note" comment: `QK_NORM_EPS = 1e-5f`, distinct from `ln_eps=1e-6`).
    /// Only present (`Some`) when the model both has qn/kn weight tensors
    /// *and* the calling layer index is `>= cfg.qknorm_start`.
    ///
    /// `rope`, when `Some`, carries precomputed per-token `(y, x)` integer
    /// grid positions (as a `TensorId` of `n*2` f32-encoded values — the
    /// arena only stores `f32`; values are exact for any realistic token
    /// count) plus the rotation base `freq`, applied via
    /// `da_kernels::rope2d` after qk-norm. Only present when the calling
    /// layer index is `>= cfg.rope_start`.
    Attention {
        q: TensorId,
        k: TensorId,
        v: TensorId,
        heads: usize,
        n: usize,
        head_dim: usize,
        qnorm: Option<(TensorId, TensorId)>,
        knorm: Option<(TensorId, TensorId)>,
        qk_norm_eps: f32,
        rope: Option<RopeParams>,
        out: TensorId,
    },
    /// Standard NCHW (batch=1) Conv2d via im2col+GEMM, matching
    /// `da_kernels::conv::conv2d`.
    Conv2d {
        input: TensorId,
        weight: TensorId,
        bias: Option<TensorId>,
        in_c: usize,
        ih: usize,
        iw: usize,
        out_c: usize,
        kh: usize,
        kw: usize,
        stride: usize,
        pad: usize,
        out: TensorId,
    },
}

impl Op {
    /// Every `TensorId` this op reads or writes, in no particular order.
    /// Used by [`Graph::compute_lifetimes`] to derive `(first_use,
    /// last_use)` intervals for `Arena::plan`.
    fn touches(&self, mut f: impl FnMut(TensorId)) {
        match *self {
            Op::Gemm { a, b, out, .. } => {
                f(a);
                f(b);
                f(out);
            }
            Op::AddBias { x, bias, .. } => {
                f(x);
                f(bias);
            }
            Op::Gelu { x } => f(x),
            Op::LayerNorm { x, g, b, .. } => {
                f(x);
                f(g);
                f(b);
            }
            Op::LayerScale { x, gamma, .. } => {
                f(x);
                f(gamma);
            }
            Op::Attention {
                q,
                k,
                v,
                out,
                qnorm,
                knorm,
                rope,
                ..
            } => {
                f(q);
                f(k);
                f(v);
                f(out);
                if let Some((g, b)) = qnorm {
                    f(g);
                    f(b);
                }
                if let Some((g, b)) = knorm {
                    f(g);
                    f(b);
                }
                if let Some(RopeParams { pos_yx, .. }) = rope {
                    f(pos_yx);
                }
            }
            Op::Conv2d {
                input,
                weight,
                bias,
                out,
                ..
            } => {
                f(input);
                f(weight);
                if let Some(bias) = bias {
                    f(bias);
                }
                f(out);
            }
        }
    }
}

/// A static op graph: a flat list of [`Op`]s plus the tensors that are
/// graph inputs (filled from `Plan::run`'s `inputs` argument) and graph
/// outputs (read back after all ops have executed).
///
/// `sizes[id.0]` is the number of `f32` elements tensor `id` holds;
/// `weight_tensors[id]`, when present, names the entry in a [`Weights`]
/// map that `Plan::run` copies into that tensor's arena slot before
/// executing any op (this is how ops reference GGUF weights while still
/// only ever taking `TensorId`s, per the op signatures above).
#[derive(Debug, Clone)]
pub struct Graph {
    pub ops: Vec<Op>,
    pub inputs: Vec<TensorId>,
    pub outputs: Vec<TensorId>,
    pub sizes: Vec<usize>,
    pub weight_tensors: HashMap<TensorId, String>,
}

impl Graph {
    pub fn builder() -> GraphBuilder {
        GraphBuilder::new()
    }

    /// Derive `(first_use, last_use)` intervals for every tensor, in "step"
    /// units (index into `ops`). Graph inputs and weights are considered
    /// live from step 0 (they're filled before any op runs); graph outputs
    /// are kept alive through a virtual step `ops.len()` so nothing in the
    /// arena gets reused for their memory before `Plan::run` reads them
    /// back, right after the last op executes.
    pub(crate) fn compute_lifetimes(&self) -> Vec<(usize, usize)> {
        let n = self.sizes.len();
        let mut first = vec![usize::MAX; n];
        let mut last = vec![0usize; n];

        let touch = |id: TensorId, step: usize, first: &mut [usize], last: &mut [usize]| {
            let i = id.0;
            if step < first[i] {
                first[i] = step;
            }
            if step > last[i] {
                last[i] = step;
            }
        };

        for (step, op) in self.ops.iter().enumerate() {
            op.touches(|id| touch(id, step, &mut first, &mut last));
        }
        for &id in &self.inputs {
            touch(id, 0, &mut first, &mut last);
        }
        for &id in self.weight_tensors.keys() {
            touch(id, 0, &mut first, &mut last);
        }
        for &id in &self.outputs {
            touch(id, self.ops.len(), &mut first, &mut last);
        }

        (0..n)
            .map(|i| {
                if first[i] == usize::MAX {
                    // Never referenced by any op/input/output — give it a
                    // degenerate (but valid) lifetime rather than panic.
                    (0, 0)
                } else {
                    (first[i], last[i])
                }
            })
            .collect()
    }

    /// Derive tensor lifetimes and plan an [`crate::Arena`] for them once.
    /// The returned [`Plan`] reuses that single arena allocation across
    /// every `run()` call (via interior mutability) — no per-run
    /// allocation of activation storage.
    pub fn compile(&self) -> Plan {
        Plan::new(self.clone())
    }
}

/// Incrementally builds a [`Graph`]. Each op-adding method returns the
/// `TensorId` of the value it produces (or, for in-place ops, the same
/// `TensorId` that was passed in), so calls can be chained naturally.
pub struct GraphBuilder {
    sizes: Vec<usize>,
    ops: Vec<Op>,
    inputs: Vec<TensorId>,
    outputs: Vec<TensorId>,
    weight_tensors: HashMap<TensorId, String>,
}

impl GraphBuilder {
    pub fn new() -> Self {
        GraphBuilder {
            sizes: Vec::new(),
            ops: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            weight_tensors: HashMap::new(),
        }
    }

    fn alloc(&mut self, numel: usize) -> TensorId {
        let id = TensorId(self.sizes.len());
        self.sizes.push(numel);
        id
    }

    /// Declare a graph input of `numel` elements; filled from `Plan::run`'s
    /// `inputs` slice, in declaration order.
    pub fn input(&mut self, numel: usize) -> TensorId {
        let id = self.alloc(numel);
        self.inputs.push(id);
        id
    }

    /// Declare a weight tensor of `numel` elements, backed by `name` in
    /// whatever [`Weights`] is passed to `Plan::run`.
    pub fn weight(&mut self, name: impl Into<String>, numel: usize) -> TensorId {
        let id = self.alloc(numel);
        self.weight_tensors.insert(id, name.into());
        id
    }

    pub fn gemm(&mut self, a: TensorId, b: TensorId, m: usize, n: usize, k: usize) -> TensorId {
        let out = self.alloc(m * n);
        self.ops.push(Op::Gemm { a, b, out, m, n, k });
        out
    }

    pub fn add_bias(&mut self, x: TensorId, bias: TensorId, rows: usize, cols: usize) -> TensorId {
        self.ops.push(Op::AddBias { x, bias, rows, cols });
        x
    }

    pub fn gelu(&mut self, x: TensorId) -> TensorId {
        self.ops.push(Op::Gelu { x });
        x
    }

    pub fn layer_norm(
        &mut self,
        x: TensorId,
        g: TensorId,
        b: TensorId,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> TensorId {
        self.ops.push(Op::LayerNorm {
            x,
            g,
            b,
            rows,
            cols,
            eps,
        });
        x
    }

    pub fn layer_scale(&mut self, x: TensorId, gamma: TensorId, rows: usize, cols: usize) -> TensorId {
        self.ops.push(Op::LayerScale { x, gamma, rows, cols });
        x
    }

    /// Plain scaled-dot-product attention: no qk-norm, no RoPE. Equivalent
    /// to `attention_full(q, k, v, heads, n, head_dim, None, None, qk_norm_eps, None)`
    /// for any `qk_norm_eps` (unused when both norms are `None`).
    pub fn attention(
        &mut self,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        heads: usize,
        n: usize,
        head_dim: usize,
    ) -> TensorId {
        self.attention_full(q, k, v, heads, n, head_dim, None, None, 1e-5, None)
    }

    /// Full attention op: optional per-head QK-LayerNorm (`qnorm`/`knorm`,
    /// each `(gamma, beta)` of length `head_dim`, applied at `qk_norm_eps`)
    /// and optional 2D-RoPE (`rope`), both applied to `q`/`k` in place before
    /// the softmax(QK^T/sqrt(d))V core. See [`Op::Attention`]'s doc comment
    /// for the full contract.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_full(
        &mut self,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        heads: usize,
        n: usize,
        head_dim: usize,
        qnorm: Option<(TensorId, TensorId)>,
        knorm: Option<(TensorId, TensorId)>,
        qk_norm_eps: f32,
        rope: Option<RopeParams>,
    ) -> TensorId {
        let out = self.alloc(heads * n * head_dim);
        self.ops.push(Op::Attention {
            q,
            k,
            v,
            heads,
            n,
            head_dim,
            qnorm,
            knorm,
            qk_norm_eps,
            rope,
            out,
        });
        out
    }

    #[allow(clippy::too_many_arguments)]
    pub fn conv2d(
        &mut self,
        input: TensorId,
        weight: TensorId,
        bias: Option<TensorId>,
        in_c: usize,
        ih: usize,
        iw: usize,
        out_c: usize,
        kh: usize,
        kw: usize,
        stride: usize,
        pad: usize,
    ) -> TensorId {
        let oh = (ih + 2 * pad - kh) / stride + 1;
        let ow = (iw + 2 * pad - kw) / stride + 1;
        let out = self.alloc(out_c * oh * ow);
        self.ops.push(Op::Conv2d {
            input,
            weight,
            bias,
            in_c,
            ih,
            iw,
            out_c,
            kh,
            kw,
            stride,
            pad,
            out,
        });
        out
    }

    /// Mark `id` as a graph output; `Plan::run` reads it back (in the order
    /// `output` was called) after all ops have executed.
    pub fn output(&mut self, id: TensorId) {
        self.outputs.push(id);
    }

    pub fn build(self) -> Graph {
        Graph {
            ops: self.ops,
            inputs: self.inputs,
            outputs: self.outputs,
            sizes: self.sizes,
            weight_tensors: self.weight_tensors,
        }
    }
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// A name -> tensor map for GGUF-sourced weights, holding either plain
/// `f32` tensors or quantized `q8_0` blocks (plus their logical shape).
///
/// This task only defines the type and lets tests populate it manually;
/// actually loading it from a real GGUF file is da-engine's job (M5).
#[derive(Default)]
pub struct Weights {
    f32: HashMap<String, Vec<f32>>,
    q8_0: HashMap<String, (Vec<da_gguf::BlockQ8_0>, Vec<i64>)>,
}

impl Weights {
    pub fn new() -> Self {
        Weights {
            f32: HashMap::new(),
            q8_0: HashMap::new(),
        }
    }

    pub fn insert_f32(&mut self, name: impl Into<String>, data: Vec<f32>) {
        self.f32.insert(name.into(), data);
    }

    pub fn get_f32(&self, name: &str) -> Option<&[f32]> {
        self.f32.get(name).map(|v| v.as_slice())
    }

    pub fn insert_q8_0(&mut self, name: impl Into<String>, blocks: Vec<da_gguf::BlockQ8_0>, shape: Vec<i64>) {
        self.q8_0.insert(name.into(), (blocks, shape));
    }

    pub fn get_q8_0(&self, name: &str) -> Option<(&[da_gguf::BlockQ8_0], &[i64])> {
        self.q8_0.get(name).map(|(b, s)| (b.as_slice(), s.as_slice()))
    }
}
