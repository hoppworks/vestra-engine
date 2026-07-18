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
    /// Scaled-dot-product attention. `q`,`k`,`v` and `out` are
    /// `[heads, n, head_dim]` row-major, matching
    /// `da_kernels::attention::attention`.
    Attention {
        q: TensorId,
        k: TensorId,
        v: TensorId,
        heads: usize,
        n: usize,
        head_dim: usize,
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
            Op::Attention {
                q, k, v, out, ..
            } => {
                f(q);
                f(k);
                f(v);
                f(out);
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

    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        &mut self,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        heads: usize,
        n: usize,
        head_dim: usize,
    ) -> TensorId {
        let out = self.alloc(heads * n * head_dim);
        self.ops.push(Op::Attention {
            q,
            k,
            v,
            heads,
            n,
            head_dim,
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
