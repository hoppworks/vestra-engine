use da_kernels::gemm::{FaerGemm, Gemm};
use da_kernels::Kernels;

use crate::backend::Backend;
use crate::graph::{Op, Weights};
use crate::tensor::TensorId;
use crate::Arena;

/// The single-threaded CPU backend: SIMD-dispatched elementwise/attention
/// kernels (via [`Kernels`]) plus a GEMM implementation (via [`Gemm`]),
/// both from `da-kernels`.
pub struct CpuBackend {
    kernels: Kernels,
    gemm: Box<dyn Gemm>,
}

impl CpuBackend {
    /// Detect the host ISA for elementwise kernels and use `faer` for GEMM.
    pub fn new() -> Self {
        CpuBackend {
            kernels: Kernels::detect(),
            gemm: Box::new(FaerGemm),
        }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

/// Local newtype wrapping a `&dyn Gemm` so it can itself implement the
/// (foreign) `Gemm` trait — needed because `da_kernels::conv::conv2d` is
/// generic over `impl Gemm` (so `Sized`), and `Box<dyn Gemm>` can't be
/// passed there directly, and implementing `Gemm` for `Box<dyn Gemm>`
/// itself would violate the orphan rules (both the trait and `Box<dyn
/// Gemm>`'s inner type are foreign to this crate). `GemmRef` is a local
/// type, so implementing the foreign `Gemm` trait for it is fine.
struct GemmRef<'a>(&'a dyn Gemm);
impl<'a> Gemm for GemmRef<'a> {
    fn gemm(&self, m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
        self.0.gemm(m, n, k, a, b, c);
    }
}

/// Get a raw `(ptr, len)` pair for tensor `id`'s slot in `arena`, without
/// holding on to a borrow of `arena` itself.
///
/// # Safety contract relied on by callers
/// `Arena::plan` (Task 12) guarantees that any two tensors whose `(first_use,
/// last_use)` intervals overlap are given *disjoint* memory. `Graph::compute_lifetimes`
/// (this task) touches every operand and the output of an `Op` at the same
/// step, so every op's operands and output necessarily have overlapping
/// lifetimes at that step and therefore live in disjoint arena regions.
/// That's what makes it sound to reconstruct multiple simultaneous
/// (immutable + mutable) slices from raw pointers below, bypassing
/// `Arena::buf`'s single-`&mut self` borrow, for a single op's tensors.
fn raw_parts(arena: &mut Arena, id: TensorId) -> (*mut f32, usize) {
    let s = arena.buf(id);
    (s.as_mut_ptr(), s.len())
}

impl Backend for CpuBackend {
    fn execute(&self, op: &Op, arena: &mut Arena, _weights: &Weights) {
        match *op {
            Op::Gemm { a, b, out, m, n, k } => {
                let (a_ptr, a_len) = raw_parts(arena, a);
                let (b_ptr, b_len) = raw_parts(arena, b);
                let (out_ptr, out_len) = raw_parts(arena, out);
                // SAFETY: see `raw_parts` doc comment — a, b, out all have
                // overlapping lifetime at this step, so they're disjoint.
                let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, a_len) };
                let b_slice = unsafe { std::slice::from_raw_parts(b_ptr, b_len) };
                let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, out_len) };
                self.gemm.gemm(m, n, k, a_slice, b_slice, out_slice);
            }
            Op::AddBias { x, bias, rows, cols } => {
                let (x_ptr, x_len) = raw_parts(arena, x);
                let (bias_ptr, bias_len) = raw_parts(arena, bias);
                // SAFETY: see `raw_parts` doc comment.
                let x_slice = unsafe { std::slice::from_raw_parts_mut(x_ptr, x_len) };
                let bias_slice = unsafe { std::slice::from_raw_parts(bias_ptr, bias_len) };
                da_kernels::scalar::add_bias_rows(x_slice, rows, cols, bias_slice);
            }
            Op::Gelu { x } => {
                self.kernels.gelu(arena.buf(x));
            }
            Op::LayerNorm {
                x,
                g,
                b,
                rows,
                cols,
                eps,
            } => {
                let (x_ptr, x_len) = raw_parts(arena, x);
                let (g_ptr, g_len) = raw_parts(arena, g);
                let (b_ptr, b_len) = raw_parts(arena, b);
                // SAFETY: see `raw_parts` doc comment.
                let x_slice = unsafe { std::slice::from_raw_parts_mut(x_ptr, x_len) };
                let g_slice = unsafe { std::slice::from_raw_parts(g_ptr, g_len) };
                let b_slice = unsafe { std::slice::from_raw_parts(b_ptr, b_len) };
                da_kernels::scalar::layernorm(x_slice, rows, cols, g_slice, b_slice, eps);
            }
            Op::Attention {
                q,
                k,
                v,
                heads,
                n,
                head_dim,
                out,
            } => {
                let (q_ptr, q_len) = raw_parts(arena, q);
                let (k_ptr, k_len) = raw_parts(arena, k);
                let (v_ptr, v_len) = raw_parts(arena, v);
                let (out_ptr, out_len) = raw_parts(arena, out);
                // SAFETY: see `raw_parts` doc comment.
                let q_slice = unsafe { std::slice::from_raw_parts(q_ptr, q_len) };
                let k_slice = unsafe { std::slice::from_raw_parts(k_ptr, k_len) };
                let v_slice = unsafe { std::slice::from_raw_parts(v_ptr, v_len) };
                let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, out_len) };
                da_kernels::attention(q_slice, k_slice, v_slice, heads, n, head_dim, out_slice);
            }
            Op::Conv2d {
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
            } => {
                let (input_ptr, input_len) = raw_parts(arena, input);
                let (weight_ptr, weight_len) = raw_parts(arena, weight);
                let bias_raw = bias.map(|id| raw_parts(arena, id));
                let (out_ptr, out_len) = raw_parts(arena, out);
                // SAFETY: see `raw_parts` doc comment.
                let input_slice = unsafe { std::slice::from_raw_parts(input_ptr, input_len) };
                let weight_slice = unsafe { std::slice::from_raw_parts(weight_ptr, weight_len) };
                let bias_slice =
                    bias_raw.map(|(ptr, len)| unsafe { std::slice::from_raw_parts(ptr, len) });
                let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, out_len) };
                da_kernels::conv2d(
                    input_slice,
                    in_c,
                    ih,
                    iw,
                    weight_slice,
                    out_c,
                    kh,
                    kw,
                    stride,
                    pad,
                    bias_slice,
                    &GemmRef(self.gemm.as_ref()),
                    out_slice,
                );
            }
        }
    }
}
