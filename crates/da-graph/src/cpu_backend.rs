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
///
/// That soundness argument depends entirely on an invariant enforced in a
/// *different* file (`Arena::plan`'s free-list release logic in `arena.rs`).
/// Nothing here would catch a future refactor there that weakens it (e.g.
/// changing the strict `<` in `block.last_use < first_use` to `<=`). Each
/// call site below therefore also fetches `arena.range(id)` (in debug builds
/// only, via `#[cfg(debug_assertions)]`) and checks it against every other
/// operand's range with `assert_disjoint_or_shared_read` before doing the
/// unsafe pointer split — see that function's doc comment for the backstop
/// this provides.
fn raw_parts(arena: &mut Arena, id: TensorId) -> (*mut f32, usize) {
    let s = arena.buf(id);
    (s.as_mut_ptr(), s.len())
}

/// Debug-only backstop for the safety argument in `raw_parts`' doc comment.
///
/// Takes every arena range touched by a single `Op`'s operands, each tagged
/// with whether it's read mutably (i.e. is a write target) by this op, and
/// asserts that any two ranges overlap only if *both* are read-only —
/// two immutable views may safely alias, but a write must never overlap
/// any other range (read or write). This doesn't prove `raw_parts` sound in
/// release builds (the check is compiled out there, matching this
/// codebase's established `debug_assert!` pattern), but it turns a future
/// silent-aliasing regression in `Arena::plan` into an immediate test
/// failure instead of a mysterious crash much later.
#[cfg(debug_assertions)]
fn assert_disjoint_or_shared_read(ranges: &[(&str, usize, usize, bool)]) {
    for i in 0..ranges.len() {
        for j in (i + 1)..ranges.len() {
            let (name_a, off_a, len_a, write_a) = ranges[i];
            let (name_b, off_b, len_b, write_b) = ranges[j];
            if !write_a && !write_b {
                // Two read-only views of the same/overlapping memory are
                // safe to alias; not the hazard this check guards against.
                continue;
            }
            let disjoint = off_a + len_a <= off_b || off_b + len_b <= off_a;
            debug_assert!(
                disjoint,
                "cpu_backend: arena ranges for `{name_a}` [{off_a}, {end_a}) and `{name_b}` \
                 [{off_b}, {end_b}) overlap while at least one is mutably aliased — this \
                 would violate the raw_parts safety invariant (Arena::plan must give \
                 overlapping-lifetime tensors disjoint memory; see the safety comment on \
                 `raw_parts` in cpu_backend.rs and `Arena::plan` in arena.rs)",
                end_a = off_a + len_a,
                end_b = off_b + len_b,
            );
        }
    }
}

impl Backend for CpuBackend {
    fn execute(&self, op: &Op, arena: &mut Arena, _weights: &Weights) {
        match *op {
            Op::Gemm { a, b, out, m, n, k } => {
                #[cfg(debug_assertions)]
                assert_disjoint_or_shared_read(&[
                    ("a", arena.range(a).0, arena.range(a).1, false),
                    ("b", arena.range(b).0, arena.range(b).1, false),
                    ("out", arena.range(out).0, arena.range(out).1, true),
                ]);
                let (a_ptr, a_len) = raw_parts(arena, a);
                let (b_ptr, b_len) = raw_parts(arena, b);
                let (out_ptr, out_len) = raw_parts(arena, out);
                // SAFETY: see `raw_parts` doc comment — a, b, out all have
                // overlapping lifetime at this step, so they're disjoint
                // (verified above by `assert_disjoint_or_shared_read` in
                // debug builds).
                let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, a_len) };
                let b_slice = unsafe { std::slice::from_raw_parts(b_ptr, b_len) };
                let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, out_len) };
                self.gemm.gemm(m, n, k, a_slice, b_slice, out_slice);
            }
            Op::AddBias { x, bias, rows, cols } => {
                #[cfg(debug_assertions)]
                assert_disjoint_or_shared_read(&[
                    ("x", arena.range(x).0, arena.range(x).1, true),
                    ("bias", arena.range(bias).0, arena.range(bias).1, false),
                ]);
                let (x_ptr, x_len) = raw_parts(arena, x);
                let (bias_ptr, bias_len) = raw_parts(arena, bias);
                // SAFETY: see `raw_parts` doc comment (verified above by
                // `assert_disjoint_or_shared_read` in debug builds).
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
                #[cfg(debug_assertions)]
                assert_disjoint_or_shared_read(&[
                    ("x", arena.range(x).0, arena.range(x).1, true),
                    ("g", arena.range(g).0, arena.range(g).1, false),
                    ("b", arena.range(b).0, arena.range(b).1, false),
                ]);
                let (x_ptr, x_len) = raw_parts(arena, x);
                let (g_ptr, g_len) = raw_parts(arena, g);
                let (b_ptr, b_len) = raw_parts(arena, b);
                // SAFETY: see `raw_parts` doc comment (verified above by
                // `assert_disjoint_or_shared_read` in debug builds).
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
                #[cfg(debug_assertions)]
                assert_disjoint_or_shared_read(&[
                    ("q", arena.range(q).0, arena.range(q).1, false),
                    ("k", arena.range(k).0, arena.range(k).1, false),
                    ("v", arena.range(v).0, arena.range(v).1, false),
                    ("out", arena.range(out).0, arena.range(out).1, true),
                ]);
                let (q_ptr, q_len) = raw_parts(arena, q);
                let (k_ptr, k_len) = raw_parts(arena, k);
                let (v_ptr, v_len) = raw_parts(arena, v);
                let (out_ptr, out_len) = raw_parts(arena, out);
                // SAFETY: see `raw_parts` doc comment (verified above by
                // `assert_disjoint_or_shared_read` in debug builds).
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
                #[cfg(debug_assertions)]
                {
                    let mut ranges = vec![
                        ("input", arena.range(input).0, arena.range(input).1, false),
                        ("weight", arena.range(weight).0, arena.range(weight).1, false),
                        ("out", arena.range(out).0, arena.range(out).1, true),
                    ];
                    if let Some(id) = bias {
                        let (off, len) = arena.range(id);
                        ranges.push(("bias", off, len, false));
                    }
                    assert_disjoint_or_shared_read(&ranges);
                }
                let (input_ptr, input_len) = raw_parts(arena, input);
                let (weight_ptr, weight_len) = raw_parts(arena, weight);
                let bias_raw = bias.map(|id| raw_parts(arena, id));
                let (out_ptr, out_len) = raw_parts(arena, out);
                // SAFETY: see `raw_parts` doc comment (verified above by
                // `assert_disjoint_or_shared_read` in debug builds).
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
