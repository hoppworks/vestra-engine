pub mod attention;
pub mod conv;
pub mod gemm;
pub mod q8_0_dot;
pub mod resample;
pub mod rope;
pub mod scalar;

mod dispatch;
#[cfg(target_arch = "x86_64")]
mod simd_avx512;

pub use attention::{attention, attention_naive};
pub use conv::{conv2d, conv2d_naive, conv_transpose2d};
pub use dispatch::{Isa, Kernels};
pub use q8_0_dot::quantize_row_q8_0;
pub use resample::{bilinear_resize, bilinear_resize_align_corners, bilinear_resize_naive};
pub use rope::rope2d;

pub fn qkv_f32_da3_base(
    input: &[f32], weight: &[f32], bias: &[f32], q: &mut [f32], k: &mut [f32], v: &mut [f32],
) -> bool {
    da3_kernels::qkv_f32_da3_base(input, weight, bias, q, k, v)
}
