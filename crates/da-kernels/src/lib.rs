pub mod attention;
pub mod gemm;
pub mod q8_0_dot;
pub mod rope;
pub mod scalar;

mod dispatch;
#[cfg(target_arch = "x86_64")]
mod simd_avx512;

pub use attention::{attention, attention_naive};
pub use dispatch::{Isa, Kernels};
pub use q8_0_dot::quantize_row_q8_0;
pub use rope::rope2d;
