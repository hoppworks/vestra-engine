pub mod gemm;
pub mod q8_0_dot;
pub mod scalar;

mod dispatch;
#[cfg(target_arch = "x86_64")]
mod simd_avx512;

pub use dispatch::{Isa, Kernels};
pub use q8_0_dot::quantize_row_q8_0;
