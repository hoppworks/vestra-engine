pub mod gemm;
pub mod scalar;

mod dispatch;
#[cfg(target_arch = "x86_64")]
mod simd_avx512;

pub use dispatch::{Isa, Kernels};
