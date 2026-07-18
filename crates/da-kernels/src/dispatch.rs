/// Detected/selected instruction set architecture for the vectorized kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Isa {
    Avx512,
    Avx2,
    Scalar,
}

/// Runtime kernel dispatcher. Detects the host ISA once at construction
/// (`Kernels::detect()`) and routes each kernel call to the fastest
/// implementation available on this machine, falling back to the scalar
/// reference kernels (Task 5) everywhere else — including non-x86_64 hosts.
pub struct Kernels {
    isa: Isa,
}

impl Kernels {
    /// Detect the best available ISA on this host. On non-x86_64 targets
    /// (e.g. this development machine, aarch64/Apple Silicon) this always
    /// returns `Isa::Scalar`, since `is_x86_feature_detected!` and the
    /// AVX-512/AVX2 kernels only exist on `target_arch = "x86_64"`.
    pub fn detect() -> Kernels {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx512f") {
                return Kernels { isa: Isa::Avx512 };
            }
            if std::is_x86_feature_detected!("avx2") {
                return Kernels { isa: Isa::Avx2 };
            }
        }
        Kernels { isa: Isa::Scalar }
    }

    pub fn isa(&self) -> Isa {
        self.isa
    }

    /// In-place GELU. Result must be within the tolerance band of the
    /// scalar reference kernel (`scalar::gelu`).
    pub fn gelu(&self, x: &mut [f32]) {
        match self.isa {
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512 => unsafe { crate::simd_avx512::gelu_avx512(x) },
            _ => crate::scalar::gelu(x),
        }
    }

    /// In-place elementwise `dst += src`. Result must exactly match the
    /// scalar reference kernel (`scalar::add`).
    pub fn add(&self, dst: &mut [f32], src: &[f32]) {
        match self.isa {
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512 => unsafe { crate::simd_avx512::add_avx512(dst, src) },
            _ => crate::scalar::add(dst, src),
        }
    }
}
