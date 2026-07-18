use crate::scalar;

pub trait Gemm {
    fn gemm(&self, m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]);
}

pub struct ScalarGemm;
impl Gemm for ScalarGemm {
    fn gemm(&self, m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
        scalar::gemm_f32(m, n, k, a, b, c);
    }
}

pub struct FaerGemm;
impl Gemm for FaerGemm {
    fn gemm(&self, m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
        use faer::Parallelism;
        // row-major Slices als faer-Views mit expliziten Strides interpretieren.
        let a = unsafe { faer::mat::from_raw_parts::<f32>(a.as_ptr(), m, k, k as isize, 1) };
        let b = unsafe { faer::mat::from_raw_parts::<f32>(b.as_ptr(), k, n, n as isize, 1) };
        let cm = unsafe { faer::mat::from_raw_parts_mut::<f32>(c.as_mut_ptr(), m, n, n as isize, 1) };
        faer::linalg::matmul::matmul(cm, a, b, None, 1.0, Parallelism::None);
    }
}

pub struct GemmWithEpilogue<G: Gemm> { pub inner: G }
impl<G: Gemm> GemmWithEpilogue<G> {
    pub fn gemm_bias_gelu(&self, m: usize, n: usize, k: usize, a: &[f32], b: &[f32],
                          bias: Option<&[f32]>, gelu: bool, c: &mut [f32]) {
        self.inner.gemm(m, n, k, a, b, c);
        if let Some(bias) = bias { scalar::add_bias_rows(c, m, n, bias); }
        if gelu { scalar::gelu(c); }
    }
}
