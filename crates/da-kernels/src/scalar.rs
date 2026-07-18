pub fn gemm_f32(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    debug_assert_eq!(a.len(), m*k); debug_assert_eq!(b.len(), k*n); debug_assert_eq!(c.len(), m*n);
    for i in 0..m {
        for j in 0..n { c[i*n+j] = 0.0; }
        for p in 0..k {
            let aip = a[i*k+p];
            for j in 0..n { c[i*n+j] += aip * b[p*n+j]; }
        }
    }
}

pub fn add(dst: &mut [f32], src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    for i in 0..dst.len() { dst[i] += src[i]; }
}

pub fn add_bias_rows(x: &mut [f32], rows: usize, cols: usize, bias: &[f32]) {
    debug_assert_eq!(x.len(), rows*cols); debug_assert_eq!(bias.len(), cols);
    for r in 0..rows { for c in 0..cols { x[r*cols+c] += bias[c]; } }
}

pub fn layernorm(x: &mut [f32], rows: usize, cols: usize, gamma: &[f32], beta: &[f32], eps: f32) {
    debug_assert_eq!(x.len(), rows*cols);
    for r in 0..rows {
        let row = &mut x[r*cols..(r+1)*cols];
        let mean = row.iter().sum::<f32>() / cols as f32;
        let var = row.iter().map(|v| { let d = v - mean; d*d }).sum::<f32>() / cols as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for c in 0..cols { row[c] = (row[c] - mean) * inv * gamma[c] + beta[c]; }
    }
}

pub fn gelu(x: &mut [f32]) {
    const INV_SQRT2: f32 = 0.707_106_78;
    for v in x.iter_mut() { *v = 0.5 * *v * (1.0 + erf(*v * INV_SQRT2)); }
}

// Abramowitz–Stegun 7.1.26 erf-Approximation (|error| < 1.5e-7).
fn erf(x: f32) -> f32 {
    let s = x.signum(); let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0 - (((((1.061_405_4*t - 1.453_152_0)*t) + 1.421_413_7)*t - 0.284_496_74)*t + 0.254_829_59)*t * (-x*x).exp();
    s * y
}

pub fn softmax_rows(x: &mut [f32], rows: usize, cols: usize) {
    debug_assert_eq!(x.len(), rows*cols);
    for r in 0..rows {
        let row = &mut x[r*cols..(r+1)*cols];
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for v in row.iter_mut() { *v = (*v - m).exp(); sum += *v; }
        let inv = 1.0 / sum;
        for v in row.iter_mut() { *v *= inv; }
    }
}
