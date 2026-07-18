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

/// q8_0 x q8_0 -> f32 GEMM: A is `m x k` and B is `n x k`, both stored as
/// row-major q8_0 blocks (`k/QK8_0` blocks per row; B's rows are the *n*
/// columns of the logical `k x n` matrix, i.e. B is pre-transposed into
/// row blocks the same way A is). This is the scalar oracle: every other
/// backend (AVX-512/VNNI) must match this within the test's tolerance band.
pub fn gemm_q8_0(
    m: usize,
    n: usize,
    k: usize,
    a_q: &[da_gguf::BlockQ8_0],
    b_q: &[da_gguf::BlockQ8_0],
    c: &mut [f32],
) {
    debug_assert_eq!(k % da_gguf::QK8_0, 0, "k must be a multiple of QK8_0");
    let blocks_per_row = k / da_gguf::QK8_0;
    debug_assert_eq!(a_q.len(), m * blocks_per_row);
    debug_assert_eq!(b_q.len(), n * blocks_per_row);
    debug_assert_eq!(c.len(), m * n);

    for i in 0..m {
        let a_row = &a_q[i * blocks_per_row..(i + 1) * blocks_per_row];
        for j in 0..n {
            let b_row = &b_q[j * blocks_per_row..(j + 1) * blocks_per_row];
            let mut acc = 0f32;
            for bi in 0..blocks_per_row {
                let ab = &a_row[bi];
                let bb = &b_row[bi];
                let mut isum: i32 = 0;
                for l in 0..da_gguf::QK8_0 {
                    isum += ab.qs[l] as i32 * bb.qs[l] as i32;
                }
                acc += ab.d.to_f32() * bb.d.to_f32() * isum as f32;
            }
            c[i * n + j] = acc;
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
    const INV_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2;
    for v in x.iter_mut() { *v = 0.5 * *v * (1.0 + erf(*v * INV_SQRT2)); }
}

// Abramowitz–Stegun 7.1.26 erf-Approximation (|error| < 1.5e-7).
fn erf(x: f32) -> f32 {
    let s = x.signum(); let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0 - (((((1.061_405_4*t - 1.453_152_0)*t) + 1.421_413_7)*t - 0.284_496_74)*t + 0.254_829_59)*t * (-x*x).exp();
    s * y
}

/// In-place per-column ("LayerScale") scale: `x[r,c] *= gamma[c]`. Mirrors
/// `add_bias_rows`'s shape convention exactly (row-major `[rows, cols]`,
/// one scale factor per column, broadcast over rows).
pub fn layerscale(x: &mut [f32], rows: usize, cols: usize, gamma: &[f32]) {
    debug_assert_eq!(x.len(), rows * cols);
    debug_assert_eq!(gamma.len(), cols);
    for r in 0..rows {
        for c in 0..cols {
            x[r * cols + c] *= gamma[c];
        }
    }
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
