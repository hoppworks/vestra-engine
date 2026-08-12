//! Scaled-dot-product attention: a naive GEMM+softmax+GEMM oracle and a
//! GEMM-accelerated production implementation plus a tiled online-softmax
//! reference used for validation.

/// Naive reference: per head, `softmax(Q @ K^T / sqrt(head_dim)) @ V`.
///
/// `q`, `k`, `v` are `[heads, n, head_dim]` row-major; `out` is written in
/// the same layout.
pub fn attention_naive(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    heads: usize,
    n: usize,
    head_dim: usize,
    out: &mut [f32],
) {
    assert_eq!(q.len(), heads * n * head_dim);
    assert_eq!(k.len(), heads * n * head_dim);
    assert_eq!(v.len(), heads * n * head_dim);
    assert_eq!(out.len(), heads * n * head_dim);

    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut scores = vec![0f32; n];
    for h in 0..heads {
        let qh = &q[h * n * head_dim..(h + 1) * n * head_dim];
        let kh = &k[h * n * head_dim..(h + 1) * n * head_dim];
        let vh = &v[h * n * head_dim..(h + 1) * n * head_dim];
        let oh = &mut out[h * n * head_dim..(h + 1) * n * head_dim];
        for i in 0..n {
            let qi = &qh[i * head_dim..(i + 1) * head_dim];
            let mut max_score = f32::NEG_INFINITY;
            for j in 0..n {
                let kj = &kh[j * head_dim..(j + 1) * head_dim];
                let dot: f32 = qi.iter().zip(kj.iter()).map(|(a, b)| a * b).sum();
                let s = dot * scale;
                scores[j] = s;
                if s > max_score {
                    max_score = s;
                }
            }
            let mut sum = 0f32;
            for value in &mut scores {
                *value = (*value - max_score).exp();
                sum += *value;
            }
            let inv_sum = 1.0f32 / sum;
            let oi = &mut oh[i * head_dim..(i + 1) * head_dim];
            oi.fill(0.0);
            for j in 0..n {
                let w = scores[j] * inv_sum;
                let vj = &vh[j * head_dim..(j + 1) * head_dim];
                for d in 0..head_dim {
                    oi[d] += w * vj[d];
                }
            }
        }
    }
}

#[cfg(test)]
const KV_TILE: usize = 64;

/// Tiled online-softmax attention.  Every `(head, query)` row is independent,
/// so Rayon distributes rows while preserving each row's exact K traversal and
/// F32 accumulation order.
pub fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    heads: usize,
    n: usize,
    head_dim: usize,
    out: &mut [f32],
) {
    assert_eq!(q.len(), heads * n * head_dim);
    assert_eq!(k.len(), heads * n * head_dim);
    assert_eq!(v.len(), heads * n * head_dim);
    assert_eq!(out.len(), heads * n * head_dim);

    let scale = 1.0f32 / (head_dim as f32).sqrt();
    for h in 0..heads {
        let base = h * n * head_dim;
        let qh = &q[base..base + n * head_dim];
        let kh = &k[base..base + n * head_dim];
        let vh = &v[base..base + n * head_dim];
        let oh = &mut out[base..base + n * head_dim];
        let mut scores = vec![0.0; n * n];
        matmul_qk_transposed(qh, kh, n, head_dim, &mut scores);
        for score in &mut scores {
            *score *= scale;
        }
        da_kernels_softmax_rows(&mut scores, n);
        matmul(scores.as_slice(), vh, n, head_dim, n, oh);
    }
}

fn matmul_qk_transposed(q: &[f32], k: &[f32], n: usize, head_dim: usize, out: &mut [f32]) {
    let q =
        unsafe { faer::mat::from_raw_parts::<f32>(q.as_ptr(), n, head_dim, head_dim as isize, 1) };
    // K is row-major `[n, head_dim]`; this is its `[head_dim, n]` transpose
    // view, not an allocation or layout conversion.
    let kt =
        unsafe { faer::mat::from_raw_parts::<f32>(k.as_ptr(), head_dim, n, 1, head_dim as isize) };
    let out =
        unsafe { faer::mat::from_raw_parts_mut::<f32>(out.as_mut_ptr(), n, n, n as isize, 1) };
    faer::linalg::matmul::matmul(out, q, kt, None, 1.0, faer::get_global_parallelism());
}

fn matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, out: &mut [f32]) {
    let a = unsafe { faer::mat::from_raw_parts::<f32>(a.as_ptr(), m, k, k as isize, 1) };
    let b = unsafe { faer::mat::from_raw_parts::<f32>(b.as_ptr(), k, n, n as isize, 1) };
    let out =
        unsafe { faer::mat::from_raw_parts_mut::<f32>(out.as_mut_ptr(), m, n, n as isize, 1) };
    faer::linalg::matmul::matmul(out, a, b, None, 1.0, faer::get_global_parallelism());
}

fn da_kernels_softmax_rows(values: &mut [f32], cols: usize) {
    for row in values.chunks_exact_mut(cols) {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for value in row.iter_mut() {
            *value = (*value - max).exp();
            sum += *value;
        }
        let inv_sum = 1.0 / sum;
        for value in row.iter_mut() {
            *value *= inv_sum;
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
fn attention_row(
    qh: &[f32],
    kh: &[f32],
    vh: &[f32],
    i: usize,
    n: usize,
    head_dim: usize,
    scale: f32,
    acc: &mut [f32],
    oi: &mut [f32],
) {
    let qi = &qh[i * head_dim..(i + 1) * head_dim];
    let mut running_max = f32::NEG_INFINITY;
    let mut running_sum = 0f32;
    acc.fill(0.0);
    let mut j0 = 0usize;
    while j0 < n {
        let j1 = (j0 + KV_TILE).min(n);
        let mut local_scores = [0f32; KV_TILE];
        let mut tile_max = f32::NEG_INFINITY;
        for (t, j) in (j0..j1).enumerate() {
            let kj = &kh[j * head_dim..(j + 1) * head_dim];
            let dot: f32 = qi.iter().zip(kj.iter()).map(|(a, b)| a * b).sum();
            let score = dot * scale;
            local_scores[t] = score;
            if score > tile_max {
                tile_max = score;
            }
        }
        let new_max = running_max.max(tile_max);
        let correction = if running_max.is_finite() {
            (running_max - new_max).exp()
        } else {
            0.0
        };
        running_sum *= correction;
        for value in acc.iter_mut() {
            *value *= correction;
        }
        for (t, j) in (j0..j1).enumerate() {
            let p = (local_scores[t] - new_max).exp();
            running_sum += p;
            let vj = &vh[j * head_dim..(j + 1) * head_dim];
            for d in 0..head_dim {
                acc[d] += p * vj[d];
            }
        }
        running_max = new_max;
        j0 = j1;
    }
    let inv_sum = 1.0f32 / running_sum;
    for d in 0..head_dim {
        oi[d] = acc[d] * inv_sum;
    }
}

#[cfg(test)]
pub(crate) fn attention_serial(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    heads: usize,
    n: usize,
    head_dim: usize,
    out: &mut [f32],
) {
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut acc = vec![0.0; head_dim];
    for row in 0..heads * n {
        let h = row / n;
        let i = row % n;
        let base = h * n * head_dim;
        attention_row(
            &q[base..base + n * head_dim],
            &k[base..base + n * head_dim],
            &v[base..base + n * head_dim],
            i,
            n,
            head_dim,
            scale,
            &mut acc,
            &mut out[row * head_dim..(row + 1) * head_dim],
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(len: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn gemm_attention_matches_online_attention() {
        let (heads, n, dim) = (3, 71, 16);
        let q = values(heads * n * dim, 0xA771_0001);
        let k = values(heads * n * dim, 0xA771_0002);
        let v = values(heads * n * dim, 0xA771_0003);
        let mut parallel = vec![0.0; heads * n * dim];
        let mut serial = vec![0.0; heads * n * dim];
        attention(&q, &k, &v, heads, n, dim, &mut parallel);
        attention_serial(&q, &k, &v, heads, n, dim, &mut serial);
        for (gemm, online) in parallel.iter().zip(serial.iter()) {
            assert!((gemm - online).abs() < 1e-4, "gemm={gemm} online={online}");
        }
    }
}
