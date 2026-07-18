//! Scaled-dot-product attention: a naive gemm+softmax+gemm oracle and a
//! tiled, online-softmax ("flash-attention"-style) implementation that must
//! match it bit-for-bit within floating point tolerance.

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

            // scores[j] = <q_i, k_j> * scale
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

            // softmax
            let mut sum = 0f32;
            for j in 0..n {
                let e = (scores[j] - max_score).exp();
                scores[j] = e;
                sum += e;
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

/// Key/value tile size for the online-softmax pass.
const KV_TILE: usize = 64;

/// Tiled, online-softmax scaled-dot-product attention. Mathematically
/// equivalent to [`attention_naive`], processing keys/values in blocks and
/// maintaining a running max/sum (flash-attention-style) instead of
/// materializing the full `[n, n]` score matrix.
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
    let mut acc = vec![0f32; head_dim];

    for h in 0..heads {
        let qh = &q[h * n * head_dim..(h + 1) * n * head_dim];
        let kh = &k[h * n * head_dim..(h + 1) * n * head_dim];
        let vh = &v[h * n * head_dim..(h + 1) * n * head_dim];
        let oh = &mut out[h * n * head_dim..(h + 1) * n * head_dim];

        for i in 0..n {
            let qi = &qh[i * head_dim..(i + 1) * head_dim];

            let mut running_max = f32::NEG_INFINITY;
            let mut running_sum = 0f32;
            acc.iter_mut().for_each(|a| *a = 0.0);

            let mut j0 = 0usize;
            while j0 < n {
                let j1 = (j0 + KV_TILE).min(n);

                // Local scores for this key/value tile.
                let mut local_scores = [0f32; KV_TILE];
                let mut tile_max = f32::NEG_INFINITY;
                for (t, j) in (j0..j1).enumerate() {
                    let kj = &kh[j * head_dim..(j + 1) * head_dim];
                    let dot: f32 = qi.iter().zip(kj.iter()).map(|(a, b)| a * b).sum();
                    let s = dot * scale;
                    local_scores[t] = s;
                    if s > tile_max {
                        tile_max = s;
                    }
                }

                let new_max = running_max.max(tile_max);
                let correction = if running_max.is_finite() {
                    (running_max - new_max).exp()
                } else {
                    0.0
                };

                // Rescale existing accumulator/sum to the new running max.
                running_sum *= correction;
                for a in acc.iter_mut() {
                    *a *= correction;
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
            let oi = &mut oh[i * head_dim..(i + 1) * head_dim];
            for d in 0..head_dim {
                oi[d] = acc[d] * inv_sum;
            }
        }
    }
}
