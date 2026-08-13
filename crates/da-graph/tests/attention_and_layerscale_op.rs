use da_graph::{CpuBackend, Graph, RopeParams, Weights};

/// `Op::LayerScale` (Task 17's additive graph-op) must match
/// `da_kernels::scalar::layerscale` called directly on the same data.
#[test]
fn layer_scale_op_matches_manual_scalar_call() {
    let rows = 2usize;
    let cols = 3usize;
    let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let gamma: Vec<f32> = vec![10.0, -1.0, 0.5];

    let mut b = Graph::builder();
    let x_id = b.input(rows * cols);
    let g_id = b.weight("gamma", cols);
    let y = b.layer_scale(x_id, g_id, rows, cols);
    b.output(y);
    let graph = b.build();
    let plan = graph.compile();
    let backend = CpuBackend::new();

    let mut weights = Weights::new();
    weights.insert_f32("gamma", gamma.clone());

    let out = plan.run(&backend, &[&x], &weights);
    let graph_y = &out[0];

    let mut manual = x.clone();
    da_kernels::scalar::layerscale(&mut manual, rows, cols, &gamma);

    assert_eq!(graph_y, &manual);
}

/// With `qnorm = None, knorm = None, rope = None`, `Op::Attention` must
/// degrade to exactly `da_kernels::attention::attention` on the same q/k/v —
/// i.e. the Task 17 revision to the op is purely additive for callers that
/// don't opt into the new fields (this task's own "plain attention still
/// works" regression coverage, dump-independent).
#[test]
fn attention_op_without_qknorm_or_rope_matches_plain_attention() {
    let heads = 2usize;
    let n = 3usize;
    let head_dim = 4usize;
    let len = heads * n * head_dim;

    let mut rng = Xorshift32(0xA5A5_1234);
    let q: Vec<f32> = random_vec(&mut rng, len);
    let k: Vec<f32> = random_vec(&mut rng, len);
    let v: Vec<f32> = random_vec(&mut rng, len);

    let mut b = Graph::builder();
    let q_id = b.input(len);
    let k_id = b.input(len);
    let v_id = b.input(len);
    let out_id = b.attention(q_id, k_id, v_id, heads, n, head_dim);
    b.output(out_id);
    let graph = b.build();
    let plan = graph.compile();
    let backend = CpuBackend::new();
    let weights = Weights::new();

    let out = plan.run(&backend, &[&q, &k, &v], &weights);
    let graph_out = &out[0];

    let mut manual = vec![0f32; len];
    da_kernels::attention(&q, &k, &v, heads, n, head_dim, &mut manual);

    assert_eq!(graph_out.len(), manual.len());
    let max_diff = graph_out
        .iter()
        .zip(manual.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff < 1e-6, "max|d| = {max_diff}");
}

/// Trap #1 regression test: the per-head QK-LayerNorm inside `Op::Attention`
/// must use `qk_norm_eps` (the field baked into the op), which for the real
/// model is `1e-5` — a *different* value from the block-level `ln_eps`
/// (`1e-6`). This constructs data specifically chosen so that swapping in
/// the wrong eps produces a detectably different attention output, then
/// asserts our graph path (built with `qk_norm_eps = 1e-5`) matches a manual
/// reference that also uses `1e-5`, and does NOT match a manual reference
/// that (wrongly) uses `1e-6` — closing this as a real regression test, not
/// just documentation.
///
/// Also exercises trap #2 (normalizing over `head_dim`, not the full
/// `embed_dim`): `gamma`/`beta` here are length `head_dim`, and the qnorm is
/// applied per `(head, token)` row of that length — the graph path and the
/// manual reference both compute it this way, and disagreement with either
/// wrong-eps OR wrong-axis reference would show up as a max-diff failure.
#[test]
fn attention_op_qk_norm_uses_qk_norm_eps_not_ln_eps() {
    let heads = 2usize;
    let n = 2usize;
    let head_dim = 4usize;
    let len = heads * n * head_dim;

    // Near-constant-variance rows (tiny synthetic spread) so that a
    // 1e-6 vs 1e-5 eps difference is actually visible in the normalized
    // output rather than being swamped by a large variance.
    let mut rng = Xorshift32(0xC0DE_F00D);
    let mut q = vec![0f32; len];
    let mut k = vec![0f32; len];
    for row in 0..(heads * n) {
        let base = 100.0 + row as f32;
        for d in 0..head_dim {
            q[row * head_dim + d] = base + (rng.next_f32() * 1e-3);
            k[row * head_dim + d] = base - (rng.next_f32() * 1e-3);
        }
    }
    let v: Vec<f32> = random_vec(&mut rng, len);

    let qn_gamma = vec![1.0f32; head_dim];
    let qn_beta = vec![0.0f32; head_dim];
    let kn_gamma = vec![1.0f32; head_dim];
    let kn_beta = vec![0.0f32; head_dim];

    const QK_NORM_EPS: f32 = 1e-5;
    const WRONG_LN_EPS: f32 = 1e-6;

    let mut b = Graph::builder();
    let q_id = b.input(len);
    let k_id = b.input(len);
    let v_id = b.input(len);
    let qng_id = b.weight("qn_g", head_dim);
    let qnb_id = b.weight("qn_b", head_dim);
    let kng_id = b.weight("kn_g", head_dim);
    let knb_id = b.weight("kn_b", head_dim);
    let out_id = b.attention_full(
        q_id,
        k_id,
        v_id,
        heads,
        n,
        head_dim,
        Some((qng_id, qnb_id)),
        Some((kng_id, knb_id)),
        QK_NORM_EPS,
        None,
    );
    b.output(out_id);
    let graph = b.build();
    let plan = graph.compile();
    let backend = CpuBackend::new();

    let mut weights = Weights::new();
    weights.insert_f32("qn_g", qn_gamma.clone());
    weights.insert_f32("qn_b", qn_beta.clone());
    weights.insert_f32("kn_g", kn_gamma.clone());
    weights.insert_f32("kn_b", kn_beta.clone());

    let out = plan.run(&backend, &[&q, &k, &v], &weights);
    let graph_out = out[0].clone();

    // Manual reference using the CORRECT eps (1e-5).
    let manual_correct = manual_qk_norm_attention(
        &q,
        &k,
        &v,
        heads,
        n,
        head_dim,
        &qn_gamma,
        &qn_beta,
        &kn_gamma,
        &kn_beta,
        QK_NORM_EPS,
    );
    // Manual reference using the WRONG eps (block ln_eps, 1e-6).
    let manual_wrong = manual_qk_norm_attention(
        &q,
        &k,
        &v,
        heads,
        n,
        head_dim,
        &qn_gamma,
        &qn_beta,
        &kn_gamma,
        &kn_beta,
        WRONG_LN_EPS,
    );

    let diff_correct = max_abs_diff(&graph_out, &manual_correct);
    let diff_wrong = max_abs_diff(&graph_out, &manual_wrong);

    assert!(
        diff_correct < 1e-6,
        "graph output should match the 1e-5-eps reference: max|d|={diff_correct}"
    );
    assert!(
        diff_wrong > 1e-8,
        "graph output should NOT match a reference using the wrong (ln_eps=1e-6) epsilon — \
         if this fails, the op is silently using the wrong eps (trap #1 regressed): max|d|={diff_wrong}"
    );
}

/// Sanity check that `RopeParams`/the `rope` field at least wires through
/// and changes the output relative to no-RoPE (full numeric parity against
/// the C++ reference is unverified — see `da_kernels::rope`'s module doc —
/// this only proves the graph plumbing actually applies it).
#[test]
fn attention_op_rope_changes_output_relative_to_no_rope() {
    let heads = 1usize;
    let n = 2usize;
    let head_dim = 4usize; // must be a multiple of 4 for rope2d
    let len = heads * n * head_dim;

    let mut rng = Xorshift32(0x1234_5678);
    let q: Vec<f32> = random_vec(&mut rng, len);
    let k: Vec<f32> = random_vec(&mut rng, len);
    let v: Vec<f32> = random_vec(&mut rng, len);
    // Distinct (y,x) positions per token so rotation is non-trivial.
    let pos_yx_f32: Vec<f32> = vec![0.0, 0.0, 1.0, 2.0];

    let backend = CpuBackend::new();
    let weights = Weights::new();

    // No RoPE.
    let mut b1 = Graph::builder();
    let q1 = b1.input(len);
    let k1 = b1.input(len);
    let v1 = b1.input(len);
    let out1 = b1.attention(q1, k1, v1, heads, n, head_dim);
    b1.output(out1);
    let plan1 = b1.build().compile();
    let no_rope = plan1.run(&backend, &[&q, &k, &v], &weights)[0].clone();

    // With RoPE.
    let mut b2 = Graph::builder();
    let q2 = b2.input(len);
    let k2 = b2.input(len);
    let v2 = b2.input(len);
    let pos2 = b2.input(n * 2);
    let out2 = b2.attention_full(
        q2,
        k2,
        v2,
        heads,
        n,
        head_dim,
        None,
        None,
        1e-5,
        Some(RopeParams {
            pos_yx: pos2,
            freq: 100.0,
        }),
    );
    b2.output(out2);
    let plan2 = b2.build().compile();
    let with_rope = plan2.run(&backend, &[&q, &k, &v, &pos_yx_f32], &weights)[0].clone();

    assert_eq!(no_rope.len(), with_rope.len());
    let diff = max_abs_diff(&no_rope, &with_rope);
    assert!(
        diff > 1e-4,
        "RoPE should change attention output: max|d|={diff}"
    );
}

#[allow(clippy::too_many_arguments)]
fn manual_qk_norm_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    heads: usize,
    n: usize,
    head_dim: usize,
    qn_g: &[f32],
    qn_b: &[f32],
    kn_g: &[f32],
    kn_b: &[f32],
    eps: f32,
) -> Vec<f32> {
    let mut qc = q.to_vec();
    let mut kc = k.to_vec();
    da_kernels::scalar::layernorm(&mut qc, heads * n, head_dim, qn_g, qn_b, eps);
    da_kernels::scalar::layernorm(&mut kc, heads * n, head_dim, kn_g, kn_b, eps);
    let mut out = vec![0f32; heads * n * head_dim];
    da_kernels::attention(&qc, &kc, v, heads, n, head_dim, &mut out);
    out
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Deterministic, dependency-free PRNG (Xorshift32) for reproducible test data.
struct Xorshift32(u32);
impl Xorshift32 {
    fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        ((x as f32) / (u32::MAX as f32)) * 2.0 - 1.0
    }
}
fn random_vec(rng: &mut Xorshift32, n: usize) -> Vec<f32> {
    (0..n).map(|_| rng.next_f32()).collect()
}
