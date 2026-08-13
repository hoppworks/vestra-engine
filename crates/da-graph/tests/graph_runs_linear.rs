use vestra_graph::{CpuBackend, Graph, Weights};

/// y = gelu(x @ W + b), computed once through a `Graph`/`Plan`/`CpuBackend`
/// and once by calling the `vestra_kernels` scalar functions directly on the
/// same data; the two must match within float tolerance.
#[test]
fn linear_gelu_graph_matches_manual() {
    let m = 2usize;
    let k = 3usize;
    let n = 2usize;

    let x: Vec<f32> = vec![0.1, 0.2, 0.3, -0.4, 0.5, -0.6];
    let w: Vec<f32> = vec![0.7, -0.1, 0.2, 0.4, -0.3, 0.9];
    let bias: Vec<f32> = vec![0.05, -0.02];

    // --- graph path ---
    let mut b = Graph::builder();
    let x_id = b.input(m * k);
    let w_id = b.weight("w", k * n);
    let bias_id = b.weight("b", n);
    let gemm_out = b.gemm(x_id, w_id, m, n, k);
    b.add_bias(gemm_out, bias_id, m, n);
    let y = b.gelu(gemm_out);
    b.output(y);
    let graph = b.build();

    let plan = graph.compile();
    let backend = CpuBackend::new();

    let mut weights = Weights::new();
    weights.insert_f32("w", w.clone());
    weights.insert_f32("b", bias.clone());

    let outputs = plan.run(&backend, &[&x], &weights);
    assert_eq!(outputs.len(), 1);
    let graph_y = &outputs[0];

    // --- manual path: same math, straight from vestra_kernels ---
    let mut manual = vec![0.0f32; m * n];
    vestra_kernels::scalar::gemm_f32(m, n, k, &x, &w, &mut manual);
    vestra_kernels::scalar::add_bias_rows(&mut manual, m, n, &bias);
    vestra_kernels::scalar::gelu(&mut manual);

    assert_eq!(graph_y.len(), manual.len());
    let max_diff = graph_y
        .iter()
        .zip(manual.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff < 1e-5, "max|d| = {max_diff}");
}

/// Running the same compiled `Plan` twice must produce identical results —
/// this is the brief's specified proxy for "zero forward allocations": if
/// `run` allocated fresh, un-reused activation storage on the second call,
/// nothing here would prove the arena is actually being reused, but the
/// result equality (and, at the code-review level, `Plan::run`'s
/// implementation never calling anything but `Arena::buf`/`copy_from_slice`
/// in its hot loop) together support the invariant.
#[test]
fn run_twice_on_same_plan_is_deterministic() {
    let m = 2usize;
    let k = 3usize;
    let n = 2usize;

    let x: Vec<f32> = vec![0.1, 0.2, 0.3, -0.4, 0.5, -0.6];
    let w: Vec<f32> = vec![0.7, -0.1, 0.2, 0.4, -0.3, 0.9];
    let bias: Vec<f32> = vec![0.05, -0.02];

    let mut b = Graph::builder();
    let x_id = b.input(m * k);
    let w_id = b.weight("w", k * n);
    let bias_id = b.weight("b", n);
    let gemm_out = b.gemm(x_id, w_id, m, n, k);
    b.add_bias(gemm_out, bias_id, m, n);
    let y = b.gelu(gemm_out);
    b.output(y);
    let graph = b.build();

    let plan = graph.compile();
    let backend = CpuBackend::new();

    let mut weights = Weights::new();
    weights.insert_f32("w", w);
    weights.insert_f32("b", bias);

    let first = plan.run(&backend, &[&x], &weights);
    let second = plan.run(&backend, &[&x], &weights);
    assert_eq!(first, second);
}
