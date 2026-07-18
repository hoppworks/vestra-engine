use criterion::{black_box, criterion_group, criterion_main, Criterion};
use da_kernels::scalar::gemm_f32;

fn gemm_bench(c: &mut Criterion) {
    c.bench_function("gemm_f32_64x64", |b| {
        let m = 64;
        let n = 64;
        let k = 64;
        let a = vec![1.0; m * k];
        let b = vec![1.0; k * n];
        let mut c_out = vec![0.0; m * n];
        b.iter(|| {
            gemm_f32(black_box(m), black_box(n), black_box(k), black_box(&a), black_box(&b), black_box(&mut c_out));
        })
    });
}

criterion_group!(benches, gemm_bench);
criterion_main!(benches);
