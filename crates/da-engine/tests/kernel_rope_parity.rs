use da_parity::{assert_parity, dumps_path, Dumps};
use vestra_kernels::rope2d;

#[test]
fn rope2d_matches_reference() {
    let (g, m) = (dumps_path("reference.gguf"), dumps_path("manifest.json"));
    if !g.exists() {
        eprintln!("[skip] no dumps");
        return;
    }
    let d = Dumps::open(&g, &m).unwrap();
    let rin = d.reference("rope_in").unwrap(); // (1,1,4,64) -> heads=1,n=4,head_dim=64
    let rpos = d.reference("rope_pos").unwrap(); // (1,4,2) y,x als f32
    let rout = d.reference("rope_out").unwrap();
    let pos: Vec<i64> = rpos.data.iter().map(|&v| v as i64).collect();
    let mut x = rin.data.clone();
    rope2d(&mut x, 1, 4, 64, &pos, 100.0);
    assert_parity(&x, &rout.data, d.atol(), d.rtol(), "rope2d");
}
