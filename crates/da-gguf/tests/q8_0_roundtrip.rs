use half::f16;
use vestra_gguf::{dequantize_q8_0, BlockQ8_0, QK8_0};

#[test]
fn dequant_matches_scale_times_qs() {
    // Ein Block mit bekanntem Scale d und Quanten qs -> Werte = d * qs.
    let d = 0.5f32;
    let mut qs = [0i8; 32];
    for (i, value) in qs.iter_mut().enumerate() {
        *value = (i as i8) - 16;
    } // -16..15
    let blk = BlockQ8_0 {
        d: f16::from_f32(d),
        qs,
    };
    let mut out = vec![0f32; QK8_0];
    dequantize_q8_0(std::slice::from_ref(&blk), &mut out);
    for (i, actual) in out.iter().enumerate() {
        let expected = d * ((i as i32 - 16) as f32);
        assert!(
            (*actual - expected).abs() < 1e-3,
            "i={i} got={} exp={}",
            actual,
            expected
        );
    }
}
