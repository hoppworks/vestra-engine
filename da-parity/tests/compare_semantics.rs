use vestra_parity::compare;

#[test]
fn within_tolerance_passes() {
    let got = [1.0f32, 2.0, 3.0];
    let refr = [1.001f32, 1.999, 3.0];
    let r = compare(&got, &refr, 2e-3, 2e-3, "unit");
    assert!(r.ok, "should pass: max_abs={}", r.max_abs);
}

#[test]
fn beyond_tolerance_fails_and_reports_worst() {
    let got = [1.0f32, 2.0, 5.0];
    let refr = [1.0f32, 2.0, 3.0];
    let r = compare(&got, &refr, 2e-3, 2e-3, "unit");
    assert!(!r.ok);
    assert_eq!(r.worst, 2);
    assert!((r.max_abs - 2.0).abs() < 1e-9);
}

#[test]
fn empty_is_never_a_pass() {
    let r = compare(&[], &[], 1.0, 1.0, "empty");
    assert!(!r.ok);
}
