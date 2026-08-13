use da_graph::arena::Arena;

#[test]
fn disjoint_lifetimes_reuse_memory() {
    // t0 lebt [0,1], t1 lebt [2,3] -> dürfen denselben Offset teilen.
    let a = Arena::plan(&[100, 100], &[(0, 1), (2, 3)]);
    assert_eq!(
        a.total_floats(),
        100,
        "disjoint tensors should share buffer"
    );
}

#[test]
fn overlapping_lifetimes_get_separate_memory() {
    let a = Arena::plan(&[100, 100], &[(0, 3), (1, 2)]);
    assert_eq!(a.total_floats(), 200);
}
