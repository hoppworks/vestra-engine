/// The shape of a tensor, expressed as a list of dimension extents.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape(pub Vec<usize>);

impl Shape {
    pub fn new(dims: impl Into<Vec<usize>>) -> Self {
        Shape(dims.into())
    }

    /// Total number of elements described by this shape (product of dims).
    /// A shape with no dimensions (a scalar) has `numel() == 1`.
    pub fn numel(&self) -> usize {
        self.0.iter().product()
    }
}

/// A stable identifier for a tensor within a graph. Corresponds to the
/// tensor's index in whatever ordered collection (e.g. `sizes`/`lifetimes`
/// slices passed to `Arena::plan`) defines the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TensorId(pub usize);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numel_multiplies_dims() {
        let s = Shape(vec![2, 3, 4]);
        assert_eq!(s.numel(), 24);
    }

    #[test]
    fn numel_scalar_is_one() {
        let s = Shape(vec![]);
        assert_eq!(s.numel(), 1);
    }
}
