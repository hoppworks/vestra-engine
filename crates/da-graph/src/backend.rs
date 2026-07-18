use std::cell::RefCell;

use crate::graph::{Graph, Op, Weights};
use crate::Arena;

/// Executes a single [`Op`] against a mutable [`Arena`] (where activation
/// tensors live) and an immutable [`Weights`] map (where named GGUF
/// weights live). Implementations dispatch each `Op` variant to the
/// matching `da_kernels` function.
pub trait Backend {
    fn execute(&self, op: &Op, arena: &mut Arena, weights: &Weights);
}

/// A compiled [`Graph`]: tensor lifetimes have been derived and a single
/// [`Arena`] has been planned and allocated once, up front. `run()` reuses
/// that same arena on every call — it only ever calls `Arena::buf()` to
/// get subslices of the one allocation made here in `compile()`, plus
/// `copy_from_slice` to fill inputs/weights. The only allocation left in
/// the hot path is the `Vec<Vec<f32>>` this function must return (mandated
/// by its signature) — there is no allocation of activation/intermediate
/// storage.
///
/// The arena is wrapped in a `RefCell` so `run(&self, ...)` (an immutable
/// method, per the required signature) can still mutate the shared arena
/// across repeated calls.
pub struct Plan {
    graph: Graph,
    arena: RefCell<Arena>,
}

impl Plan {
    pub(crate) fn new(graph: Graph) -> Plan {
        let lifetimes = graph.compute_lifetimes();
        let arena = Arena::plan(&graph.sizes, &lifetimes);
        Plan {
            graph,
            arena: RefCell::new(arena),
        }
    }

    /// Fill graph inputs/weights into the arena, run every op through
    /// `backend`, and read back the graph outputs.
    pub fn run(&self, backend: &dyn Backend, inputs: &[&[f32]], weights: &Weights) -> Vec<Vec<f32>> {
        let mut arena = self.arena.borrow_mut();

        assert_eq!(
            inputs.len(),
            self.graph.inputs.len(),
            "expected {} inputs, got {}",
            self.graph.inputs.len(),
            inputs.len()
        );
        for (&id, &data) in self.graph.inputs.iter().zip(inputs.iter()) {
            arena.buf(id).copy_from_slice(data);
        }
        for (&id, name) in &self.graph.weight_tensors {
            let data = weights
                .get_f32(name)
                .unwrap_or_else(|| panic!("Weights missing f32 entry {name:?}"));
            arena.buf(id).copy_from_slice(data);
        }

        for op in &self.graph.ops {
            backend.execute(op, &mut arena, weights);
        }

        self.graph
            .outputs
            .iter()
            .map(|&id| arena.buf(id).to_vec())
            .collect()
    }
}
