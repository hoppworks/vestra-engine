use crate::tensor::TensorId;

/// A pre-planned, reused `f32` buffer arena.
///
/// `Arena::plan` computes, ahead of time, a byte-offset assignment for every
/// tensor such that tensors whose lifetimes ([`first_use`, `last_use`])
/// don't overlap can share the same backing memory. Once planned, the
/// arena's backing storage is allocated exactly once; `buf()` only ever
/// hands out subslices of that single allocation — the forward pass makes
/// zero additional allocations.
pub struct Arena {
    storage: Vec<f32>,
    /// offset (in f32 elements) into `storage` for each tensor, indexed by
    /// `TensorId`.
    offsets: Vec<usize>,
    /// size (in f32 elements) of each tensor, indexed by `TensorId`.
    sizes: Vec<usize>,
}

/// A free block of the arena available for reuse: `[offset, offset+size)`.
#[derive(Clone, Copy)]
struct FreeBlock {
    offset: usize,
    size: usize,
}

/// A block currently occupied by a tensor, held until its `last_use` passes.
#[derive(Clone, Copy)]
struct ActiveBlock {
    offset: usize,
    size: usize,
    last_use: usize,
}

impl Arena {
    /// Plan buffer offsets for a set of tensors given their sizes (in `f32`
    /// elements) and lifetimes (`(first_use, last_use)`, inclusive, in
    /// "step" units — e.g. graph execution order).
    ///
    /// Uses a greedy linear-scan interval allocator: tensors are processed
    /// in order of `first_use`; before placing a tensor, any active blocks
    /// whose `last_use` has already passed (i.e. is strictly less than the
    /// new tensor's `first_use`) are released back into a free list. The
    /// new tensor is then placed into the smallest free block that fits
    /// (best-fit), splitting off any leftover space back into the free
    /// list; if no free block fits, the arena is extended (bump-allocated
    /// at the current high-water mark).
    pub fn plan(sizes: &[usize], lifetimes: &[(usize, usize)]) -> Arena {
        assert_eq!(
            sizes.len(),
            lifetimes.len(),
            "sizes and lifetimes must have the same length"
        );

        let n = sizes.len();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| lifetimes[i].0);

        let mut offsets = vec![0usize; n];
        let mut free_blocks: Vec<FreeBlock> = Vec::new();
        let mut active: Vec<ActiveBlock> = Vec::new();
        let mut high_water: usize = 0;

        for &i in &order {
            let size = sizes[i];
            let (first_use, last_use) = lifetimes[i];

            // Release any active blocks whose lifetime has ended before
            // this tensor's first use back into the free list.
            let mut still_active = Vec::with_capacity(active.len());
            for block in active.drain(..) {
                if block.last_use < first_use {
                    free_blocks.push(FreeBlock {
                        offset: block.offset,
                        size: block.size,
                    });
                } else {
                    still_active.push(block);
                }
            }
            active = still_active;

            // Coalesce adjacent free blocks so larger tensors can reuse
            // space freed by several smaller ones.
            coalesce(&mut free_blocks);

            // Best-fit: smallest free block that's large enough.
            let mut best_idx: Option<usize> = None;
            for (idx, block) in free_blocks.iter().enumerate() {
                if block.size >= size {
                    match best_idx {
                        Some(b) if free_blocks[b].size <= block.size => {}
                        _ => best_idx = Some(idx),
                    }
                }
            }

            let offset = if let Some(idx) = best_idx {
                let block = free_blocks.remove(idx);
                if block.size > size {
                    free_blocks.push(FreeBlock {
                        offset: block.offset + size,
                        size: block.size - size,
                    });
                }
                block.offset
            } else {
                let offset = high_water;
                high_water += size;
                offset
            };

            offsets[i] = offset;
            active.push(ActiveBlock {
                offset,
                size,
                last_use,
            });
        }

        Arena {
            storage: vec![0.0; high_water],
            offsets,
            sizes: sizes.to_vec(),
        }
    }

    /// Total number of `f32` elements backing this arena.
    pub fn total_floats(&self) -> usize {
        self.storage.len()
    }

    /// Returns a mutable slice into the arena's backing storage for the
    /// given tensor. No allocation occurs here — this is a subslice of the
    /// single allocation made in `plan`.
    pub fn buf(&mut self, id: TensorId) -> &mut [f32] {
        let offset = self.offsets[id.0];
        let size = self.sizes[id.0];
        &mut self.storage[offset..offset + size]
    }

    /// Returns `(offset, size)` — in `f32` elements — of tensor `id`'s slot
    /// in the backing storage. `pub(crate)`-only: this exists purely so
    /// `cpu_backend`'s debug-only aliasing assertions can check that the
    /// unsafe pointer splits in `raw_parts` never touch overlapping ranges,
    /// without giving outside callers a way to reach into arena internals.
    /// Only called from `#[cfg(debug_assertions)]`-gated call sites, so it's
    /// unused (and would warn as dead code) in release builds.
    #[cfg_attr(not(debug_assertions), allow(dead_code))]
    pub(crate) fn range(&self, id: TensorId) -> (usize, usize) {
        (self.offsets[id.0], self.sizes[id.0])
    }
}

fn coalesce(blocks: &mut Vec<FreeBlock>) {
    if blocks.len() < 2 {
        return;
    }
    blocks.sort_by_key(|b| b.offset);
    let mut merged: Vec<FreeBlock> = Vec::with_capacity(blocks.len());
    for block in blocks.drain(..) {
        if let Some(last) = merged.last_mut() {
            if last.offset + last.size == block.offset {
                last.size += block.size;
                continue;
            }
        }
        merged.push(block);
    }
    *blocks = merged;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disjoint_lifetimes_reuse_memory() {
        let a = Arena::plan(&[100, 100], &[(0, 1), (2, 3)]);
        assert_eq!(a.total_floats(), 100);
    }

    #[test]
    fn overlapping_lifetimes_get_separate_memory() {
        let a = Arena::plan(&[100, 100], &[(0, 3), (1, 2)]);
        assert_eq!(a.total_floats(), 200);
    }

    #[test]
    fn three_tensors_two_disjoint_one_overlapping() {
        // t0 [0,1], t1 [2,3] disjoint with t0 -> share.
        // t2 [1,2] overlaps both t0's tail and t1's head -> needs its own space.
        let sizes = [50, 50, 50];
        let lifetimes = [(0, 1), (2, 3), (1, 2)];
        let a = Arena::plan(&sizes, &lifetimes);
        // t0 and t1 can share one 50-slot block; t2 overlaps the boundary
        // (t0 ends at 1, t2 starts at 1 -> overlap since last_use < first_use
        // is required to free) so it needs a second block.
        assert_eq!(a.total_floats(), 100);
    }

    #[test]
    fn buf_returns_correct_length_and_is_writable() {
        let mut a = Arena::plan(&[4, 4], &[(0, 1), (2, 3)]);
        let b0 = a.buf(TensorId(0));
        assert_eq!(b0.len(), 4);
        b0.copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let b1 = a.buf(TensorId(1));
        assert_eq!(b1.len(), 4);
        b1.copy_from_slice(&[5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn many_tensors_bound_peak_usage() {
        // A chain where each tensor only overlaps its immediate neighbor:
        // peak concurrent live set is 2, so total should be roughly bounded
        // by 2x the max size rather than growing with tensor count.
        let n = 20;
        let sizes: Vec<usize> = vec![10; n];
        let lifetimes: Vec<(usize, usize)> = (0..n).map(|i| (i, i + 1)).collect();
        let a = Arena::plan(&sizes, &lifetimes);
        assert!(
            a.total_floats() <= 30,
            "expected bounded reuse, got {}",
            a.total_floats()
        );
    }
}
