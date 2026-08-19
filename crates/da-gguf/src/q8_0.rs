//! GGML-compatible Q8_0 wire-layout decoding.
//!
//! The 32-value block shape, f16 scale, and packed i8 layout follow ggml at
//! pinned revision `eced84c86f8b012c752c016f7fe789adea168e1e` (MIT). The
//! decoding loop is an independent Rust implementation. See the
//! repository-root `THIRD_PARTY_NOTICES.md`.

use half::f16;

pub const QK8_0: usize = 32;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlockQ8_0 {
    pub d: f16,
    pub qs: [i8; 32],
}

pub struct TensorQ8_0 {
    pub shape: Vec<i64>,
    pub blocks: Vec<BlockQ8_0>,
}

pub fn dequantize_q8_0(blocks: &[BlockQ8_0], out: &mut [f32]) {
    debug_assert_eq!(out.len(), blocks.len() * QK8_0);
    for (bi, b) in blocks.iter().enumerate() {
        let d = b.d.to_f32();
        let base = bi * QK8_0;
        for j in 0..QK8_0 {
            out[base + j] = d * b.qs[j] as f32;
        }
    }
}

// Rohbytes eines Q8_0-Tensors (dicht gepackt, 34 Bytes/Block) → Vec<BlockQ8_0>.
pub(crate) fn read_blocks(bytes: &[u8], n_elems: usize) -> Vec<BlockQ8_0> {
    let nblocks = n_elems / QK8_0;
    let mut v = Vec::with_capacity(nblocks);
    let mut p = 0usize;
    for _ in 0..nblocks {
        let d = f16::from_le_bytes([bytes[p], bytes[p + 1]]);
        p += 2;
        let mut qs = [0i8; 32];
        for j in 0..32 {
            qs[j] = bytes[p + j] as i8;
        }
        p += 32;
        v.push(BlockQ8_0 { d, qs });
    }
    v
}
