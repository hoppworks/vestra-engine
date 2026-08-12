use crate::gemm::Gemm;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
struct WinogradFilterKey {
    in_c: usize,
    out_c: usize,
    words_hash: u64,
}

static WINOGRAD_FILTERS: OnceLock<Mutex<HashMap<WinogradFilterKey, Arc<[f32]>>>> = OnceLock::new();

fn winograd_filter_key(weight: &[f32], in_c: usize, out_c: usize) -> WinogradFilterKey {
    // The model weights are immutable.  Hashing their exact F32 bit patterns
    // makes cache reuse safe across separately loaded models as well.
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for value in weight {
        hash ^= u64::from(value.to_bits());
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    WinogradFilterKey { in_c, out_c, words_hash: hash }
}

fn transformed_winograd_filter(weight: &[f32], in_c: usize, out_c: usize) -> Arc<[f32]> {
    let key = winograd_filter_key(weight, in_c, out_c);
    let cache = WINOGRAD_FILTERS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache.lock().expect("Winograd filter cache mutex poisoned");
    if let Some(filter) = cache.get(&key) {
        return Arc::clone(filter);
    }
    let mut transformed = vec![0.0; out_c * in_c * 16];
    for oc in 0..out_c {
        for ic in 0..in_c {
            let g = &weight[(oc * in_c + ic) * 9..(oc * in_c + ic + 1) * 9];
            let mut t = [[0.0; 3]; 4];
            for j in 0..3 {
                let (a, b, c) = (g[j], g[3 + j], g[6 + j]);
                t[0][j] = a;
                t[1][j] = 0.5 * (a + b + c);
                t[2][j] = 0.5 * (a - b + c);
                t[3][j] = c;
            }
            let u = &mut transformed[(oc * in_c + ic) * 16..(oc * in_c + ic + 1) * 16];
            for i in 0..4 {
                let (a, b, c) = (t[i][0], t[i][1], t[i][2]);
                u[i * 4] = a;
                u[i * 4 + 1] = 0.5 * (a + b + c);
                u[i * 4 + 2] = 0.5 * (a - b + c);
                u[i * 4 + 3] = c;
            }
        }
    }
    let transformed: Arc<[f32]> = transformed.into();
    cache.insert(key, Arc::clone(&transformed));
    transformed
}

/// im2col: expandiert `input` (NCHW, N=1) zu einer `(out_c_rows=kh*kw*in_c) x (oh*ow)`
/// Spaltenmatrix, sodass `conv2d` als eine einzige GEMM (`weight_mat @ col`) berechnet
/// werden kann. `col` ist row-major mit Shape `(in_c*kh*kw) x (oh*ow)`.
fn im2col(
    input: &[f32],
    in_c: usize,
    ih: usize,
    iw: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    pad: usize,
    oh: usize,
    ow: usize,
    col: &mut [f32],
) {
    debug_assert_eq!(input.len(), in_c * ih * iw);
    debug_assert_eq!(col.len(), in_c * kh * kw * oh * ow);
    let out_spatial = oh * ow;
    // Rows are independent.  The large high-resolution DPT convolutions
    // materialize hundreds of MiB here, so distribute the copy while keeping
    // each row's coordinate mapping and values exactly unchanged.
    col.par_chunks_mut(out_spatial)
        .enumerate()
        .for_each(|(row_idx, row)| {
            let c = row_idx / (kh * kw);
            let kernel_idx = row_idx % (kh * kw);
            let ky = kernel_idx / kw;
            let kx = kernel_idx % kw;
            let in_plane = &input[c * ih * iw..(c + 1) * ih * iw];
            for oy in 0..oh {
                let iy = oy as isize * stride as isize + ky as isize - pad as isize;
                for ox in 0..ow {
                    let ix = ox as isize * stride as isize + kx as isize - pad as isize;
                    row[oy * ow + ox] = if iy >= 0 && iy < ih as isize && ix >= 0 && ix < iw as isize {
                        in_plane[iy as usize * iw + ix as usize]
                    } else {
                        0.0
                    };
                }
            }
        });
}

/// Winograd F(2x2, 3x3) convolution for the common stride-1, pad-1 DPT
/// shape.  The transforms use only additions, subtractions and halves.
#[allow(clippy::too_many_arguments)]
fn conv3x3_winograd_f2(
    input: &[f32], in_c: usize, ih: usize, iw: usize, weight: &[f32], out_c: usize,
    bias: Option<&[f32]>, out: &mut [f32],
) {
    let oh = ih;
    let ow = iw;
    let tiles_y = oh.div_ceil(2);
    let tiles_x = ow.div_ceil(2);
    let tiles = tiles_y * tiles_x;
    let transformed = transformed_winograd_filter(weight, in_c, out_c);
    let mut tile_out = vec![0.0; tiles * out_c * 4];
    tile_out
        .par_chunks_mut(out_c * 4)
        .enumerate()
        .for_each_init(
            || vec![0.0; in_c * 16],
            |v, (tile, dst)| {
            let ty = tile / tiles_x;
            let tx = tile % tiles_x;
            let y = ty * 2;
            let x = tx * 2;
            for ic in 0..in_c {
                let d = &mut v[ic * 16..(ic + 1) * 16];
                for dy in 0..4 {
                    let sy = y as isize + dy as isize - 1;
                    for dx in 0..4 {
                        let sx = x as isize + dx as isize - 1;
                        d[dy * 4 + dx] = if sy >= 0 && sy < ih as isize && sx >= 0 && sx < iw as isize {
                            input[(ic * ih + sy as usize) * iw + sx as usize]
                        } else { 0.0 };
                    }
                }
                let mut m = [0.0; 16];
                for j in 0..4 {
                    let (a, b, c, d3) = (d[j], d[4 + j], d[8 + j], d[12 + j]);
                    m[j] = a - c;
                    m[4 + j] = b + c;
                    m[8 + j] = c - b;
                    m[12 + j] = b - d3;
                }
                for i in 0..4 {
                    let (a, b, c, d3) = (m[i * 4], m[i * 4 + 1], m[i * 4 + 2], m[i * 4 + 3]);
                    d[i * 4] = a - c;
                    d[i * 4 + 1] = b + c;
                    d[i * 4 + 2] = c - b;
                    d[i * 4 + 3] = b - d3;
                }
            }
            for oc in 0..out_c {
                let mut m = [0.0; 16];
                for ic in 0..in_c {
                    let u = &transformed[(oc * in_c + ic) * 16..(oc * in_c + ic + 1) * 16];
                    let vv = &v[ic * 16..(ic + 1) * 16];
                    for p in 0..16 { m[p] += u[p] * vv[p]; }
                }
                let mut p = [0.0; 8];
                for j in 0..4 {
                    p[j] = m[j] + m[4 + j] + m[8 + j];
                    p[4 + j] = m[4 + j] - m[8 + j] - m[12 + j];
                }
                let b = bias.map_or(0.0, |values| values[oc]);
                let values = &mut dst[oc * 4..oc * 4 + 4];
                values[0] = p[0] + p[1] + p[2] + b;
                values[1] = p[1] - p[2] - p[3] + b;
                values[2] = p[4] + p[5] + p[6] + b;
                values[3] = p[5] - p[6] - p[7] + b;
            }
            },
        );
    // Output channels own disjoint NCHW planes. Parallelizing the scatter
    // avoids serially copying the full tile buffer after the parallel
    // Winograd transform/multiply stage.
    out.par_chunks_mut(oh * ow)
        .enumerate()
        .for_each(|(oc, out_plane)| {
            for ty in 0..tiles_y {
                for tx in 0..tiles_x {
                    let tile = (ty * tiles_x + tx) * out_c * 4;
                let values = &tile_out[tile + oc * 4..tile + oc * 4 + 4];
                for dy in 0..2 {
                    for dx in 0..2 {
                        let oy = ty * 2 + dy;
                        let ox = tx * 2 + dx;
                        if oy < oh && ox < ow { out_plane[oy * ow + ox] = values[dy * 2 + dx]; }
                    }
                }
            }
            }
        });
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn im2col_serial(
    input: &[f32], in_c: usize, ih: usize, iw: usize, kh: usize, kw: usize,
    stride: usize, pad: usize, oh: usize, ow: usize, col: &mut [f32],
) {
    let out_spatial = oh * ow;
    for c in 0..in_c {
        let in_plane = &input[c * ih * iw..(c + 1) * ih * iw];
        for ky in 0..kh {
            for kx in 0..kw {
                let row_idx = (c * kh + ky) * kw + kx;
                let row = &mut col[row_idx * out_spatial..(row_idx + 1) * out_spatial];
                for oy in 0..oh {
                    let iy = oy as isize * stride as isize + ky as isize - pad as isize;
                    for ox in 0..ow {
                        let ix = ox as isize * stride as isize + kx as isize - pad as isize;
                        row[oy * ow + ox] = if iy >= 0 && iy < ih as isize && ix >= 0 && ix < iw as isize {
                            in_plane[iy as usize * iw + ix as usize]
                        } else { 0.0 };
                    }
                }
            }
        }
    }
}

/// Standard-Conv2D (NCHW, Batch=1) via im2col + GEMM.
///
/// - `input`: `in_c*ih*iw`
/// - `weight`: `out_c*in_c*kh*kw` (PyTorch/GGUF-Layout: OIHW)
/// - `bias`: optional `out_c`
/// - `out`: `out_c*oh*ow`, mit `oh = (ih + 2*pad - kh)/stride + 1` (analog `ow`)
#[allow(clippy::too_many_arguments)]
pub fn conv2d(
    input: &[f32],
    in_c: usize,
    ih: usize,
    iw: usize,
    weight: &[f32],
    out_c: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    pad: usize,
    bias: Option<&[f32]>,
    gemm: &impl Gemm,
    out: &mut [f32],
) {
    debug_assert_eq!(input.len(), in_c * ih * iw);
    debug_assert_eq!(weight.len(), out_c * in_c * kh * kw);
    let oh = (ih + 2 * pad - kh) / stride + 1;
    let ow = (iw + 2 * pad - kw) / stride + 1;
    debug_assert_eq!(out.len(), out_c * oh * ow);

    if kh == 3 && kw == 3 && stride == 1 && pad == 1 {
        conv3x3_winograd_f2(input, in_c, ih, iw, weight, out_c, bias, out);
        return;
    }

    let k = in_c * kh * kw;
    let n = oh * ow;
    // For a 1x1 stride-1 no-padding convolution the im2col matrix is
    // exactly the existing NCHW input viewed as `[in_c, ih*iw]`.  Building
    // and filling a duplicate matrix only adds an allocation and a full
    // memory pass; use the input directly while preserving the same GEMM
    // operand order and F32 accumulation.
    if kh == 1 && kw == 1 && stride == 1 && pad == 0 {
        debug_assert_eq!(n, ih * iw);
        gemm.gemm(out_c, n, k, weight, input, out);
        if let Some(bias) = bias {
            debug_assert_eq!(bias.len(), out_c);
            for oc in 0..out_c {
                let row = &mut out[oc * n..(oc + 1) * n];
                let b = bias[oc];
                for value in row {
                    *value += b;
                }
            }
        }
        return;
    }
    let mut col = vec![0f32; k * n];
    im2col(input, in_c, ih, iw, kh, kw, stride, pad, oh, ow, &mut col);

    // weight ist bereits (out_c) x (in_c*kh*kw) row-major (OIHW geflattet) - passt
    // direkt als GEMM-Operand A.
    gemm.gemm(out_c, n, k, weight, &col, out);

    if let Some(bias) = bias {
        // Bias ist pro Ausgabekanal (Zeile in der out_c x n Matrix), nicht
        // pro Spalte - `scalar::add_bias_rows` broadcastet stattdessen
        // spaltenweise (fuer GEMM-Feature-Bias gedacht), daher hier direkt.
        debug_assert_eq!(bias.len(), out_c);
        for oc in 0..out_c {
            let row = &mut out[oc * n..(oc + 1) * n];
            let b = bias[oc];
            for v in row.iter_mut() {
                *v += b;
            }
        }
    }
}

/// Naives, direktes Conv2D ohne im2col/GEMM - Orakel zur Verifikation von
/// [`conv2d`] gegen brute-force Referenzsemantik.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_naive(
    input: &[f32],
    in_c: usize,
    ih: usize,
    iw: usize,
    weight: &[f32],
    out_c: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    pad: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    debug_assert_eq!(input.len(), in_c * ih * iw);
    debug_assert_eq!(weight.len(), out_c * in_c * kh * kw);
    let oh = (ih + 2 * pad - kh) / stride + 1;
    let ow = (iw + 2 * pad - kw) / stride + 1;
    debug_assert_eq!(out.len(), out_c * oh * ow);

    for oc in 0..out_c {
        for oy in 0..oh {
            for ox in 0..ow {
                let mut acc = 0f32;
                for ic in 0..in_c {
                    for ky in 0..kh {
                        let iy = oy as isize * stride as isize + ky as isize - pad as isize;
                        if iy < 0 || iy >= ih as isize {
                            continue;
                        }
                        for kx in 0..kw {
                            let ix = ox as isize * stride as isize + kx as isize - pad as isize;
                            if ix < 0 || ix >= iw as isize {
                                continue;
                            }
                            let iv = input[(ic * ih + iy as usize) * iw + ix as usize];
                            let wv = weight[((oc * in_c + ic) * kh + ky) * kw + kx];
                            acc += iv * wv;
                        }
                    }
                }
                if let Some(b) = bias {
                    acc += b[oc];
                }
                out[(oc * oh + oy) * ow + ox] = acc;
            }
        }
    }
}

/// ConvTranspose2D (NCHW, Batch=1), z.B. fuer die DPT-resize-Layer (k4s4).
/// Kein Padding-Parameter noetig fuer die hier verwendeten k=s (exaktes
/// Upsampling ohne Ueberlappung/Randbeschnitt); `output_padding` wird nicht
/// unterstuetzt.
///
/// - `weight`: `in_c*out_c*kh*kw` (PyTorch-ConvTranspose-Layout: IOHW)
/// - `bias`: optional `out_c`
/// - `oh = (ih-1)*stride + kh`, analog `ow`.
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose2d(
    input: &[f32],
    in_c: usize,
    ih: usize,
    iw: usize,
    weight: &[f32],
    out_c: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    debug_assert_eq!(input.len(), in_c * ih * iw);
    debug_assert_eq!(weight.len(), in_c * out_c * kh * kw);
    let oh = (ih - 1) * stride + kh;
    let ow = (iw - 1) * stride + kw;
    debug_assert_eq!(out.len(), out_c * oh * ow);

    // The DPT resize layers use kernel == stride with no padding.  Each input
    // spatial location therefore owns a disjoint output tile, so output
    // channels can be computed independently.  Keep the inner input-channel
    // accumulation order identical to the serial scatter path below.
    if kh == stride && kw == stride {
        out.par_chunks_mut(oh * ow)
            .enumerate()
            .for_each(|(oc, plane)| {
                for iy in 0..ih {
                    for ix in 0..iw {
                        for ky in 0..kh {
                            let oy = iy * stride + ky;
                            for kx in 0..kw {
                                let ox = ix * stride + kx;
                                let mut sum = 0.0;
                                for ic in 0..in_c {
                                    let iv = input[(ic * ih + iy) * iw + ix];
                                    let wv = weight[((ic * out_c + oc) * kh + ky) * kw + kx];
                                    sum += iv * wv;
                                }
                                plane[oy * ow + ox] = sum;
                            }
                        }
                    }
                }
                if let Some(bias) = bias {
                    for value in plane {
                        *value += bias[oc];
                    }
                }
            });
        return;
    }

    out.fill(0.0);
    for ic in 0..in_c {
        for iy in 0..ih {
            for ix in 0..iw {
                let iv = input[(ic * ih + iy) * iw + ix];
                if iv == 0.0 {
                    continue;
                }
                for oc in 0..out_c {
                    for ky in 0..kh {
                        let oy = iy * stride + ky;
                        for kx in 0..kw {
                            let ox = ix * stride + kx;
                            let wv = weight[((ic * out_c + oc) * kh + ky) * kw + kx];
                            out[(oc * oh + oy) * ow + ox] += iv * wv;
                        }
                    }
                }
            }
        }
    }
    if let Some(bias) = bias {
        for oc in 0..out_c {
            let plane = &mut out[oc * oh * ow..(oc + 1) * oh * ow];
            for v in plane.iter_mut() {
                *v += bias[oc];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gemm::ScalarGemm;

    #[test]
    fn conv2d_matches_naive_and_direct_math_1x1() {
        // 1x1 conv is a per-pixel GEMM: in_c=3, out_c=2, spatial=2x2.
        // weight[oc][ic] chosen so output is hand-checkable.
        let in_c = 3;
        let (ih, iw) = (2, 2);
        let input: Vec<f32> = (1..=(in_c * ih * iw) as i32).map(|v| v as f32).collect();
        // input plane0: [1,2,3,4], plane1: [5,6,7,8], plane2: [9,10,11,12]
        let out_c = 2;
        // weight oc0: [1,0,0] (picks plane0), oc1: [0,0,1] (picks plane2)
        let weight = vec![
            1.0, 0.0, 0.0, // oc0
            0.0, 0.0, 1.0, // oc1
        ];
        let bias = vec![10.0, -1.0];
        let mut out = vec![0f32; out_c * ih * iw];
        conv2d(
            &input, in_c, ih, iw, &weight, out_c, 1, 1, 1, 0, Some(&bias), &ScalarGemm, &mut out,
        );
        // oc0 = plane0 + 10 = [11,12,13,14]; oc1 = plane2 - 1 = [8,9,10,11]
        assert_eq!(out, vec![11.0, 12.0, 13.0, 14.0, 8.0, 9.0, 10.0, 11.0]);

        let mut out_naive = vec![0f32; out_c * ih * iw];
        conv2d_naive(
            &input, in_c, ih, iw, &weight, out_c, 1, 1, 1, 0, Some(&bias), &mut out_naive,
        );
        assert_eq!(out, out_naive);
    }

    #[test]
    fn conv2d_1x1_fast_path_is_bitwise_generic_im2col() {
        let (in_c, out_c, ih, iw) = (5, 7, 4, 3);
        let mut rng = Xorshift32(0x1A11_C011);
        let input = random_vec(&mut rng, in_c * ih * iw);
        let weight = random_vec(&mut rng, out_c * in_c);
        let bias = random_vec(&mut rng, out_c);
        let mut fast = vec![0.0; out_c * ih * iw];
        conv2d(
            &input,
            in_c,
            ih,
            iw,
            &weight,
            out_c,
            1,
            1,
            1,
            0,
            Some(&bias),
            &ScalarGemm,
            &mut fast,
        );

        let mut col = vec![0.0; in_c * ih * iw];
        im2col(&input, in_c, ih, iw, 1, 1, 1, 0, ih, iw, &mut col);
        let mut generic = vec![0.0; out_c * ih * iw];
        ScalarGemm.gemm(out_c, ih * iw, in_c, &weight, &col, &mut generic);
        for oc in 0..out_c {
            for value in &mut generic[oc * ih * iw..(oc + 1) * ih * iw] {
                *value += bias[oc];
            }
        }
        assert_eq!(fast, generic);
    }

    #[test]
    fn conv2d_matches_naive_3x3_stride2_pad1() {
        let in_c = 2;
        let (ih, iw) = (7, 5);
        let out_c = 3;
        let mut rng = Xorshift32(0xC0FF_EE01);
        let input = random_vec(&mut rng, in_c * ih * iw);
        let weight = random_vec(&mut rng, out_c * in_c * 3 * 3);
        let bias = random_vec(&mut rng, out_c);

        let stride = 2;
        let pad = 1;
        let oh = (ih + 2 * pad - 3) / stride + 1;
        let ow = (iw + 2 * pad - 3) / stride + 1;

        let mut out_gemm = vec![0f32; out_c * oh * ow];
        conv2d(
            &input, in_c, ih, iw, &weight, out_c, 3, 3, stride, pad, Some(&bias), &ScalarGemm,
            &mut out_gemm,
        );
        let mut out_naive = vec![0f32; out_c * oh * ow];
        conv2d_naive(
            &input, in_c, ih, iw, &weight, out_c, 3, 3, stride, pad, Some(&bias), &mut out_naive,
        );
        for i in 0..out_gemm.len() {
            assert!(
                (out_gemm[i] - out_naive[i]).abs() < 1e-3,
                "i={i} gemm={} naive={}",
                out_gemm[i],
                out_naive[i]
            );
        }
    }

    #[test]
    fn conv_transpose2d_hand_checked_1x1_spatial_k4s4() {
        // 1x1 spatial input, 2 in-channels, 2 out-channels, k=4 s=4: this is
        // the DPT resize-layer shape family. With a single input pixel the
        // whole output IS the kernel (scaled by input and summed over ic),
        // making it hand-checkable.
        let in_c = 2;
        let (ih, iw) = (1, 1);
        let out_c = 2;
        let kh = 4;
        let kw = 4;
        let stride = 4;
        let input = vec![2.0, 3.0]; // ic0=2, ic1=3
        // weight layout IOHW: [ic][oc][kh][kw]
        let mut weight = vec![0f32; in_c * out_c * kh * kw];
        // ic0->oc0: all ones (16 elems)
        for i in 0..16 {
            weight[i] = 1.0;
        }
        // ic0->oc1: all zeros (default)
        // ic1->oc0: all zeros
        // ic1->oc1: constant 5.0
        for i in 0..16 {
            weight[(1 * out_c + 1) * 16 + i] = 5.0;
        }
        let bias = vec![1.0, -2.0];
        let oh = (ih - 1) * stride + kh;
        let ow = (iw - 1) * stride + kw;
        let mut out = vec![0f32; out_c * oh * ow];
        conv_transpose2d(
            &input, in_c, ih, iw, &weight, out_c, kh, kw, stride, Some(&bias), &mut out,
        );
        // oc0 = ic0*1.0 + ic1*0.0 + bias0 = 2*1 + 1 = 3 everywhere (4x4=16 elems)
        // oc1 = ic0*0.0 + ic1*5.0 + bias1 = 3*5 - 2 = 13 everywhere
        assert_eq!(&out[0..16], &[3.0; 16][..]);
        assert_eq!(&out[16..32], &[13.0; 16][..]);
    }

    #[test]
    fn conv_transpose2d_matches_naive_oracle() {
        let in_c = 3;
        let (ih, iw) = (2, 3);
        let out_c = 2;
        let kh = 4;
        let kw = 4;
        let stride = 4;
        let mut rng = Xorshift32(0xFEED_1234);
        let input = random_vec(&mut rng, in_c * ih * iw);
        let weight = random_vec(&mut rng, in_c * out_c * kh * kw);
        let bias = random_vec(&mut rng, out_c);

        let oh = (ih - 1) * stride + kh;
        let ow = (iw - 1) * stride + kw;
        let mut out = vec![0f32; out_c * oh * ow];
        conv_transpose2d(
            &input, in_c, ih, iw, &weight, out_c, kh, kw, stride, Some(&bias), &mut out,
        );

        let mut out_naive = vec![0f32; out_c * oh * ow];
        conv_transpose2d_naive(
            &input, in_c, ih, iw, &weight, out_c, kh, kw, stride, Some(&bias), &mut out_naive,
        );
        for i in 0..out.len() {
            assert!(
                (out[i] - out_naive[i]).abs() < 1e-4,
                "i={i} fast={} naive={}",
                out[i],
                out_naive[i]
            );
        }
    }

    #[test]
    fn nonoverlap_transpose_fast_path_is_bitwise_serial_scatter() {
        let (in_c, out_c, ih, iw, kernel) = (3, 4, 3, 2, 2);
        let mut rng = Xorshift32(0x7A4E_0001);
        let input = random_vec(&mut rng, in_c * ih * iw);
        let weight = random_vec(&mut rng, in_c * out_c * kernel * kernel);
        let bias = random_vec(&mut rng, out_c);
        let (oh, ow) = (ih * kernel, iw * kernel);
        let mut fast = vec![0.0; out_c * oh * ow];
        conv_transpose2d(
            &input,
            in_c,
            ih,
            iw,
            &weight,
            out_c,
            kernel,
            kernel,
            kernel,
            Some(&bias),
            &mut fast,
        );

        let mut serial = vec![0.0; out_c * oh * ow];
        for ic in 0..in_c {
            for iy in 0..ih {
                for ix in 0..iw {
                    let iv = input[(ic * ih + iy) * iw + ix];
                    for oc in 0..out_c {
                        for ky in 0..kernel {
                            for kx in 0..kernel {
                                serial[(oc * oh + iy * kernel + ky) * ow + ix * kernel + kx] +=
                                    iv * weight[((ic * out_c + oc) * kernel + ky) * kernel + kx];
                            }
                        }
                    }
                }
            }
        }
        for oc in 0..out_c {
            for value in &mut serial[oc * oh * ow..(oc + 1) * oh * ow] {
                *value += bias[oc];
            }
        }
        assert_eq!(fast, serial);
    }

    #[test]
    fn parallel_im2col_is_bitwise_serial() {
        let (in_c, ih, iw, kh, kw, stride, pad) = (3, 7, 9, 3, 3, 2, 1);
        let (oh, ow) = ((ih + 2 * pad - kh) / stride + 1, (iw + 2 * pad - kw) / stride + 1);
        let mut rng = Xorshift32(0x1A2C_0001); // deterministic test seed
        let input = random_vec(&mut rng, in_c * ih * iw);
        let mut parallel = vec![0.0; in_c * kh * kw * oh * ow];
        let mut serial = vec![0.0; parallel.len()];
        im2col(&input, in_c, ih, iw, kh, kw, stride, pad, oh, ow, &mut parallel);
        im2col_serial(&input, in_c, ih, iw, kh, kw, stride, pad, oh, ow, &mut serial);
        assert_eq!(parallel, serial);
    }

    #[test]
    fn winograd_f2_matches_direct_3x3_oracle() {
        let (in_c, out_c, h, w) = (3, 5, 7, 9);
        let mut rng = Xorshift32(0xF2F2_0001);
        let input = random_vec(&mut rng, in_c * h * w);
        let weight = random_vec(&mut rng, out_c * in_c * 9);
        let bias = random_vec(&mut rng, out_c);
        let mut winograd = vec![0.0; out_c * h * w];
        let mut direct = vec![0.0; winograd.len()];
        conv3x3_winograd_f2(&input, in_c, h, w, &weight, out_c, Some(&bias), &mut winograd);
        conv2d_naive(&input, in_c, h, w, &weight, out_c, 3, 3, 1, 1, Some(&bias), &mut direct);
        for (i, (got, expected)) in winograd.iter().zip(direct.iter()).enumerate() {
            assert!((got - expected).abs() < 2e-5, "i={i} got={got} expected={expected}");
        }
    }

    /// Deterministischer, dependency-freier PRNG (Xorshift32) fuer reproduzierbare
    /// Testdaten.
    struct Xorshift32(u32);
    impl Xorshift32 {
        fn next_f32(&mut self) -> f32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            self.0 = x;
            ((x as f32) / (u32::MAX as f32)) * 2.0 - 1.0
        }
    }
    fn random_vec(rng: &mut Xorshift32, n: usize) -> Vec<f32> {
        (0..n).map(|_| rng.next_f32()).collect()
    }

    /// Zweites, unabhaengig formuliertes Orakel fuer conv_transpose2d (direkte
    /// "scatter"-Definition ohne die Optimierungen der Hauptimplementierung -
    /// hier ist die Hauptimplementierung selbst schon die naive Variante,
    /// daher spiegelt dieses Orakel die mathematische Definition explizit als
    /// "gather" (aus Sicht des Outputs) statt "scatter" (aus Sicht des Inputs).
    #[allow(clippy::too_many_arguments)]
    fn conv_transpose2d_naive(
        input: &[f32],
        in_c: usize,
        ih: usize,
        iw: usize,
        weight: &[f32],
        out_c: usize,
        kh: usize,
        kw: usize,
        stride: usize,
        bias: Option<&[f32]>,
        out: &mut [f32],
    ) {
        let oh = (ih - 1) * stride + kh;
        let ow = (iw - 1) * stride + kw;
        debug_assert_eq!(out.len(), out_c * oh * ow);
        for oc in 0..out_c {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut acc = 0f32;
                    for ic in 0..in_c {
                        for ky in 0..kh {
                            if oy < ky {
                                continue;
                            }
                            let num = oy - ky;
                            if num % stride != 0 {
                                continue;
                            }
                            let iy = num / stride;
                            if iy >= ih {
                                continue;
                            }
                            for kx in 0..kw {
                                if ox < kx {
                                    continue;
                                }
                                let numx = ox - kx;
                                if numx % stride != 0 {
                                    continue;
                                }
                                let ix = numx / stride;
                                if ix >= iw {
                                    continue;
                                }
                                let iv = input[(ic * ih + iy) * iw + ix];
                                let wv = weight[((ic * out_c + oc) * kh + ky) * kw + kx];
                                acc += iv * wv;
                            }
                        }
                    }
                    if let Some(b) = bias {
                        acc += b[oc];
                    }
                    out[(oc * oh + oy) * ow + ox] = acc;
                }
            }
        }
    }
}
