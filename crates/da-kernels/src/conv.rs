use crate::gemm::Gemm;

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
                        let v = if iy >= 0 && iy < ih as isize && ix >= 0 && ix < iw as isize {
                            in_plane[iy as usize * iw + ix as usize]
                        } else {
                            0.0
                        };
                        row[oy * ow + ox] = v;
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

    let k = in_c * kh * kw;
    let n = oh * ow;
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
