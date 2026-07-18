/// Bilineares Resize (NCHW, Batch=1), half-pixel-centers Konvention
/// (entspricht PyTorch `F.interpolate(..., mode="bilinear", align_corners=False)`,
/// der in DPT-artigen Netzen ueblichen Variante). Randpixel werden geklemmt
/// (clamp-to-edge), kein Padding.
///
/// - `input`: `c*ih*iw`
/// - `out`: `c*oh*ow`
pub fn bilinear_resize(input: &[f32], c: usize, ih: usize, iw: usize, oh: usize, ow: usize, out: &mut [f32]) {
    debug_assert_eq!(input.len(), c * ih * iw);
    debug_assert_eq!(out.len(), c * oh * ow);

    if ih == 0 || iw == 0 || oh == 0 || ow == 0 {
        return;
    }

    let scale_y = ih as f32 / oh as f32;
    let scale_x = iw as f32 / ow as f32;

    // Pro-Zeile/Spalte vorab die Quellkoordinate + Nachbarindizes + Gewicht
    // berechnen (getrennt fuer y und x), das spart die wiederholte
    // Neuberechnung pro Kanal.
    let (y0s, y1s, wy): (Vec<usize>, Vec<usize>, Vec<f32>) = (0..oh)
        .map(|oy| src_coord(oy, scale_y, ih))
        .fold((Vec::with_capacity(oh), Vec::with_capacity(oh), Vec::with_capacity(oh)), |mut acc, (a, b, w)| {
            acc.0.push(a);
            acc.1.push(b);
            acc.2.push(w);
            acc
        });
    let (x0s, x1s, wx): (Vec<usize>, Vec<usize>, Vec<f32>) = (0..ow)
        .map(|ox| src_coord(ox, scale_x, iw))
        .fold((Vec::with_capacity(ow), Vec::with_capacity(ow), Vec::with_capacity(ow)), |mut acc, (a, b, w)| {
            acc.0.push(a);
            acc.1.push(b);
            acc.2.push(w);
            acc
        });

    for ch in 0..c {
        let plane = &input[ch * ih * iw..(ch + 1) * ih * iw];
        let out_plane = &mut out[ch * oh * ow..(ch + 1) * oh * ow];
        for oy in 0..oh {
            let (y0, y1, fy) = (y0s[oy], y1s[oy], wy[oy]);
            let row0 = &plane[y0 * iw..(y0 + 1) * iw];
            let row1 = &plane[y1 * iw..(y1 + 1) * iw];
            for ox in 0..ow {
                let (x0, x1, fx) = (x0s[ox], x1s[ox], wx[ox]);
                let top = row0[x0] * (1.0 - fx) + row0[x1] * fx;
                let bot = row1[x0] * (1.0 - fx) + row1[x1] * fx;
                out_plane[oy * ow + ox] = top * (1.0 - fy) + bot * fy;
            }
        }
    }
}

/// Fuer eine Zielkoordinate `dst` entlang einer Achse mit `len_in` Quellelementen
/// und `scale = len_in/len_out`: liefert `(idx0, idx1, frac)` mit `idx0<=idx1`
/// geklemmt in `[0, len_in-1]` und `frac` das Interpolationsgewicht Richtung `idx1`.
fn src_coord(dst: usize, scale: f32, len_in: usize) -> (usize, usize, f32) {
    let src = (dst as f32 + 0.5) * scale - 0.5;
    let src_clamped = src.max(0.0);
    let idx0 = src_clamped.floor() as usize;
    let idx0 = idx0.min(len_in.saturating_sub(1));
    let idx1 = (idx0 + 1).min(len_in.saturating_sub(1));
    let frac = if idx1 == idx0 { 0.0 } else { src_clamped - idx0 as f32 };
    let frac = frac.clamp(0.0, 1.0);
    (idx0, idx1, frac)
}

/// Unabhaengig formuliertes, ungefaltetes Referenz-Orakel: berechnet fuer
/// jedes Ausgabepixel direkt (ohne Vorab-Tabellen) die Quellkoordinate neu.
/// Dient als zweite, redundante Implementierung zur Verifikation von
/// [`bilinear_resize`].
pub fn bilinear_resize_naive(input: &[f32], c: usize, ih: usize, iw: usize, oh: usize, ow: usize, out: &mut [f32]) {
    debug_assert_eq!(input.len(), c * ih * iw);
    debug_assert_eq!(out.len(), c * oh * ow);
    if ih == 0 || iw == 0 || oh == 0 || ow == 0 {
        return;
    }
    let scale_y = ih as f32 / oh as f32;
    let scale_x = iw as f32 / ow as f32;
    for ch in 0..c {
        for oy in 0..oh {
            let sy = ((oy as f32 + 0.5) * scale_y - 0.5).max(0.0);
            let y0 = (sy.floor() as usize).min(ih - 1);
            let y1 = (y0 + 1).min(ih - 1);
            let fy = if y1 == y0 { 0.0 } else { (sy - y0 as f32).clamp(0.0, 1.0) };
            for ox in 0..ow {
                let sx = ((ox as f32 + 0.5) * scale_x - 0.5).max(0.0);
                let x0 = (sx.floor() as usize).min(iw - 1);
                let x1 = (x0 + 1).min(iw - 1);
                let fx = if x1 == x0 { 0.0 } else { (sx - x0 as f32).clamp(0.0, 1.0) };

                let get = |y: usize, x: usize| input[(ch * ih + y) * iw + x];
                let top = get(y0, x0) * (1.0 - fx) + get(y0, x1) * fx;
                let bot = get(y1, x0) * (1.0 - fx) + get(y1, x1) * fx;
                out[(ch * oh + oy) * ow + ox] = top * (1.0 - fy) + bot * fy;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bilinear_resize_identity_hand_checked() {
        // oh==ih, ow==iw => scale=1 everywhere => output must equal input
        // exactly, a fully hand-verifiable case independent of the
        // half-pixel-centers convention details.
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut out = vec![0f32; 9];
        bilinear_resize(&input, 1, 3, 3, 3, 3, &mut out);
        assert_eq!(out, input);
    }

    #[test]
    fn bilinear_resize_upsample_2x_hand_checked() {
        // 1x2 -> 1x4 upsample (single row, exercises only x-axis).
        // half-pixel centers: src_x(dst) = (dst+0.5)*0.5 - 0.5
        // dst=0: -0.25 -> clamp 0.0            -> idx0=0,idx1=1,frac=0.0 -> in[0]
        // dst=1: 0.25                          -> idx0=0,idx1=1,frac=0.25
        // dst=2: 0.75                          -> idx0=0,idx1=1,frac=0.75
        // dst=3: 1.25 -> clamp to idx1 (iw-1=1) -> idx0=1,idx1=1,frac=0.0 -> in[1]
        let input = vec![10.0, 20.0];
        let mut out = vec![0f32; 4];
        bilinear_resize(&input, 1, 1, 2, 1, 4, &mut out);
        let expected = vec![
            10.0,
            10.0 * 0.75 + 20.0 * 0.25,
            10.0 * 0.25 + 20.0 * 0.75,
            20.0,
        ];
        for i in 0..4 {
            assert!((out[i] - expected[i]).abs() < 1e-6, "i={i} got={} want={}", out[i], expected[i]);
        }
    }

    #[test]
    fn bilinear_resize_matches_naive_oracle_multichannel() {
        let c = 3;
        let (ih, iw) = (5, 7);
        let (oh, ow) = (9, 4);
        let mut rng = Xorshift32(0xABCD_1234);
        let input = random_vec(&mut rng, c * ih * iw);

        let mut out = vec![0f32; c * oh * ow];
        let mut out_naive = vec![0f32; c * oh * ow];
        bilinear_resize(&input, c, ih, iw, oh, ow, &mut out);
        bilinear_resize_naive(&input, c, ih, iw, oh, ow, &mut out_naive);

        for i in 0..out.len() {
            assert!(
                (out[i] - out_naive[i]).abs() < 1e-5,
                "i={i} fast={} naive={}",
                out[i],
                out_naive[i]
            );
        }
    }

    #[test]
    fn bilinear_resize_downsample_matches_naive_oracle() {
        let c = 2;
        let (ih, iw) = (16, 16);
        let (oh, ow) = (5, 5);
        let mut rng = Xorshift32(0x1357_9BDF);
        let input = random_vec(&mut rng, c * ih * iw);

        let mut out = vec![0f32; c * oh * ow];
        let mut out_naive = vec![0f32; c * oh * ow];
        bilinear_resize(&input, c, ih, iw, oh, ow, &mut out);
        bilinear_resize_naive(&input, c, ih, iw, oh, ow, &mut out_naive);

        for i in 0..out.len() {
            assert!((out[i] - out_naive[i]).abs() < 1e-5, "i={i} fast={} naive={}", out[i], out_naive[i]);
        }
    }

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
}
