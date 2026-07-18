//! AVX-512F kernels. This module is only ever compiled on `target_arch =
//! "x86_64"` (see the `#[cfg(...)]` gate on the `mod simd_avx512;`
//! declaration in `lib.rs`). It is developed on an aarch64 (Apple Silicon)
//! host where AVX-512 cannot execute or even be emulated; see
//! `docs/optimization-log.md` for how these kernels are verified on this
//! machine (x86_64 compile-check + scalar oracle only — NOT numerically
//! verified on real x86-64 hardware yet).

use core::arch::x86_64::*;

const LANES: usize = 16; // f32 lanes per __m512

/// Vectorized `expf` (Cephes-style range reduction + minimax polynomial),
/// matching the accuracy needs of `erf_avx512` below. Operates on all 16
/// lanes of `x` at once.
#[target_feature(enable = "avx512f")]
unsafe fn exp_avx512(x: __m512) -> __m512 {
    let exp_hi = _mm512_set1_ps(88.376_26);
    let exp_lo = _mm512_set1_ps(-88.376_26);
    let log2ef = _mm512_set1_ps(1.442_695_04);
    let half = _mm512_set1_ps(0.5);
    let one = _mm512_set1_ps(1.0);
    let ln2_hi = _mm512_set1_ps(0.693_359_375);
    let ln2_lo = _mm512_set1_ps(-2.121_944_4e-4);

    let p0 = _mm512_set1_ps(1.987_569_15e-4);
    let p1 = _mm512_set1_ps(1.398_199_95e-3);
    let p2 = _mm512_set1_ps(8.333_451_9e-3);
    let p3 = _mm512_set1_ps(4.166_579_6e-2);
    let p4 = _mm512_set1_ps(1.666_666_5e-1);
    let p5 = _mm512_set1_ps(5.000_000_1e-1);

    let x = _mm512_min_ps(x, exp_hi);
    let x = _mm512_max_ps(x, exp_lo);

    // fx = floor(x * log2(e) + 0.5)
    let fx0 = _mm512_fmadd_ps(x, log2ef, half);
    let fx_trunc = _mm512_cvtepi32_ps(_mm512_cvttps_epi32(fx0));
    let gt_mask = _mm512_cmp_ps_mask(fx_trunc, fx0, _CMP_GT_OQ);
    let fx = _mm512_mask_sub_ps(fx_trunc, gt_mask, fx_trunc, one);

    // x -= fx * ln2 (in two steps for precision)
    let x = _mm512_fnmadd_ps(fx, ln2_hi, x);
    let x = _mm512_fnmadd_ps(fx, ln2_lo, x);

    let z = _mm512_mul_ps(x, x);

    let mut y = p0;
    y = _mm512_fmadd_ps(y, x, p1);
    y = _mm512_fmadd_ps(y, x, p2);
    y = _mm512_fmadd_ps(y, x, p3);
    y = _mm512_fmadd_ps(y, x, p4);
    y = _mm512_fmadd_ps(y, x, p5);
    y = _mm512_fmadd_ps(y, z, x);
    let y = _mm512_add_ps(y, one);

    // pow2n = 2^fx via direct exponent-bit construction
    let emm0 = _mm512_cvttps_epi32(fx);
    let emm0 = _mm512_add_epi32(emm0, _mm512_set1_epi32(0x7f));
    let emm0 = _mm512_slli_epi32(emm0, 23);
    let pow2n = _mm512_castsi512_ps(emm0);

    _mm512_mul_ps(y, pow2n)
}

/// Vectorized `erf` using the same Abramowitz-Stegun 7.1.26 approximation
/// as `scalar::erf` (|error| < 1.5e-7), so `gelu_avx512` stays within the
/// tolerance band of `scalar::gelu`.
#[target_feature(enable = "avx512f")]
unsafe fn erf_avx512(x: __m512) -> __m512 {
    let abs_mask = _mm512_set1_epi32(0x7fff_ffff);

    let xi = _mm512_castps_si512(x);
    let x_abs = _mm512_castsi512_ps(_mm512_and_epi32(xi, abs_mask));

    let zero = _mm512_setzero_ps();
    let one = _mm512_set1_ps(1.0);
    let neg_one = _mm512_set1_ps(-1.0);
    let neg_mask = _mm512_cmp_ps_mask(x, zero, _CMP_LT_OQ);
    let sign = _mm512_mask_blend_ps(neg_mask, one, neg_one);

    let a1 = _mm512_set1_ps(0.254_829_59);
    let a2 = _mm512_set1_ps(-0.284_496_74);
    let a3 = _mm512_set1_ps(1.421_413_7);
    let a4 = _mm512_set1_ps(-1.453_152_0);
    let a5 = _mm512_set1_ps(1.061_405_4);
    let c = _mm512_set1_ps(0.327_591_1);

    // t = 1 / (1 + c * x_abs)
    let denom = _mm512_fmadd_ps(c, x_abs, one);
    let t = _mm512_div_ps(one, denom);

    let mut poly = a5;
    poly = _mm512_fmadd_ps(poly, t, a4);
    poly = _mm512_fmadd_ps(poly, t, a3);
    poly = _mm512_fmadd_ps(poly, t, a2);
    poly = _mm512_fmadd_ps(poly, t, a1);
    poly = _mm512_mul_ps(poly, t);

    let neg_x2 = _mm512_sub_ps(zero, _mm512_mul_ps(x_abs, x_abs));
    let exp_term = exp_avx512(neg_x2);

    let y = _mm512_fnmadd_ps(poly, exp_term, one); // 1 - poly * exp_term
    _mm512_mul_ps(sign, y)
}

/// In-place GELU (exact-erf formulation): `x = 0.5*x*(1 + erf(x/sqrt(2)))`.
/// See `scalar::gelu` for the reference implementation this must match
/// within tolerance.
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn gelu_avx512(x: &mut [f32]) {
    const INV_SQRT2: f32 = 0.707_106_78;
    let inv_sqrt2 = _mm512_set1_ps(INV_SQRT2);
    let half = _mm512_set1_ps(0.5);
    let one = _mm512_set1_ps(1.0);

    let n = x.len();
    let main = n - (n % LANES);

    let mut i = 0;
    while i < main {
        let ptr = x.as_mut_ptr().add(i);
        let v = _mm512_loadu_ps(ptr);
        let arg = _mm512_mul_ps(v, inv_sqrt2);
        let e = erf_avx512(arg);
        let one_plus_e = _mm512_add_ps(one, e);
        let half_v = _mm512_mul_ps(half, v);
        let out = _mm512_mul_ps(half_v, one_plus_e);
        _mm512_storeu_ps(ptr, out);
        i += LANES;
    }

    debug_assert!(n - main < LANES);
    if main < n {
        crate::scalar::gelu(&mut x[main..n]);
    }
}

/// In-place elementwise `dst += src`, trivial AVX-512F load/add/store with
/// a scalar tail for the remainder (< 16 lanes).
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn add_avx512(dst: &mut [f32], src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    let n = dst.len();
    let main = n - (n % LANES);

    let mut i = 0;
    while i < main {
        let d_ptr = dst.as_mut_ptr().add(i);
        let s_ptr = src.as_ptr().add(i);
        let d = _mm512_loadu_ps(d_ptr);
        let s = _mm512_loadu_ps(s_ptr);
        let r = _mm512_add_ps(d, s);
        _mm512_storeu_ps(d_ptr, r);
        i += LANES;
    }

    debug_assert!(n - main < LANES);
    if main < n {
        crate::scalar::add(&mut dst[main..n], &src[main..n]);
    }
}
