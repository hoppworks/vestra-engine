//! The camera pose head (`CamPose::pose`): takes the layer-11 camera token
//! (`BackboneOutputs::cam_tokens[3]`, `[D]` — `D = 2*embed_dim` when
//! `cfg.cat_token`, e.g. `1536` for DA3-BASE) and produces a 9-element
//! `pose_enc` vector via a small MLP + 3 linear heads, then decodes that
//! into extrinsics (`3x4` affine-inverse of camera-to-world) and intrinsics
//! (`3x3` `K` matrix, resolution-dependent).
//!
//! Ported line-for-line from the real C++ reference, `../src/cam_pose.cpp`
//! (85 lines, read directly during this task's investigation, not
//! reverse-engineered) — see this module's per-function doc comments for
//! the exact correspondence.
//!
//! ## Step 1 — MLP + heads -> `pose_enc[9]`
//!
//! ```text
//! feat = relu(linear(cam.bb0.weight, cam_token, cam.bb0.bias))   // [D] -> [hidden0]
//! feat = relu(linear(cam.bb2.weight, feat,      cam.bb2.bias))   // [hidden0] -> [hidden1]
//! t_head   = linear(cam.fc_t.weight,   feat, cam.fc_t.bias)          // [3], NO relu
//! q_head   = linear(cam.fc_q.weight,   feat, cam.fc_q.bias)          // [4], NO relu
//! fov_head = relu(linear(cam.fc_fov.weight, feat, cam.fc_fov.bias))  // [2], relu AFTER the linear
//! pose_enc = concat([t_head, q_head, fov_head])  // [9] = [Tx,Ty,Tz, qi,qj,qk,qr, fov_h,fov_w]
//! ```
//!
//! Hidden dims for `bb0`/`bb2` are never hardcoded — they're derived from
//! the loaded weight tensors' shapes at runtime (`bias.len()` gives the
//! output width, `weight.len() / bias.len()` gives the input width), same
//! pattern as every other linear layer in this codebase.
//!
//! ## Linear-weight orientation
//!
//! Same convention as `vit_block.rs`'s "Linear-weight orientation" doc
//! comment: this module's `linear_vec` helper expects `w_name` already laid
//! out `[in_features, out_features]` — i.e. the **transpose** of the raw
//! PyTorch/GGUF `nn.Linear.weight` layout (`[out_features, in_features]`).
//! Real GGUF weight loading (transposing on load) is Task 20's job, not
//! this task's.
//!
//! ## Step 2 — `pose_encoding_to_extri_intri`: decode the 9-vector
//!
//! Quaternion is **XYZW, scalar-LAST** (`qr` is the `w`/scalar component,
//! NOT scalar-first). `T = [Tx,Ty,Tz]` from `pose_enc[0..3]` IS the c2w
//! translation directly (no separate transform). `extrinsics =
//! affine_inverse([R | T])`, `affine_inverse([R|T]) = [R^T | -R^T @ T]`.
//! `intrinsics` uses the caller-supplied target pixel resolution `(h, w)`
//! (NOT any config-derived size) via `fy = (H/2)/tan(fov_h/2)`, `fx =
//! (W/2)/tan(fov_w/2)`, each denominator floored at `1e-6` matching the C++
//! `std::max(std::tan(...), 1e-6f)`.
//!
//! ## Weight tensor names (verbatim from `../src/cam_pose.cpp`'s
//! `ml_.tensor(...)` calls)
//!
//! `cam.bb0.weight`/`.bias`, `cam.bb2.weight`/`.bias`, `cam.fc_t.weight`/
//! `.bias`, `cam.fc_q.weight`/`.bias`, `cam.fc_fov.weight`/`.bias`.
//!
//! ## Honesty note
//!
//! The MLP/head forward pass (`cam.bb0`/`cam.bb2`/`cam.fc_t`/`cam.fc_q`/
//! `cam.fc_fov`) is structurally transcribed from the C++ source but NOT
//! numerically cross-checked against real weights/dumps in this
//! environment (`tests/pose_parity.rs` skips cleanly — no
//! `../models/da3-base-f16.gguf` or `../dumps/reference.gguf` present). The
//! quaternion-to-rotation-matrix, FOV-to-intrinsics, and affine-inverse
//! math in `pose_encoding_to_extri_intri` (this module's `decode` function)
//! is fully self-contained arithmetic with no weight dependency at all, so
//! it IS fully verified here via synthetic unit tests (see this module's
//! `#[cfg(test)]` block) — known quaternions, known FOVs, and the
//! `affine_inverse(affine_inverse(x)) == x` round-trip property.

use da_graph::Weights;
use da_kernels::gemm::{Gemm, ScalarGemm};
use da_kernels::scalar;

use crate::config::EngineError;
use crate::ModelConfig;

/// Output of [`cam_pose`]: the raw 9-element pose encoding plus its decoded
/// extrinsics (row-major `3x4`, `[R^T | -R^T@T]`) and intrinsics (row-major
/// `3x3`, `[[fx,0,W/2],[0,fy,H/2],[0,0,1]]`).
#[derive(Debug, Clone, PartialEq)]
pub struct PoseOut {
    pub extrinsics: [f32; 12],
    pub intrinsics: [f32; 9],
    pub pose_enc: [f32; 9],
}

fn get_weight<'a>(weights: &'a Weights, name: &str) -> &'a [f32] {
    weights
        .get_f32(name)
        .unwrap_or_else(|| panic!("missing weight tensor {name:?}"))
}

/// `y = x[1,k] @ w[k,n] + bias[n]`. `w` must already be laid out
/// `[in_features, out_features]` — see this module's doc comment.
fn linear_vec(x: &[f32], k: usize, w: &[f32], b: &[f32]) -> Vec<f32> {
    let n = b.len();
    debug_assert_eq!(
        w.len(),
        k * n,
        "linear weight shape mismatch: expected {k}*{n}, got {}",
        w.len()
    );
    let mut y = vec![0f32; n];
    ScalarGemm.gemm(1, n, k, x, w, &mut y);
    scalar::add_bias_rows(&mut y, 1, n, b);
    y
}

fn relu(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = v.max(0.0);
    }
}

/// Runs the `cam.bb0`/`cam.bb2`/`cam.fc_t`/`cam.fc_q`/`cam.fc_fov` MLP +
/// heads, returning `pose_enc[9] = [Tx,Ty,Tz, qi,qj,qk,qr, fov_h,fov_w]`.
/// Mirrors `../src/cam_pose.cpp::CamPose::pose`'s `be_.compute(...)` graph
/// lambda line-for-line.
fn mlp_pose_enc(cam_token: &[f32], d: usize, weights: &Weights) -> [f32; 9] {
    let bb0_w = get_weight(weights, "cam.bb0.weight");
    let bb0_b = get_weight(weights, "cam.bb0.bias");
    let mut feat = linear_vec(cam_token, d, bb0_w, bb0_b);
    relu(&mut feat);
    let hidden0 = bb0_b.len();

    let bb2_w = get_weight(weights, "cam.bb2.weight");
    let bb2_b = get_weight(weights, "cam.bb2.bias");
    let mut feat = linear_vec(&feat, hidden0, bb2_w, bb2_b);
    relu(&mut feat);
    let hidden1 = bb2_b.len();

    let fc_t_w = get_weight(weights, "cam.fc_t.weight");
    let fc_t_b = get_weight(weights, "cam.fc_t.bias");
    let t_head = linear_vec(&feat, hidden1, fc_t_w, fc_t_b); // [3], no relu
    debug_assert_eq!(t_head.len(), 3);

    let fc_q_w = get_weight(weights, "cam.fc_q.weight");
    let fc_q_b = get_weight(weights, "cam.fc_q.bias");
    let q_head = linear_vec(&feat, hidden1, fc_q_w, fc_q_b); // [4], no relu
    debug_assert_eq!(q_head.len(), 4);

    let fc_fov_w = get_weight(weights, "cam.fc_fov.weight");
    let fc_fov_b = get_weight(weights, "cam.fc_fov.bias");
    let mut fov_head = linear_vec(&feat, hidden1, fc_fov_w, fc_fov_b); // [2]
    relu(&mut fov_head); // relu AFTER fc_fov
    debug_assert_eq!(fov_head.len(), 2);

    [
        t_head[0],
        t_head[1],
        t_head[2],
        q_head[0],
        q_head[1],
        q_head[2],
        q_head[3],
        fov_head[0],
        fov_head[1],
    ]
}

/// `pose_encoding_to_extri_intri`: decodes a 9-element `pose_enc` into
/// extrinsics (`3x4` row-major) + intrinsics (`3x3` row-major) at target
/// pixel resolution `(h, w)`. Pure arithmetic, no weight dependency — see
/// this module's doc comment and `#[cfg(test)]` block.
///
/// Ported directly from `../src/cam_pose.cpp::CamPose::pose`'s
/// post-MLP section.
fn decode(pe: &[f32; 9], h: usize, w: usize) -> ([f32; 12], [f32; 9]) {
    let (tx, ty, tz) = (pe[0], pe[1], pe[2]);
    // quaternion XYZW, scalar-LAST (qr is the scalar/w component).
    let (qi, qj, qk, qr) = (pe[3], pe[4], pe[5], pe[6]);
    let (fov_h, fov_w) = (pe[7], pe[8]);

    // quat_to_mat -> R (3x3 row-major)
    let s = 2.0f32 / (qi * qi + qj * qj + qk * qk + qr * qr);
    let r = [
        [
            1.0 - s * (qj * qj + qk * qk),
            s * (qi * qj - qk * qr),
            s * (qi * qk + qj * qr),
        ],
        [
            s * (qi * qj + qk * qr),
            1.0 - s * (qi * qi + qk * qk),
            s * (qj * qk - qi * qr),
        ],
        [
            s * (qi * qk - qj * qr),
            s * (qj * qk + qi * qr),
            1.0 - s * (qi * qi + qj * qj),
        ],
    ];

    let t = [tx, ty, tz]; // c2w translation, no separate transform

    // intrinsics K
    let fy = (h as f32 / 2.0) / (fov_h / 2.0).tan().max(1e-6);
    let fx = (w as f32 / 2.0) / (fov_w / 2.0).tan().max(1e-6);
    let mut intrinsics = [0f32; 9];
    intrinsics[0 * 3 + 0] = fx;
    intrinsics[0 * 3 + 2] = w as f32 / 2.0;
    intrinsics[1 * 3 + 1] = fy;
    intrinsics[1 * 3 + 2] = h as f32 / 2.0;
    intrinsics[2 * 3 + 2] = 1.0;

    // affine_inverse(c2w) = [R^T | -R^T @ T]
    let mut rt = [[0f32; 3]; 3];
    for row in 0..3 {
        for col in 0..3 {
            rt[row][col] = r[col][row];
        }
    }
    let mut neg_rt_t = [0f32; 3];
    for row in 0..3 {
        let acc: f32 = (0..3).map(|c| rt[row][c] * t[c]).sum();
        neg_rt_t[row] = -acc;
    }
    let mut extrinsics = [0f32; 12];
    for row in 0..3 {
        extrinsics[row * 4] = rt[row][0];
        extrinsics[row * 4 + 1] = rt[row][1];
        extrinsics[row * 4 + 2] = rt[row][2];
        extrinsics[row * 4 + 3] = neg_rt_t[row];
    }

    (extrinsics, intrinsics)
}

/// Runs the camera pose head: `cam_token` (`BackboneOutputs::cam_tokens[3]`,
/// the layer-11 out_layer token) -> `pose_enc[9]` (MLP + 3 heads) ->
/// `(extrinsics, intrinsics)` at target pixel resolution `(h, w)`.
///
/// # Errors
/// Returns [`EngineError::CamTokenDimMismatch`] if `cam_token` is empty or
/// its length doesn't match `cam.bb0.weight`'s input dim — mirrors
/// `../src/cam_pose.cpp`'s `if (cam_token.empty() || !bb0 ||
/// (int64_t)cam_token.size() != bb0->ne[0]) return false;` validation.
///
/// # Panics
/// If any of the `cam.bb0`/`cam.bb2`/`cam.fc_t`/`cam.fc_q`/`cam.fc_fov`
/// weight/bias tensors are missing from `weights` (a structural
/// weight-loading bug, not a runtime/input error — same convention as
/// `dpt_head.rs`/`vit_block.rs`'s `get_weight` helpers).
pub fn cam_pose(
    cam_token: &[f32],
    h: usize,
    w: usize,
    _cfg: &ModelConfig,
    weights: &Weights,
) -> Result<PoseOut, EngineError> {
    let bb0_w = get_weight(weights, "cam.bb0.weight");
    let bb0_b = get_weight(weights, "cam.bb0.bias");
    let hidden0 = bb0_b.len();
    debug_assert!(hidden0 > 0, "cam.bb0.bias must be non-empty");
    let d = bb0_w.len() / hidden0;

    if cam_token.is_empty() || cam_token.len() != d {
        return Err(EngineError::CamTokenDimMismatch {
            expected: d,
            got: cam_token.len(),
        });
    }

    let pose_enc = mlp_pose_enc(cam_token, d, weights);
    let (extrinsics, intrinsics) = decode(&pose_enc, h, w);

    Ok(PoseOut {
        extrinsics,
        intrinsics,
        pose_enc,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn approx(a: f32, b: f32, eps: f32) {
        assert!((a - b).abs() <= eps, "expected {a} ~= {b} (eps={eps})");
    }

    /// Minimal synthetic `ModelConfig`, unused by `cam_pose`'s own logic
    /// (it derives all shapes from the weight tensors, not `cfg`) but
    /// required by the function signature — same pattern as
    /// `backbone.rs::tests::test_cfg`.
    fn test_cfg() -> ModelConfig {
        ModelConfig {
            arch: "depthanything3".to_string(),
            patch_size: 14,
            image_size: 28,
            embed_dim: 8,
            depth: 1,
            num_heads: 2,
            head_dim: 4,
            mlp_hidden: 16,
            num_register: 0,
            rope_start: -1,
            qknorm_start: -1,
            rope_freq: 100.0,
            ln_eps: 1e-6,
            out_layers: vec![],
            ffn_type: "mlp".to_string(),
            alt_start: -1,
            cat_token: true,
            head_features: 1,
            head_max_depth: 1.0,
            img_mean: [0.0, 0.0, 0.0],
            img_std: [1.0, 1.0, 1.0],
            img_resize_mode: "bilinear".to_string(),
            cam_dim_in: 1,
            head_pos_embed: true,
        }
    }

    fn r_from_pe(pe: &[f32; 9]) -> [[f32; 3]; 3] {
        let (qi, qj, qk, qr) = (pe[3], pe[4], pe[5], pe[6]);
        let s = 2.0f32 / (qi * qi + qj * qj + qk * qk + qr * qr);
        [
            [
                1.0 - s * (qj * qj + qk * qk),
                s * (qi * qj - qk * qr),
                s * (qi * qk + qj * qr),
            ],
            [
                s * (qi * qj + qk * qr),
                1.0 - s * (qi * qi + qk * qk),
                s * (qj * qk - qi * qr),
            ],
            [
                s * (qi * qk - qj * qr),
                s * (qj * qk + qi * qr),
                1.0 - s * (qi * qi + qj * qj),
            ],
        ]
    }

    /// Identity quaternion (qi,qj,qk,qr)=(0,0,0,1) must decode to R=identity:
    /// with T=0, extrinsics must be exactly [I | 0] and intrinsics'
    /// principal point / focal lengths must match the closed-form K
    /// formula. This proves the quaternion math's base case independent of
    /// any model weights.
    #[test]
    fn identity_quaternion_gives_identity_rotation_and_zero_translation() {
        // fov_h = fov_w = 2*atan(1) = pi/2 => tan(fov/2) = tan(pi/4) = 1.
        let fov = 2.0 * (1.0f32).atan();
        let pe = [0.0, 0.0, 0.0, /*q*/ 0.0, 0.0, 0.0, 1.0, fov, fov];
        let (h, w) = (100usize, 200usize);
        let (ext, intr) = decode(&pe, h, w);

        // extrinsics = [R^T | -R^T@T] = [I | 0]
        let expected_ext = [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0,
        ];
        for i in 0..12 {
            approx(ext[i], expected_ext[i], EPS);
        }

        // K = [[fx,0,W/2],[0,fy,H/2],[0,0,1]], fy=(H/2)/tan(fov/2)=H/2, fx=W/2 (tan(pi/4)=1)
        approx(intr[0], w as f32 / 2.0, 1e-3); // fx
        approx(intr[2], w as f32 / 2.0, EPS); // cx
        approx(intr[4], h as f32 / 2.0, 1e-3); // fy
        approx(intr[5], h as f32 / 2.0, EPS); // cy
        approx(intr[8], 1.0, EPS);
        approx(intr[1], 0.0, EPS);
        approx(intr[3], 0.0, EPS);
        approx(intr[6], 0.0, EPS);
        approx(intr[7], 0.0, EPS);
    }

    /// A known 90-degree rotation about Z (quaternion (x,y,z,w) =
    /// (0,0,sin45,cos45)) must produce the textbook Rz(90) matrix
    /// [[0,-1,0],[1,0,0],[0,0,1]]. This is an independently-computable
    /// ground truth (standard robotics/graphics quaternion convention),
    /// verifying the exact sign/index convention of this module's
    /// quat-to-matrix formula (a place transcription errors easily hide).
    #[test]
    fn known_90_degree_z_rotation_quaternion_matches_textbook_matrix() {
        let half = std::f32::consts::FRAC_PI_4; // 45 degrees
        let (qi, qj, qk, qr) = (0.0, 0.0, half.sin(), half.cos());
        let pe = [0.0, 0.0, 0.0, qi, qj, qk, qr, 0.1, 0.1];
        let r = r_from_pe(&pe);
        let expected = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        for row in 0..3 {
            for col in 0..3 {
                approx(r[row][col], expected[row][col], 1e-6);
            }
        }
    }

    /// For ANY unit quaternion, R must be orthogonal: R @ R^T == I. This is
    /// a general algebraic property of the quat->rotation-matrix formula
    /// (independent of any specific numeric ground truth), exercised across
    /// several distinct quaternions to catch any orientation/sign bug that
    /// a single fixed-value test could miss.
    #[test]
    fn rotation_matrix_is_orthogonal_for_various_unit_quaternions() {
        let cases: [[f32; 4]; 4] = [
            [0.0, 0.0, 0.0, 1.0],
            [0.5, 0.5, 0.5, 0.5],
            [0.1, 0.2, 0.3, (1.0 - 0.01 - 0.04 - 0.09f32).sqrt()],
            [-0.2, 0.6, -0.3, (1.0 - 0.04 - 0.36 - 0.09f32).sqrt()],
        ];
        for q in cases {
            let pe = [0.0, 0.0, 0.0, q[0], q[1], q[2], q[3], 0.1, 0.1];
            let r = r_from_pe(&pe);
            // R @ R^T
            let mut rrt = [[0f32; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    rrt[i][j] = (0..3).map(|k| r[i][k] * r[j][k]).sum();
                }
            }
            for i in 0..3 {
                for j in 0..3 {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    approx(rrt[i][j], expected, 1e-4);
                }
            }
        }
    }

    /// `affine_inverse(affine_inverse(X)) == X` for the affine group: this
    /// module's `decode` computes `extrinsics = affine_inverse(c2w)`.
    /// Feeding `extrinsics` back through the SAME affine-inverse formula
    /// (manually, since `extrinsics` is itself a valid `[R|T]`-shaped
    /// affine transform: R^T is a valid rotation, translation is a valid
    /// vector) must reconstruct the original `c2w = [R|T]`. This checks the
    /// `-R^T @ T` sign and index bookkeeping independent of any model
    /// weights.
    #[test]
    fn affine_inverse_round_trip_recovers_original_transform() {
        let half = std::f32::consts::FRAC_PI_4 / 2.0; // 22.5 degrees, arbitrary non-trivial axis
        let (qi, qj, qk) = (0.2, -0.4, 0.3);
        let n = (qi * qi + qj * qj + qk * qk + half.cos() * half.cos()).sqrt();
        let (qi, qj, qk, qr) = (qi / n, qj / n, qk / n, half.cos() / n);
        let (tx, ty, tz) = (1.5f32, -2.25f32, 0.75f32);
        let pe = [tx, ty, tz, qi, qj, qk, qr, 0.3, 0.4];

        let r = r_from_pe(&pe);
        let t = [tx, ty, tz];
        let (ext, _intr) = decode(&pe, 64, 64);

        // ext = [R^T | -R^T@T] per decode(); now invert AGAIN by treating
        // ext as its own [R2|T2] and applying the same affine_inverse.
        let r2 = [
            [ext[0], ext[1], ext[2]],
            [ext[4], ext[5], ext[6]],
            [ext[8], ext[9], ext[10]],
        ];
        let t2 = [ext[3], ext[7], ext[11]];
        let mut r2t = [[0f32; 3]; 3];
        for row in 0..3 {
            for col in 0..3 {
                r2t[row][col] = r2[col][row];
            }
        }
        let mut neg_r2t_t2 = [0f32; 3];
        for row in 0..3 {
            neg_r2t_t2[row] = -(0..3).map(|c| r2t[row][c] * t2[c]).sum::<f32>();
        }

        // r2t must recover the original R, neg_r2t_t2 must recover T.
        for row in 0..3 {
            for col in 0..3 {
                approx(r2t[row][col], r[row][col], 1e-4);
            }
            approx(neg_r2t_t2[row], t[row], 1e-4);
        }
    }

    /// A known FOV (90 degrees, tan(45deg)=1) at a known non-square
    /// resolution must produce exact fx/fy/cx/cy — verifying the intrinsics
    /// formula's H/W assignment (a place a transposition bug would hide,
    /// since H and W both feed the same formula shape).
    #[test]
    fn known_fov_and_resolution_gives_exact_intrinsics() {
        let fov = 2.0 * (1.0f32).atan(); // 90 degrees, tan(fov/2) = 1
        let pe = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, fov, fov];
        let (h, w) = (480usize, 640usize);
        let (_ext, intr) = decode(&pe, h, w);
        approx(intr[0], w as f32 / 2.0, 1e-3); // fx = (W/2)/tan(45deg) = W/2
        approx(intr[4], h as f32 / 2.0, 1e-3); // fy = (H/2)/tan(45deg) = H/2
        approx(intr[2], w as f32 / 2.0, EPS); // cx = W/2
        approx(intr[5], h as f32 / 2.0, EPS); // cy = H/2
    }

    /// Degenerate fov near/at zero must not divide by zero or produce
    /// infinities: `tan(fov/2)` is floored at `1e-6` (matching the C++
    /// `std::max(std::tan(...), 1e-6f)`), so fx/fy must be large-but-finite.
    #[test]
    fn near_zero_fov_is_clamped_not_infinite() {
        let pe = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let (_ext, intr) = decode(&pe, 100, 100);
        assert!(intr[0].is_finite() && intr[0] > 0.0);
        assert!(intr[4].is_finite() && intr[4] > 0.0);
    }

    /// End-to-end `cam_pose` dimension-mismatch validation: an empty or
    /// wrong-length `cam_token` must return `EngineError::CamTokenDimMismatch`
    /// rather than panicking — mirrors the C++'s `if (cam_token.empty() ||
    /// ...) return false;` guard. Uses tiny synthetic weights (not real
    /// model weights) purely to exercise the shape-check path.
    #[test]
    fn cam_pose_rejects_wrong_length_cam_token() {
        let mut weights = Weights::new();
        let d = 4usize;
        let hidden0 = 3usize;
        weights.insert_f32("cam.bb0.weight", vec![0.0; d * hidden0]);
        weights.insert_f32("cam.bb0.bias", vec![0.0; hidden0]);

        let cfg = test_cfg();
        let empty: Vec<f32> = vec![];
        let err = cam_pose(&empty, 10, 10, &cfg, &weights).unwrap_err();
        assert!(matches!(
            err,
            EngineError::CamTokenDimMismatch {
                expected: 4,
                got: 0
            }
        ));

        let wrong = vec![0.0f32; d + 1];
        let err = cam_pose(&wrong, 10, 10, &cfg, &weights).unwrap_err();
        assert!(matches!(
            err,
            EngineError::CamTokenDimMismatch {
                expected: 4,
                got: 5
            }
        ));
    }
}
