//! Conv-Mechanik-Test.
//!
//! Volle numerische Parity gegen die DPT-Head Dumps (`proj0_*`, `convs3_*`,
//! `convt0_*`) braucht die conv/conv_transpose *Gewichte* aus dem Modell-GGUF,
//! das in dieser Umgebung nicht vorliegt (kein `dumps/*.gguf` mit Modellgewichten
//! - nur Aktivierungs-Dumps). Das wird erst in Task 20 (DPT-Head ueber die
//! Engine) scharfgeschaltet.
//!
//! Was hier tatsaechlich verifiziert wird:
//! 1. Mechanische Korrektheit von `conv2d` (im2col+GEMM) und
//!    `conv_transpose2d` gegen selbst geschriebene naive/brute-force
//!    Referenzimplementierungen, inkl. hand-durchgerechneter Kleinstfaelle
//!    (siehe `#[cfg(test)]`-Module in `src/conv.rs` und `src/resample.rs` -
//!    das ist die primaere, scharfe Verifikation dieser Task).
//! 2. Formtreue der Dump-Shapes (`proj0_in` etc.) gegen die im Task-Brief
//!    dokumentierten Shapes, sofern `../dumps/reference.gguf` existiert
//!    (in dieser Umgebung nicht vorhanden -> skip, wie bei den anderen
//!    dump-gated Tests in diesem Crate, z.B. `rope_parity.rs`).

use da_parity::{dumps_path, Dumps};

#[test]
fn conv_1x1_proj0_shape_matches_dump_if_present() {
    let (g, m) = (dumps_path("reference.gguf"), dumps_path("manifest.json"));
    if !g.exists() {
        eprintln!("[skip] no dumps");
        return;
    }
    let d = Dumps::open(&g, &m).unwrap();
    let inp = d.reference("proj0_in").unwrap(); // (1,1536,16,16)
    let out_ref = d.reference("proj0_out").unwrap(); // (1,96,16,16)
    assert_eq!(inp.shape, vec![1, 1536, 16, 16]);
    assert_eq!(out_ref.shape, vec![1, 96, 16, 16]);
    // Gewichte fuer projects[0] kommen aus dem Modell-GGUF (hier nicht
    // verfuegbar) -> keine numerische Assertion hier, siehe Task 20.
}

#[test]
fn conv_3x3_s2p1_convs3_shape_matches_dump_if_present() {
    let (g, m) = (dumps_path("reference.gguf"), dumps_path("manifest.json"));
    if !g.exists() {
        eprintln!("[skip] no dumps");
        return;
    }
    let d = Dumps::open(&g, &m).unwrap();
    let inp = d.reference("convs3_in").ok();
    let out_ref = d.reference("convs3_out").ok();
    if let (Some(inp), Some(out_ref)) = (inp, out_ref) {
        // 768-channel k3s2p1 conv, per Brief.
        assert_eq!(inp.shape[1], 768);
        assert_eq!(out_ref.shape[1], 768);
    } else {
        eprintln!("[skip] convs3_in/out not present in this dump set");
    }
}

#[test]
fn convtranspose_k4s4_convt0_shape_matches_dump_if_present() {
    let (g, m) = (dumps_path("reference.gguf"), dumps_path("manifest.json"));
    if !g.exists() {
        eprintln!("[skip] no dumps");
        return;
    }
    let d = Dumps::open(&g, &m).unwrap();
    let inp = d.reference("convt0_in").ok();
    let out_ref = d.reference("convt0_out").ok();
    if let (Some(inp), Some(out_ref)) = (inp, out_ref) {
        // ConvTranspose k4s4, 96->96 per Brief: spatial should be exactly 4x
        // upsampled (oh = (ih-1)*4+4 = ih*4).
        assert_eq!(inp.shape[1], 96);
        assert_eq!(out_ref.shape[1], 96);
        assert_eq!(out_ref.shape[2], inp.shape[2] * 4);
        assert_eq!(out_ref.shape[3], inp.shape[3] * 4);
    } else {
        eprintln!("[skip] convt0_in/out not present in this dump set");
    }
}
