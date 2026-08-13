//! Safe finalization of an already reconstructed structural envelope.
//!
//! Dense reconstruction remains an explicit sidecar boundary. This module
//! consumes its two durable, reviewable products: measured quality metrics and
//! a structural outer ring. It refuses to write final artifacts when the
//! quality evaluator requires a recapture.

use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use da_floorplan::{Floorplan, ScaledPoint};
use da_quality::{evaluate, CaptureMetrics, QualityAssessment, QualityStatus, QualityThresholds};
use serde::Serialize;

pub struct FinishRequest {
    pub quality_metrics: PathBuf,
    pub outer_ring: PathBuf,
    pub out: PathBuf,
    pub wall_thickness_mm: u32,
    pub wall_height_mm: u32,
}

#[derive(Debug, Serialize)]
struct FinalReport<'a> {
    schema: &'static str,
    coordinate_system: &'static str,
    status: QualityStatus,
    assessment: &'a QualityAssessment,
    artifacts: Artifacts,
}

#[derive(Debug, Serialize)]
struct Artifacts {
    floorplan_svg: Option<&'static str>,
    floorplan_glb: Option<&'static str>,
}

pub fn run_finish(req: &FinishRequest) -> Result<PathBuf, Box<dyn Error>> {
    let metrics: CaptureMetrics = serde_json::from_slice(&fs::read(&req.quality_metrics)?)?;
    let ring: Vec<ScaledPoint> = serde_json::from_slice(&fs::read(&req.outer_ring)?)?;
    let assessment = evaluate(&metrics, &QualityThresholds::default());
    fs::create_dir_all(&req.out)?;
    let report_path = req.out.join("quality-report.json");
    if assessment.status == QualityStatus::RecaptureRequired {
        let report = FinalReport {
            schema: "da-floorplan/quality-report/v1",
            coordinate_system: "Z-up metres; SVG plan coordinates are metres",
            status: assessment.status,
            assessment: &assessment,
            artifacts: Artifacts {
                floorplan_svg: None,
                floorplan_glb: None,
            },
        };
        write_report(&report_path, &report)?;
        return Err(format!(
            "reconstruction requires recapture; diagnostics written to {} and no final floorplan was exported",
            report_path.display()
        )
        .into());
    }
    let plan = Floorplan::new(
        ring,
        f64::from(req.wall_thickness_mm) / 1_000.0,
        f64::from(req.wall_height_mm) / 1_000.0,
    )?;
    fs::write(req.out.join("floorplan.svg"), plan.to_svg())?;
    fs::write(req.out.join("floorplan.glb"), plan.to_glb()?)?;
    let report = FinalReport {
        schema: "da-floorplan/quality-report/v1",
        coordinate_system: "Z-up metres; SVG plan coordinates are metres",
        status: assessment.status,
        assessment: &assessment,
        artifacts: Artifacts {
            floorplan_svg: Some("floorplan.svg"),
            floorplan_glb: Some("floorplan.glb"),
        },
    };
    write_report(&report_path, &report)?;
    Ok(report_path)
}

fn write_report(path: &Path, report: &FinalReport<'_>) -> Result<(), Box<dyn Error>> {
    fs::write(path, serde_json::to_vec_pretty(report)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use da_quality::{
        PrimaryAnchorMetrics, RegistrationMetrics, ReprojectionMetrics, TopologyMetrics,
        VideoMetrics,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn temp_dir() -> PathBuf {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("da_finish_{n}"))
    }

    fn good_metrics() -> CaptureMetrics {
        CaptureMetrics {
            video: VideoMetrics {
                width_px: 1920,
                height_px: 1080,
                frames_per_second: 30.0,
            },
            registration: RegistrationMetrics {
                keyframes_considered: 40,
                keyframes_registered: 40,
                depth_frames_considered: 40,
                depth_frames_fused: 40,
            },
            reprojection: ReprojectionMetrics {
                median_error_px: 0.5,
                p95_error_px: 1.0,
            },
            primary_anchor: PrimaryAnchorMetrics {
                observations: 3,
                relative_spread_percent: 0.5,
            },
            validation_anchor: None,
            topology: TopologyMetrics {
                connected_components: 1,
                non_manifold_edges: 0,
                self_intersections: 0,
                open_boundary_loops: 0,
            },
        }
    }

    fn write_inputs(dir: &Path, metrics: &CaptureMetrics) -> (PathBuf, PathBuf) {
        fs::create_dir_all(dir).unwrap();
        let quality = dir.join("metrics.json");
        let ring = dir.join("ring.json");
        fs::write(&quality, serde_json::to_vec(metrics).unwrap()).unwrap();
        fs::write(&ring, r#"[{"x_m":0.0,"y_m":0.0},{"x_m":4.0,"y_m":0.0},{"x_m":4.0,"y_m":3.0},{"x_m":0.0,"y_m":3.0}]"#).unwrap();
        (quality, ring)
    }

    #[test]
    fn accepted_metrics_write_both_final_exports() {
        let dir = temp_dir();
        let (quality_metrics, outer_ring) = write_inputs(&dir, &good_metrics());
        let out = dir.join("out");
        let report = run_finish(&FinishRequest {
            quality_metrics,
            outer_ring,
            out: out.clone(),
            wall_thickness_mm: 200,
            wall_height_mm: 2700,
        })
        .unwrap();
        assert!(report.is_file());
        assert_eq!(&fs::read(out.join("floorplan.glb")).unwrap()[0..4], b"glTF");
        assert!(fs::read_to_string(out.join("floorplan.svg"))
            .unwrap()
            .contains("data-unit=\"m\""));
        assert!(fs::read_to_string(report)
            .unwrap()
            .contains("scale_anchored"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn rejected_metrics_never_write_final_exports() {
        let dir = temp_dir();
        let mut metrics = good_metrics();
        metrics.registration.keyframes_registered = 2;
        let (quality_metrics, outer_ring) = write_inputs(&dir, &metrics);
        let out = dir.join("out");
        assert!(run_finish(&FinishRequest {
            quality_metrics,
            outer_ring,
            out: out.clone(),
            wall_thickness_mm: 200,
            wall_height_mm: 2700
        })
        .is_err());
        assert!(out.join("quality-report.json").is_file());
        assert!(!out.join("floorplan.glb").exists());
        assert!(!out.join("floorplan.svg").exists());
        let _ = fs::remove_dir_all(dir);
    }
}
