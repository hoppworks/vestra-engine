//! Local video and sparse-SfM orchestration for the floorplan pipeline.
//!
//! This module deliberately owns only durable, inspectable sidecar boundaries:
//! FFmpeg extracts evidence frames and COLMAP establishes the global camera
//! trajectory. Dense pose-conditioned DA3 inference and structural fusion are
//! separate stages; a successful preparation report is therefore never
//! presented as a finished floorplan.

use std::error::Error;
use std::fs;
use std::path::PathBuf;

use da_video::{
    availability, colmap_commands, colmap_model_analyzer_command, ffmpeg_extract_frames_command,
    ffprobe_command, CommandRunner, FfmpegFrameExtraction, ProcessRunner, Rational, RunId,
    RunWorkspace, ToolAvailability, ToolPaths, VideoProbe,
};
use serde::Serialize;

pub struct ScanRequest {
    pub video: PathBuf,
    pub output_root: PathBuf,
    pub run_id: String,
    pub sample_rate: Rational,
    pub scale_anchor_mm: Option<u32>,
    pub unscaled: bool,
    pub dry_run: bool,
}

#[derive(Debug, Serialize)]
struct PreparationReport {
    schema: &'static str,
    state: &'static str,
    source_video: String,
    run_id: String,
    scale_anchor_mm: Option<u32>,
    video: VideoReport,
    tools: Vec<ToolReport>,
    next_step: &'static str,
}

#[derive(Debug, Serialize)]
struct VideoReport {
    width: u32,
    height: u32,
    fps: f64,
    rotation_degrees: i32,
    duration_seconds: Option<f64>,
}

#[derive(Debug, Serialize)]
struct ToolReport {
    name: String,
    available: bool,
    detail: String,
}

/// Runs the local video → keyframes → sparse-SfM preparation flow.
///
/// The command refuses to reuse a run identifier because overwriting a prior
/// evidence directory would make a reconstruction non-reproducible. With
/// `dry_run`, only video/sidecar prerequisites are checked and a report is
/// written; FFmpeg and COLMAP do not run.
pub fn run_scan(req: &ScanRequest) -> Result<PathBuf, Box<dyn Error>> {
    if !req.video.is_file() {
        return Err(format!(
            "input video does not exist or is not a file: {}",
            req.video.display()
        )
        .into());
    }
    if req.unscaled && req.scale_anchor_mm.is_some() {
        return Err("--unscaled conflicts with --scale-anchor-mm".into());
    }
    if !req.unscaled && req.scale_anchor_mm.is_none() {
        return Err("--scale-anchor-mm is required unless --unscaled is selected".into());
    }
    if let Some(anchor) = req.scale_anchor_mm {
        if !(500..=5_000).contains(&anchor) {
            return Err("--scale-anchor-mm must be between 500 and 5000".into());
        }
    }
    let run_id = RunId::new(req.run_id.clone())?;
    let workspace = RunWorkspace::new(&req.output_root, run_id);
    if workspace.root().exists() {
        return Err(format!(
            "refusing to reuse existing reconstruction run: {}",
            workspace.root().display()
        )
        .into());
    }

    let runner = ProcessRunner;
    let paths = ToolPaths::default();
    let tools = availability(&runner, &paths);
    let tool_reports = tools
        .iter()
        .map(|(name, state)| match state {
            ToolAvailability::Available { version } => ToolReport {
                name: name.clone(),
                available: true,
                detail: version.clone(),
            },
            ToolAvailability::Unavailable { detail } => ToolReport {
                name: name.clone(),
                available: false,
                detail: detail.clone(),
            },
        })
        .collect::<Vec<_>>();
    if tool_reports.iter().any(|tool| !tool.available) {
        let missing = tool_reports
            .iter()
            .filter(|tool| !tool.available)
            .map(|tool| tool.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!("required local sidecars are unavailable: {missing}").into());
    }

    let probe_output = runner.run(&ffprobe_command(&paths, &req.video))?;
    if probe_output.exit_code != Some(0) {
        return Err(format!("ffprobe failed: {}", probe_output.stderr.trim()).into());
    }
    let probe = VideoProbe::from_ffprobe_json(&probe_output.stdout)?;
    validate_video(&probe)?;

    workspace.ensure_layout()?;
    fs::copy(&req.video, workspace.input_dir().join("walkthrough"))?;
    fs::write(workspace.ffprobe_json(), &probe_output.stdout)?;
    let report_path = workspace.reports_dir().join("preparation-report.json");
    let report = PreparationReport {
        schema: "da-floorplan/preparation-report/v1",
        state: if req.dry_run { "prerequisites-accepted" } else { "sparse-sfm-complete" },
        source_video: req.video.display().to_string(),
        run_id: workspace.run_id().to_string(),
            scale_anchor_mm: req.scale_anchor_mm,
        video: VideoReport {
            width: probe.video.width,
            height: probe.video.height,
            fps: probe.video.average_frame_rate.as_f64(),
            rotation_degrees: probe.video.rotation_degrees,
            duration_seconds: probe.duration_seconds,
        },
        tools: tool_reports,
        next_step: "Run the pinned DA3 pose-conditioned depth sidecar and write a validated frame manifest before confidence-weighted fusion and final export.",
    };
    if !req.dry_run {
        run_checked(
            &runner,
            ffmpeg_extract_frames_command(
                &paths,
                &FfmpegFrameExtraction {
                    source_video: req.video.clone(),
                    output_frames_dir: workspace.frames_dir(),
                    sample_rate: req.sample_rate,
                },
            ),
            "ffmpeg frame extraction",
        )?;
        let colmap = colmap_commands(&paths, &workspace);
        run_checked(
            &runner,
            colmap.feature_extractor,
            "COLMAP feature extraction",
        )?;
        run_checked(
            &runner,
            colmap.sequential_matcher,
            "COLMAP sequential matching",
        )?;
        // Full matching is tractable for a short room walk and prevents two
        // camera loops from being treated as unrelated temporal sequences.
        let expected_frames = fs::read_dir(workspace.frames_dir())?.count();
        if expected_frames <= 160 {
            run_checked(
                &runner,
                colmap.exhaustive_matcher,
                "COLMAP exhaustive matching fallback",
            )?;
        }
        run_checked(&runner, colmap.mapper, "COLMAP mapping")?;
        let analyzer = runner.run(&colmap_model_analyzer_command(
            &paths,
            workspace.colmap_sparse_dir().join("0"),
        ))?;
        if analyzer.exit_code != Some(0) {
            return Err(format!("COLMAP model analysis failed: {}", analyzer.stderr.trim()).into());
        }
        let analyzer_summary = format!("{}\n{}", analyzer.stdout, analyzer.stderr);
        let registered_frames = registered_images(&analyzer_summary)?;
        let registration_fraction = registered_frames as f64 / expected_frames as f64;
        if registration_fraction < 0.90 {
            let registration_percent = registration_fraction * 100.0;
            return Err(format!("capture needs recapture: only {registered_frames}/{expected_frames} sampled frames registered ({registration_percent:.0}%); walk with more sideways movement and keep wall, floor edge, and fixed room features in view").into());
        }
    }
    fs::write(&report_path, serde_json::to_vec_pretty(&report)?)?;
    Ok(report_path)
}

fn registered_images(summary: &str) -> Result<u32, Box<dyn Error>> {
    summary
        .lines()
        .find_map(|line| {
            line.split_once("Registered images:")
                .map(|(_, value)| value)
        })
        .map(str::trim)
        .ok_or_else(|| "COLMAP model analysis did not report registered images".into())
        .and_then(|value| value.parse::<u32>().map_err(|error| error.into()))
}

fn run_checked(
    runner: &impl CommandRunner,
    command: da_video::SidecarCommand,
    label: &str,
) -> Result<(), Box<dyn Error>> {
    let output = runner.run(&command)?;
    if output.exit_code == Some(0) {
        Ok(())
    } else {
        Err(format!("{label} failed: {}", output.stderr.trim()).into())
    }
}

fn validate_video(probe: &VideoProbe) -> Result<(), Box<dyn Error>> {
    let short_edge = probe.video.width.min(probe.video.height);
    let long_edge = probe.video.width.max(probe.video.height);
    if short_edge < 1_080 || long_edge < 1_920 {
        return Err(format!(
            "video is below the minimum Full-HD capture contract (short edge 1080px, long edge 1920px): {}x{}",
            probe.video.width, probe.video.height
        )
        .into());
    }
    if probe.video.average_frame_rate.as_f64() < 24.0 {
        return Err(format!(
            "video is below the minimum 24 fps capture contract: {:.3} fps",
            probe.video.average_frame_rate.as_f64()
        )
        .into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use da_video::{Rational, VideoMetadata};

    fn probe(width: u32, height: u32, fps: Rational) -> VideoProbe {
        VideoProbe {
            format_name: Some("mov".to_owned()),
            duration_seconds: Some(30.0),
            video: VideoMetadata {
                stream_index: 0,
                codec_name: Some("h264".to_owned()),
                width,
                height,
                average_frame_rate: fps,
                time_base: None,
                duration_seconds: Some(30.0),
                frame_count: Some(720),
                rotation_degrees: 0,
            },
        }
    }

    #[test]
    fn capture_contract_accepts_full_hd_at_24_fps() {
        assert!(validate_video(&probe(
            1920,
            1080,
            Rational {
                numerator: 24,
                denominator: 1
            }
        ))
        .is_ok());
        assert!(validate_video(&probe(
            1080,
            1920,
            Rational {
                numerator: 24,
                denominator: 1
            }
        ))
        .is_ok());
    }

    #[test]
    fn capture_contract_rejects_small_or_slow_video() {
        assert!(validate_video(&probe(
            1280,
            1080,
            Rational {
                numerator: 30,
                denominator: 1
            }
        ))
        .is_err());
        assert!(validate_video(&probe(
            1920,
            1080,
            Rational {
                numerator: 23,
                denominator: 1
            }
        ))
        .is_err());
    }

    #[test]
    fn reads_registered_image_count_from_colmap_summary() {
        assert_eq!(
            registered_images("Images: 18\nRegistered images: 17\nPoints: 42").unwrap(),
            17
        );
        assert_eq!(registered_images("I... Registered images: 17").unwrap(), 17);
        assert!(registered_images("Images: 18").is_err());
    }
}
