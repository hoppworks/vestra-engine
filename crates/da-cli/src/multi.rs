//! Multi-view oracle-compatible CLI path.

use std::{error::Error, path::PathBuf};

use da_engine::{Engine, QuantPref, ViewInput};

use crate::infer::{write_pfm, write_pose_json};

pub struct MultiInferRequest {
    pub model: PathBuf,
    pub images: Vec<PathBuf>,
    pub out_prefix: PathBuf,
}

/// Mirrors the pinned C++ CLI's multi-view file contract:
/// `<prefix>_view<N>.pfm` plus `<prefix>_view<N>.json` for every input view.
pub fn run_multi_infer(request: &MultiInferRequest) -> Result<(), Box<dyn Error>> {
    if request.images.len() < 2 {
        return Err("infer-multi requires at least two --image arguments".into());
    }
    let images = request
        .images
        .iter()
        .map(|path| {
            let image = image::open(path)?.to_rgb8();
            Ok::<_, image::ImageError>((
                image.width() as usize,
                image.height() as usize,
                image.into_raw(),
            ))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let inputs = images
        .iter()
        .map(|(width, height, rgb)| ViewInput {
            rgb_hwc_u8: rgb,
            h: *height,
            w: *width,
        })
        .collect::<Vec<_>>();
    let mut engine = Engine::load(&request.model, QuantPref::PreferF32)?;
    let output = engine.infer_multi_view(&inputs)?;
    for (index, view) in output.views.iter().enumerate() {
        let prefix = request.out_prefix.to_string_lossy();
        let depth = format!("{prefix}_view{index}.pfm");
        let pose = format!("{prefix}_view{index}.json");
        write_pfm(std::path::Path::new(&depth), &view.depth, view.w, view.h)?;
        write_pose_json(
            std::path::Path::new(&pose),
            &view.extrinsics,
            &view.intrinsics,
        )?;
    }
    Ok(())
}
