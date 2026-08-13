mod backbone;
mod config;
mod dpt_head;
mod engine;
mod patch_embed;
mod pos_embed;
mod pose;
mod preprocess;
mod uv_embed;
mod vit_block;

pub use backbone::{
    reference_first_order, select_reference_view_saddle, Backbone, BackboneOutputs,
    MultiViewBackboneOutputs, CAMERA_TOKEN_WEIGHT, VIT_NORM_BIAS, VIT_NORM_WEIGHT,
};
pub use config::{EngineError, ModelConfig};
pub use dpt_head::{
    dpt_head, dpt_head_debug, DepthOut, DptDebug, WinogradFilterCache, HEAD_NORM_EPS,
};
pub use engine::{
    weights_from_gguf, DepthInferOut, Engine, InferOut, MultiViewInferOut, QuantPref, ViewInput,
};
pub use patch_embed::{patch_embed, PATCH_EMBED_BIAS, PATCH_EMBED_WEIGHT};
pub use pos_embed::{
    prepare_tokens, PosEmbedCache, CLS_TOKEN_WEIGHT, POS_EMBED_WEIGHT, REGISTER_TOKENS_WEIGHT,
};
pub use pose::{cam_pose, PoseOut};
pub use preprocess::{preprocess, preprocess_letterbox, LetterboxTransform};
pub use uv_embed::{uv_embed, uv_pos_embed, UvEmbedCache};
pub use vit_block::{vit_block, QK_NORM_EPS};
