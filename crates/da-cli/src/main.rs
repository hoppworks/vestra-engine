//! `da`: the depth-anything.cpp-rs CLI. Its active subcommands are `infer` (loads a
//! GGUF model, runs inference on an image, writes depth + camera pose to
//! disk) and `bench` (loads a model once, times N inference calls, reports
//! median/p95 latency — see `bench.rs` and `../scripts/compare_e2e.sh`).

mod bench;
mod infer;

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};

use crate::bench::{print_bench_report, run_bench, BenchRequest};
use crate::infer::{run_infer, InferRequest};

#[derive(Parser, Debug)]
#[command(name = "da", about = "depth-anything.cpp-rs CLI")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Run depth + pose inference on a single image.
    Infer(InferArgs),
    /// Load a model once, time N inference calls, report median/p95 latency.
    Bench(BenchArgs),
}

#[derive(clap::Args, Debug, PartialEq, Eq)]
pub struct InferArgs {
    /// Path to the GGUF model file.
    #[arg(long)]
    pub model: PathBuf,
    /// Path to the input image (any format the `image` crate can decode).
    #[arg(long)]
    pub image: PathBuf,
    /// Output depth path. `.pfm` writes a raw float PFM; anything else
    /// (e.g. `.png`) writes a min-max normalized 8-bit grayscale PNG.
    #[arg(long)]
    pub out_depth: PathBuf,
    /// Output pose JSON path (`{ "extrinsics": [[..]], "intrinsics": [[..]] }`).
    #[arg(long)]
    pub out_pose: PathBuf,
}

#[derive(clap::Args, Debug, PartialEq, Eq)]
pub struct BenchArgs {
    /// Path to the GGUF model file.
    #[arg(long)]
    pub model: PathBuf,
    /// Path to the input image (any format the `image` crate can decode).
    #[arg(long)]
    pub image: PathBuf,
    /// Number of timed inference iterations (median/p95 computed over these).
    #[arg(long, default_value_t = 10)]
    pub repeat: usize,
    /// Number of untimed warmup iterations run before the timed ones.
    #[arg(long, default_value_t = 1)]
    pub warmup: usize,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command {
        Command::Infer(args) => {
            let req = InferRequest {
                model: args.model,
                image: args.image,
                out_depth: args.out_depth,
                out_pose: args.out_pose,
            };
            match run_infer(&req) {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("error: {e}");
                    ExitCode::FAILURE
                }
            }
        }
        Command::Bench(args) => {
            let req = BenchRequest {
                model: args.model,
                image: args.image,
                repeat: args.repeat,
                warmup: args.warmup,
            };
            match run_bench(&req) {
                Ok(stats) => {
                    print_bench_report(&req, &stats);
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("error: {e}");
                    ExitCode::FAILURE
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infer_parses_all_required_args() {
        let cli = Cli::try_parse_from([
            "da",
            "infer",
            "--model",
            "model.gguf",
            "--image",
            "in.png",
            "--out-depth",
            "out.pfm",
            "--out-pose",
            "pose.json",
        ])
        .expect("well-formed argv should parse");

        let Command::Infer(args) = cli.command else {
            panic!("expected Command::Infer")
        };
        assert_eq!(args.model, PathBuf::from("model.gguf"));
        assert_eq!(args.image, PathBuf::from("in.png"));
        assert_eq!(args.out_depth, PathBuf::from("out.pfm"));
        assert_eq!(args.out_pose, PathBuf::from("pose.json"));
    }

    #[test]
    fn infer_missing_required_arg_fails_to_parse() {
        // --out-pose is missing.
        let result = Cli::try_parse_from([
            "da",
            "infer",
            "--model",
            "model.gguf",
            "--image",
            "in.png",
            "--out-depth",
            "out.pfm",
        ]);
        assert!(
            result.is_err(),
            "missing required --out-pose should fail to parse"
        );
    }

    #[test]
    fn bench_parses_with_defaults() {
        let cli =
            Cli::try_parse_from(["da", "bench", "--model", "model.gguf", "--image", "in.png"])
                .expect("well-formed argv should parse");

        let Command::Bench(args) = cli.command else {
            panic!("expected Command::Bench")
        };
        assert_eq!(args.model, PathBuf::from("model.gguf"));
        assert_eq!(args.image, PathBuf::from("in.png"));
        assert_eq!(args.repeat, 10, "--repeat should default to 10");
        assert_eq!(args.warmup, 1, "--warmup should default to 1");
    }

    #[test]
    fn bench_parses_explicit_repeat_and_warmup() {
        let cli = Cli::try_parse_from([
            "da",
            "bench",
            "--model",
            "model.gguf",
            "--image",
            "in.png",
            "--repeat",
            "2",
            "--warmup",
            "0",
        ])
        .expect("well-formed argv should parse");

        let Command::Bench(args) = cli.command else {
            panic!("expected Command::Bench")
        };
        assert_eq!(args.repeat, 2);
        assert_eq!(args.warmup, 0);
    }

    #[test]
    fn bench_missing_required_arg_fails_to_parse() {
        let result = Cli::try_parse_from(["da", "bench", "--model", "model.gguf"]);
        assert!(
            result.is_err(),
            "missing required --image should fail to parse"
        );
    }

    #[test]
    fn missing_subcommand_fails_to_parse() {
        let result = Cli::try_parse_from(["da"]);
        assert!(result.is_err(), "no subcommand should fail to parse");
    }

    #[test]
    fn unknown_subcommand_fails_to_parse() {
        let result = Cli::try_parse_from(["da", "bogus"]);
        assert!(result.is_err(), "unknown subcommand should fail to parse");
    }
}
