//! Core logic behind `da bench`: load a model once via `da_engine::Engine`,
//! run `warmup` untimed inference calls to prime any lazy caches (pos-embed
//! caching in the C++ port shows how much a cold first-call skews numbers —
//! see `../../../benchmarks/BENCHMARK.md`'s "positional-embedding caching"
//! section), then time `repeat` further calls with `std::time::Instant` and
//! report median/p95 latency in milliseconds.
//!
//! Terminology and output format deliberately mirror
//! `../../../benchmarks/BENCHMARK.md`'s existing protocol ("1 warmup +
//! median over N timed iterations", "warm latency") so `da bench` numbers
//! are directly comparable to the numbers already documented there, and so
//! `compare_e2e.sh` (Step 5) can parse both this tool's and the C++ CLI's
//! `--repeat` bench-hook output (`src/cli.hpp`'s `--repeat N`, see
//! `examples/cli/main.cpp`'s `cmd_depth_bench`) with the same kind of
//! `key=value` line.
//!
//! Split out of `main.rs` (same rationale as `infer.rs`): the timing loop
//! and the pure median/p95 math are unit-testable without a real GGUF model
//! or PNG file.

use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use da_engine::{Engine, QuantPref};

/// `da bench` subcommand arguments (plain struct, `clap`-free — see
/// `infer.rs::InferRequest`'s doc comment for why this split exists).
pub struct BenchRequest {
    pub model: PathBuf,
    pub image: PathBuf,
    pub repeat: usize,
    pub warmup: usize,
}

/// Timing result of a `da bench` run: every per-iteration sample (ms) plus
/// the derived median/p95.
#[derive(Debug, Clone, PartialEq)]
pub struct BenchStats {
    pub samples_ms: Vec<f64>,
    pub median_ms: f64,
    pub p95_ms: f64,
}

/// Runs the full `da bench` pipeline: decode `req.image` once, `Engine::load`
/// `req.model` once, run `req.warmup` untimed `Engine::infer` calls, then
/// time `req.repeat` further calls and compute median/p95 via
/// [`compute_stats`].
///
/// Errors if `req.repeat == 0` (nothing to report a median/p95 over) before
/// touching the filesystem or the engine.
pub fn run_bench(req: &BenchRequest) -> Result<BenchStats, Box<dyn Error>> {
    if req.repeat == 0 {
        return Err("--repeat must be >= 1".into());
    }

    let img = image::open(&req.image)?.to_rgb8();
    let w = img.width() as usize;
    let h = img.height() as usize;
    let raw_hwc_u8 = img.into_raw();

    let mut engine = Engine::load(&req.model, QuantPref::PreferF32)?;

    for _ in 0..req.warmup {
        engine.infer_depth(&raw_hwc_u8, h, w)?;
    }

    let mut samples_ms = Vec::with_capacity(req.repeat);
    for _ in 0..req.repeat {
        let t0 = Instant::now();
        engine.infer_depth(&raw_hwc_u8, h, w)?;
        samples_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
    }

    let (median_ms, p95_ms) = compute_stats(&samples_ms);
    Ok(BenchStats {
        samples_ms,
        median_ms,
        p95_ms,
    })
}

/// Linear-interpolation percentile (the same convention `numpy.percentile`'s
/// default `linear` method uses): sorts a copy of `samples`, then
/// interpolates between the two nearest ranks for `p` in `[0, 100]`.
/// Panics if `samples` is empty (callers — [`run_bench`]'s `repeat == 0`
/// check — are expected to guard against that before calling).
fn percentile(sorted: &[f64], p: f64) -> f64 {
    assert!(
        !sorted.is_empty(),
        "percentile of an empty sample set is undefined"
    );
    if sorted.len() == 1 {
        return sorted[0];
    }
    let rank = (p / 100.0) * (sorted.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let frac = rank - lower as f64;
        sorted[lower] + (sorted[upper] - sorted[lower]) * frac
    }
}

/// Pure median/p95 computation over a fixed list of millisecond samples —
/// unit-testable without running any actual inference (see this module's
/// `#[cfg(test)]` block). Does not mutate or require ownership of `samples`.
pub fn compute_stats(samples: &[f64]) -> (f64, f64) {
    let mut sorted: Vec<f64> = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("bench samples must be finite"));
    (percentile(&sorted, 50.0), percentile(&sorted, 95.0))
}

/// Prints the machine-parsable bench report to stdout. The `median_ms=` and
/// `p95_ms=` lines are the load-bearing ones (parsed by `compare_e2e.sh` and
/// by this crate's own `bench_native.rs` integration test); the rest is
/// human-readable context, phrased with `BENCHMARK.md`'s own terminology
/// ("1 warmup + median over N timed iterations").
pub fn print_bench_report(req: &BenchRequest, stats: &BenchStats) {
    println!(
        "da bench: model={} image={}",
        req.model.display(),
        req.image.display()
    );
    println!(
        "protocol: {} warmup + median over {} timed iterations",
        req.warmup, req.repeat
    );
    for (i, ms) in stats.samples_ms.iter().enumerate() {
        println!("iter[{i}]_ms={ms:.3}");
    }
    println!("median_ms={:.3}", stats.median_ms);
    println!("p95_ms={:.3}", stats.p95_ms);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_stats_median_odd_count_is_middle_element() {
        let samples = [10.0, 20.0, 15.0];
        let (median, _) = compute_stats(&samples);
        assert_eq!(
            median, 15.0,
            "median of [10,15,20] should be the middle element"
        );
    }

    #[test]
    fn compute_stats_median_even_count_averages_middle_two() {
        let samples = [10.0, 20.0, 15.0, 25.0];
        // sorted: [10, 15, 20, 25] -> median = avg(15, 20) = 17.5
        let (median, _) = compute_stats(&samples);
        assert_eq!(median, 17.5);
    }

    #[test]
    fn compute_stats_is_order_independent() {
        let a = [5.0, 1.0, 4.0, 2.0, 3.0];
        let b = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(compute_stats(&a), compute_stats(&b));
    }

    #[test]
    fn compute_stats_p95_on_uniform_run_of_ten() {
        // 1..=10 ms samples: p95 (linear interp) = rank 0.95*9 = 8.55 ->
        // interpolate between sorted[8]=9 and sorted[9]=10 -> 9.55.
        let samples: Vec<f64> = (1..=10).map(|v| v as f64).collect();
        let (median, p95) = compute_stats(&samples);
        assert_eq!(median, 5.5);
        assert!((p95 - 9.55).abs() < 1e-9, "p95 should be 9.55, got {p95}");
    }

    #[test]
    fn compute_stats_single_sample_is_its_own_median_and_p95() {
        let samples = [42.0];
        let (median, p95) = compute_stats(&samples);
        assert_eq!(median, 42.0);
        assert_eq!(p95, 42.0);
    }

    #[test]
    fn compute_stats_p95_is_never_below_median_on_nondecreasing_spread() {
        let samples = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0];
        let (median, p95) = compute_stats(&samples);
        assert!(
            p95 >= median,
            "p95 ({p95}) should be >= median ({median}) on a right-skewed sample set"
        );
    }

    #[test]
    fn run_bench_rejects_zero_repeat_before_touching_filesystem() {
        let req = BenchRequest {
            model: PathBuf::from("/nonexistent/model.gguf"),
            image: PathBuf::from("/nonexistent/image.png"),
            repeat: 0,
            warmup: 1,
        };
        let err =
            run_bench(&req).expect_err("repeat=0 should error, not panic or touch the filesystem");
        assert!(
            err.to_string().contains("repeat"),
            "error should mention --repeat: {err}"
        );
    }
}
