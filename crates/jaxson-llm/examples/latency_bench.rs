//! On-device latency benchmark (F1.1c / NFR-3): measures **time-to-first-token** (TTFT)
//! and **generation throughput** (tokens/sec) for one or more local models, and checks
//! TTFT against the NFR-3 target of < 1.5 s. Use it to pick a model/quantization (OQ-2).
//!
//! ```text
//! cargo run --release -p jaxson-llm --example latency_bench --features llama -- llama3.1 qwen3
//! ```
//!
//! With no arguments it benchmarks every discovered Ollama model. Each model is run
//! `JAXSON_BENCH_RUNS` times (default 3) and the best (warmed-up) figures are reported;
//! `JAXSON_BENCH_TOKENS` (default 64) sets how many tokens to generate per run. Build with
//! `--release` for representative numbers.

#[cfg(feature = "llama")]
fn main() {
    use std::path::PathBuf;
    use std::time::Instant;

    use jaxson_llm::backends::{LlamaConfig, LlamaGenerator};
    use jaxson_llm::{ChatTemplate, GenerationConfig, Message, TextGenerator};

    const TARGET_TTFT_MS: f64 = 1500.0; // NFR-3

    let args: Vec<String> = std::env::args().skip(1).collect();
    let models = jaxson_llm::ollama::discover();
    // Resolve each arg to (display name, gguf path); default to every discovered model.
    let targets: Vec<(String, PathBuf)> = if args.is_empty() {
        models
            .iter()
            .map(|m| (m.name.clone(), m.path.clone()))
            .collect()
    } else {
        args.iter()
            .map(|a| {
                let p = PathBuf::from(a);
                if p.is_file() {
                    (a.clone(), p)
                } else {
                    let m = models
                        .iter()
                        .find(|m| m.name == *a || m.name.starts_with(a))
                        .unwrap_or_else(|| panic!("no model matching '{a}'"));
                    (m.name.clone(), m.path.clone())
                }
            })
            .collect()
    };

    let max_tokens: usize = env_usize("JAXSON_BENCH_TOKENS", 64);
    let runs: usize = env_usize("JAXSON_BENCH_RUNS", 3).max(1);

    println!(
        "NFR-3 target: time-to-first-token < {:.0} ms",
        TARGET_TTFT_MS
    );
    println!("(best of {runs} run(s), {max_tokens} tokens each)\n");
    println!(
        "{:<24} {:>10} {:>12} {:>8}",
        "model", "TTFT(ms)", "gen(tok/s)", "NFR-3"
    );
    println!("{}", "-".repeat(56));

    for (name, path) in targets {
        let mut model = match LlamaGenerator::load(&LlamaConfig {
            model_path: path,
            ..Default::default()
        }) {
            Ok(m) => m,
            Err(e) => {
                println!("{name:<24} load failed: {e}");
                continue;
            }
        };

        let prompt = ChatTemplate::for_model_name(&name).render(&[
            Message::system("You are Jaxson, a fun robot companion."),
            Message::user("Tell me a fun fact about space!"),
        ]);
        let cfg = GenerationConfig {
            max_tokens,
            temperature: 0.0,
            ..Default::default()
        };

        let mut best_ttft = f64::INFINITY;
        let mut best_tps = 0.0f64;
        for _ in 0..runs {
            let start = Instant::now();
            let mut first_token_ms: Option<f64> = None;
            let mut tokens = 0usize;
            model
                .generate(&prompt, &cfg, &mut |_| {
                    if first_token_ms.is_none() {
                        first_token_ms = Some(start.elapsed().as_secs_f64() * 1000.0);
                    }
                    tokens += 1;
                })
                .expect("generation failed");
            let total_s = start.elapsed().as_secs_f64();
            let ttft_ms = first_token_ms.unwrap_or(total_s * 1000.0);
            // Throughput over the post-first-token span (steady-state decode rate).
            let decode_s = (total_s - ttft_ms / 1000.0).max(1e-6);
            let tps = tokens.saturating_sub(1) as f64 / decode_s;
            best_ttft = best_ttft.min(ttft_ms);
            best_tps = best_tps.max(tps);
        }

        let verdict = if best_ttft < TARGET_TTFT_MS {
            "PASS"
        } else {
            "FAIL"
        };
        println!("{name:<24} {best_ttft:>10.0} {best_tps:>12.1} {verdict:>8}");
    }
}

#[cfg(feature = "llama")]
fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

#[cfg(not(feature = "llama"))]
fn main() {
    eprintln!(
        "This example needs the `llama` feature:\n  \
         cargo run --release -p jaxson-llm --example latency_bench --features llama -- [models…]"
    );
}
