//! Diagnose memory extraction with a real model: load a GGUF, feed it the exact
//! extraction prompt the agent uses, and print the **raw** model output next to the
//! parsed result. This reveals *why* a model isn't producing memories (e.g. it wrote
//! prose, emitted unterminated reasoning, or used the wrong schema).
//!
//! The first argument is an Ollama model *name* (e.g. `qwen3`, `llama3.1:8b`) — resolved
//! through the same discovery the app uses — or a direct path to a `.gguf` file.
//!
//! ```text
//! cargo run -p jaxson-extract --example extract_probe --features llama -- \
//!     qwen3 "Hi, I'm Ettore and I love hiking with my dog Rex"
//! ```
//!
//! Match the chat format to the model with `JAXSON_TEMPLATE=chatml|llama3|plain`
//! (default chatml) — the same knob the app uses.

#[cfg(feature = "llama")]
fn main() {
    use std::io::Write;

    use jaxson_extract::{extraction_messages, parse_extraction};
    use jaxson_llm::backends::{LlamaConfig, LlamaGenerator};
    use jaxson_llm::{ChatTemplate, GenerationConfig, Message, TextGenerator};

    let mut args = std::env::args().skip(1);
    let model_arg = args
        .next()
        .expect("usage: extract_probe <model-name|model.gguf> [user message …]");
    // Accept a direct .gguf path, or resolve an Ollama model name via discovery.
    let model_path = {
        let as_path = std::path::PathBuf::from(&model_arg);
        if as_path.is_file() {
            as_path
        } else {
            let models = jaxson_llm::ollama::discover();
            let hit = models
                .iter()
                .find(|m| m.name == model_arg)
                .or_else(|| models.iter().find(|m| m.name.starts_with(&model_arg)));
            match hit {
                Some(m) => {
                    eprintln!("Resolved '{}' → {}", m.name, m.path.display());
                    m.path.clone()
                }
                None => {
                    let names: Vec<&str> = models.iter().map(|m| m.name.as_str()).collect();
                    panic!("no model file or Ollama model matching '{model_arg}'. Installed: {names:?}");
                }
            }
        }
    };
    let user = {
        let rest: Vec<String> = args.collect();
        if rest.is_empty() {
            "Hi, I'm Ettore and I love hiking with my dog Rex.".to_string()
        } else {
            rest.join(" ")
        }
    };

    let template = match std::env::var("JAXSON_TEMPLATE").as_deref() {
        Ok("llama3") => ChatTemplate::Llama3,
        Ok("plain") => ChatTemplate::Plain,
        _ => ChatTemplate::ChatMl,
    };

    eprintln!("Loading {} …", model_path.display());
    let mut model = LlamaGenerator::load(&LlamaConfig {
        model_path,
        ..Default::default()
    })
    .expect("failed to load model");

    // The same two-message extraction prompt the agent sends each turn.
    let convo = [Message::user(&user)];
    let prompt = template.render(&extraction_messages(&convo));
    // Mirror Extractor::default(): deterministic (temperature 0). The token budget is
    // overridable (JAXSON_EXTRACT_MAXTOK) to probe how reasoning models behave when given
    // room to finish thinking before the JSON.
    let max_tokens = std::env::var("JAXSON_EXTRACT_MAXTOK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);
    let config = GenerationConfig {
        temperature: 0.0,
        max_tokens,
        ..Default::default()
    };
    eprintln!("(max_tokens = {max_tokens})");

    println!("\n=== user turn ===\n{user}");
    println!("\n=== raw model output ===");
    let raw = model
        .generate(&prompt, &config, &mut |piece| {
            print!("{piece}");
            std::io::stdout().flush().ok();
        })
        .expect("generation failed");

    println!("\n\n=== parse result ===");
    match parse_extraction(&raw) {
        Ok(extraction) if extraction.is_empty() => {
            println!("parsed OK, but EMPTY — the model judged nothing worth remembering.");
        }
        Ok(extraction) => {
            println!("parsed {} memory(ies):", extraction.memories.len());
            for m in &extraction.memories {
                println!("  [{:?}] {} (conf {:.2})", m.kind, m.content, m.confidence);
            }
            println!("and {} relation(s).", extraction.relations.len());
        }
        Err(e) => {
            println!("PARSE FAILED: {e}");
            println!(
                "→ The model didn't return the strict JSON schema. Try a different \
                      model/quant, or JAXSON_TEMPLATE to match its chat format."
            );
        }
    }
}

#[cfg(not(feature = "llama"))]
fn main() {
    eprintln!(
        "This example needs the `llama` feature:\n  \
         cargo run -p jaxson-extract --example extract_probe --features llama -- <model.gguf> [message]"
    );
}
