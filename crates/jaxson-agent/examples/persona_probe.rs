//! Tune Jaxson's personality: run a few canned turns through a real model with the
//! default persona and print the replies, so you can feel the voice without launching the
//! GUI. Edit `jaxson_agent::DEFAULT_PERSONA` and re-run to iterate.
//!
//! ```text
//! cargo run -p jaxson-agent --example persona_probe --features llama -- llama3.1
//! ```
//!
//! The argument is an Ollama model name (resolved via discovery) or a `.gguf` path;
//! defaults to the first installed `llama3.1`.

#[cfg(feature = "llama")]
fn main() {
    use jaxson_agent::{Agent, AgentConfig, HashEmbedder, DEFAULT_PERSONA};
    use jaxson_llm::backends::{LlamaConfig, LlamaGenerator};
    use jaxson_llm::{ChatTemplate, GenerationConfig};

    let arg = std::env::args().nth(1);
    let models = jaxson_llm::ollama::discover();
    let (model_path, template) = match &arg {
        Some(a) if std::path::Path::new(a).is_file() => {
            (std::path::PathBuf::from(a), ChatTemplate::ChatMl)
        }
        Some(a) => {
            let m = models
                .iter()
                .find(|m| m.name == *a || m.name.starts_with(a))
                .unwrap_or_else(|| panic!("no model matching '{a}'"));
            (m.path.clone(), ChatTemplate::for_model_name(&m.name))
        }
        None => {
            let m = models
                .iter()
                .find(|m| m.name.starts_with("llama3.1"))
                .expect("no llama3.1 installed; pass a model name");
            (m.path.clone(), ChatTemplate::for_model_name(&m.name))
        }
    };

    eprintln!("Loading {} …", model_path.display());
    let mut model = LlamaGenerator::load(&LlamaConfig {
        model_path,
        ..Default::default()
    })
    .expect("load model");
    let mut agent = Agent::new(DEFAULT_PERSONA).with_config(AgentConfig {
        template,
        gen_config: GenerationConfig {
            stop: template
                .stop_tokens()
                .iter()
                .map(|s| s.to_string())
                .collect(),
            max_tokens: 120,
            ..Default::default()
        },
        ..Default::default()
    });
    let embedder = HashEmbedder::default();

    let script = [
        "Hi!",
        "I had a rough day at work today.",
        "I love playing guitar",
    ];
    for (i, line) in script.iter().enumerate() {
        let turn = agent
            .respond(&mut model, &embedder, i as i64, line)
            .expect("turn");
        println!("YOU:    {line}\nJAXSON: {}\n", turn.reply);
    }
}

#[cfg(not(feature = "llama"))]
fn main() {
    eprintln!(
        "This example needs the `llama` feature:\n  \
         cargo run -p jaxson-agent --example persona_probe --features llama -- <model>"
    );
}
