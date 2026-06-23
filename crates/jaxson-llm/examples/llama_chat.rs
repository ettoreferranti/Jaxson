//! Headless smoke-test for the real `llama.cpp` backend: load a GGUF model and stream
//! one reply. Use this to confirm a model works before wiring it into the app.
//!
//! ```text
//! cargo run -p jaxson-llm --example llama_chat --features llama -- /path/to/model.gguf "Hi!"
//! ```
//!
//! Needs `cmake` + a C/C++ toolchain to build, and a local GGUF model to run.

#[cfg(feature = "llama")]
fn main() {
    use std::io::Write;

    use jaxson_llm::backends::{LlamaConfig, LlamaGenerator};
    use jaxson_llm::{ChatTemplate, GenerationConfig, Message, TextGenerator};

    let mut args = std::env::args().skip(1);
    let model_path = args
        .next()
        .expect("usage: llama_chat <model.gguf> [prompt]");
    let user = args
        .next()
        .unwrap_or_else(|| "Hi! Who are you?".to_string());

    eprintln!("Loading {model_path} …");
    let mut model = LlamaGenerator::load(&LlamaConfig {
        model_path: model_path.into(),
        ..Default::default()
    })
    .expect("failed to load model");

    let prompt = ChatTemplate::ChatMl.render(&[
        Message::system("You are Jaxson, a warm, curious companion."),
        Message::user(user),
    ]);
    let config = GenerationConfig {
        max_tokens: 256,
        ..Default::default()
    };

    print!("Jaxson: ");
    std::io::stdout().flush().ok();
    model
        .generate(&prompt, &config, &mut |piece| {
            print!("{piece}");
            std::io::stdout().flush().ok();
        })
        .expect("generation failed");
    println!();
}

#[cfg(not(feature = "llama"))]
fn main() {
    eprintln!(
        "This example needs the `llama` feature:\n  \
         cargo run -p jaxson-llm --example llama_chat --features llama -- <model.gguf> [prompt]"
    );
}
