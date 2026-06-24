//! Smoke-test the real embedding backend (F1.4b): load a model and print the embedding
//! dimension plus cosine similarities, so you can confirm related sentences score higher
//! than unrelated ones.
//!
//! ```text
//! cargo run -p jaxson-llm --example embed_probe --features llama -- qwen3
//! ```
//!
//! The argument is an Ollama model name (resolved via discovery) or a `.gguf` path.

#[cfg(feature = "llama")]
fn main() {
    use jaxson_llm::backends::{LlamaConfig, LlamaEmbedder};

    let model_arg = std::env::args()
        .nth(1)
        .expect("usage: embed_probe <model-name|model.gguf>");
    let model_path = {
        let as_path = std::path::PathBuf::from(&model_arg);
        if as_path.is_file() {
            as_path
        } else {
            let models = jaxson_llm::ollama::discover();
            let hit = models
                .iter()
                .find(|m| m.name == model_arg)
                .or_else(|| models.iter().find(|m| m.name.starts_with(&model_arg)))
                .unwrap_or_else(|| panic!("no model matching '{model_arg}'"));
            eprintln!("Resolved '{}' → {}", hit.name, hit.path.display());
            hit.path.clone()
        }
    };

    eprintln!("Loading …");
    let embedder = LlamaEmbedder::load(&LlamaConfig {
        model_path,
        // A small window is plenty for short memory snippets and keeps it light.
        n_ctx: 512,
        ..Default::default()
    })
    .expect("failed to load model");

    let embed = |t: &str| embedder.embed(t).expect("embed failed");
    let dog = embed("I love hiking with my dog");
    let dog2 = embed("Walking my puppy in the hills is the best");
    let taxes = embed("The quarterly tax filing deadline is approaching");

    // Vectors are L2-normalized, so cosine is just the dot product.
    let cos = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();

    println!("embedding dim: {}", dog.len());
    println!("cos(dog, similar dog) = {:.3}", cos(&dog, &dog2));
    println!("cos(dog, taxes)       = {:.3}", cos(&dog, &taxes));
    println!(
        "{}",
        if cos(&dog, &dog2) > cos(&dog, &taxes) {
            "OK: related sentences are closer than unrelated ones."
        } else {
            "WARN: similarity ordering looks off for this model."
        }
    );
}

#[cfg(not(feature = "llama"))]
fn main() {
    eprintln!(
        "This example needs the `llama` feature:\n  \
         cargo run -p jaxson-llm --example embed_probe --features llama -- <model.gguf>"
    );
}
