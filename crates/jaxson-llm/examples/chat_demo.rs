//! Shows the LLM orchestration end-to-end with the deterministic mock backend (so it
//! runs anywhere, no model needed). Demonstrates: assembling persona + memories +
//! history, rendering a chat template, and streaming a reply.
//!
//! ```text
//! cargo run --example chat_demo -p jaxson-llm
//! ```

use jaxson_llm::backends::MockGenerator;
use jaxson_llm::{assemble, ChatTemplate, GenerationConfig, Message, TextGenerator};

fn main() {
    let persona = "You are Jaxson, a kind, curious companion who is just getting to \
                   know its owner. Keep replies short and warm.";
    let memories = [
        "The user's name is Ettore.".to_string(),
        "The user has a dog named Pixel.".to_string(),
    ];
    let history = [
        Message::user("Hey Jaxson!"),
        Message::assistant("Hi Ettore! How's Pixel today?"),
        Message::user("She's great. What should we do this weekend?"),
    ];

    let messages = assemble(persona, &memories, &history);
    let prompt = ChatTemplate::ChatMl.render(&messages);

    println!("=== Rendered prompt (ChatML) ===\n{prompt}\n");

    println!("=== Streaming reply (mock backend) ===");
    let mut model =
        MockGenerator::new("How about a long walk by the river with Pixel, then movie night?");
    let config = GenerationConfig {
        max_tokens: 32,
        ..Default::default()
    };

    let mut full = String::new();
    model
        .generate(&prompt, &config, &mut |piece| {
            print!("{piece}");
            full.push_str(piece);
        })
        .expect("mock generation never fails");
    println!("\n\n=== Final text ===\n{full}");
}
