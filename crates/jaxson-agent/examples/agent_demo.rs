//! The whole conversation loop end-to-end with deterministic backends (no model
//! needed): retrieval + reply + learning + state, turn by turn.
//!
//! ```text
//! cargo run --example agent_demo -p jaxson-agent
//! ```

use jaxson_agent::{Agent, HashEmbedder};
use jaxson_llm::backends::ScriptedGenerator;

fn main() {
    let embedder = HashEmbedder::default();

    // Each turn the scripted model returns: (1) a chat reply, then (2) extraction JSON.
    let mut model = ScriptedGenerator::new([
        "Hi! I'm Jaxson. What's your name?".to_string(),
        json(&[("person", "User is named Ettore")]),
        "Nice to meet you, Ettore! Got any pets?".to_string(),
        json(&[("person", "Has a dog named Pixel")]),
        "Pixel sounds great — what do you two like to do?".to_string(),
        json(&[("preference", "Enjoys hiking with Pixel on weekends")]),
        "A hike this weekend sounds perfect!".to_string(),
        json(&[]),
    ]);

    let user_turns = [
        "Hello there!",
        "I'm Ettore.",
        "I have a dog named Pixel.",
        "What should Pixel and I do about hiking this weekend?",
    ];

    let mut agent =
        Agent::new("You are Jaxson, a warm, curious companion getting to know its owner.");

    for (i, input) in user_turns.iter().enumerate() {
        let turn = agent
            .respond(&mut model, &embedder, i as i64, input)
            .expect("turn should succeed");
        println!("You:     {input}");
        println!("Jaxson:  {}", turn.reply);
        println!(
            "         [learned {} | retrieved {} | familiarity {:.2} | onboarding {} | mood {:?}]\n",
            turn.learned,
            turn.retrieved,
            agent.state().familiarity(),
            agent.state().should_prioritize_onboarding(),
            turn.mood.dominant_emotion(),
        );
    }

    println!(
        "Jaxson now remembers {} things:",
        agent.graph().node_count()
    );
    for node in agent.graph().nodes() {
        println!("  • [{:?}] {}", node.kind, node.content);
    }
}

/// Build an extraction-response JSON with the given `(kind, content)` memories.
fn json(memories: &[(&str, &str)]) -> String {
    let items: Vec<String> = memories
        .iter()
        .map(|(kind, content)| {
            format!(r#"{{"kind":"{kind}","content":"{content}","confidence":0.9}}"#)
        })
        .collect();
    format!(r#"{{"memories":[{}],"relations":[]}}"#, items.join(","))
}
