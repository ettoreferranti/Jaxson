//! Shows extraction end-to-end with a mocked model response (runs anywhere): a short
//! conversation becomes memory-graph nodes and edges.
//!
//! ```text
//! cargo run --example extract_demo -p jaxson-extract
//! ```

use jaxson_extract::Extractor;
use jaxson_llm::{backends::MockGenerator, Message};
use jaxson_memory::{MemoryGraph, MemoryId};

fn main() {
    let conversation = [
        Message::user("Hi! I'm Ettore and I have a dog named Pixel."),
        Message::assistant("Nice to meet you, Ettore! Pixel sounds lovely."),
        Message::user("Yeah, we go hiking together every weekend."),
    ];

    // In production this is the real llama.cpp model; here a canned reply stands in
    // for what it would emit.
    let mut model = MockGenerator::new(
        r#"{"memories":[
            {"kind":"person","content":"User is named Ettore","confidence":0.98},
            {"kind":"person","content":"Has a dog named Pixel","confidence":0.95},
            {"kind":"preference","content":"Enjoys hiking on weekends","confidence":0.85}
        ],"relations":[
            {"from":0,"to":1,"relation":"knows","weight":0.9},
            {"from":0,"to":2,"relation":"likes","weight":0.8}
        ]}"#,
    );

    let mut n = 0u128;
    let (nodes, edges) = Extractor::default()
        .extract_into_graph(&mut model, &conversation, 0, || {
            let id = MemoryId::from_u128(n);
            n += 1;
            id
        })
        .expect("extraction should succeed");

    let mut graph = MemoryGraph::new();
    for node in &nodes {
        graph.insert_node(node.clone());
    }
    for edge in &edges {
        graph.insert_edge(*edge).expect("endpoints exist");
    }

    println!(
        "Extracted {} memories and {} relations:\n",
        graph.node_count(),
        graph.edge_count()
    );
    for node in &nodes {
        println!(
            "  • [{:?}] {} (confidence {:.2}, {:?})",
            node.kind, node.content, node.confidence, node.provenance
        );
    }
    println!();
    for edge in graph.edges() {
        let from = graph.node(edge.from).unwrap();
        let to = graph.node(edge.to).unwrap();
        println!(
            "  {} --{:?} ({:.1})--> {}",
            from.content, edge.relation, edge.weight, to.content
        );
    }
}
