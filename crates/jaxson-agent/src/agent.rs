use jaxson_affect::{action_sentiment, analyze, AffectEngine, Sentiment};
use jaxson_core::{MoodVector, RelationshipEvent, RelationshipState};
use jaxson_extract::Extractor;
use jaxson_llm::{assemble, ChatTemplate, GenerationConfig, Message, TextGenerator};
use jaxson_memory::{retrieve, MemoryGraph, MemoryId, RetrievalParams};

use crate::curiosity;
use crate::embedder::Embedder;
use crate::error::AgentError;

/// Tuning for the conversation loop.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Chat template matching the loaded model.
    pub template: ChatTemplate,
    /// Decoding config for the chat reply.
    pub gen_config: GenerationConfig,
    /// How relevant memories are retrieved each turn.
    pub retrieval: RetrievalParams,
    /// How many of the most recent messages feed the extraction pass.
    pub history_window: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        AgentConfig {
            template: ChatTemplate::ChatMl,
            gen_config: GenerationConfig::default(),
            retrieval: RetrievalParams::default(),
            history_window: 4,
        }
    }
}

/// The result of one conversation turn.
#[derive(Debug, Clone, PartialEq)]
pub struct Turn {
    /// Jaxson's reply text.
    pub reply: String,
    /// Current mood (drives the face), updated by the affect engine from the
    /// exchange's sentiment.
    pub mood: MoodVector,
    /// How many memories were learned this turn.
    pub learned: usize,
    /// How many existing memories were retrieved into context this turn.
    pub retrieved: usize,
}

/// Jaxson's conversation agent: it ties memory retrieval, the LLM, extraction, and the
/// relationship state machine into a single [`respond`](Agent::respond) call.
///
/// The model and embedder are passed in per turn (so the same agent works with the
/// mock or the real backends), and persistence is left to the caller — construct with
/// [`with_graph`](Agent::with_graph) from a loaded store and persist
/// [`graph`](Agent::graph) afterwards.
pub struct Agent {
    persona: String,
    state: RelationshipState,
    graph: MemoryGraph,
    history: Vec<Message>,
    extractor: Extractor,
    affect: AffectEngine,
    config: AgentConfig,
}

impl Agent {
    /// A fresh agent with an empty memory graph.
    pub fn new(persona: impl Into<String>) -> Self {
        Agent::with_graph(persona, MemoryGraph::new())
    }

    /// An agent seeded with an existing (e.g. persisted) memory graph.
    pub fn with_graph(persona: impl Into<String>, graph: MemoryGraph) -> Self {
        Agent {
            persona: persona.into(),
            state: RelationshipState::INITIAL,
            graph,
            history: Vec::new(),
            extractor: Extractor::default(),
            affect: AffectEngine::default(),
            config: AgentConfig::default(),
        }
    }

    /// Override the configuration (builder style).
    pub fn with_config(mut self, config: AgentConfig) -> Self {
        self.config = config;
        self
    }

    pub fn state(&self) -> &RelationshipState {
        &self.state
    }

    pub fn graph(&self) -> &MemoryGraph {
        &self.graph
    }

    /// Mutable access to the memory graph, for curation (the memory inspector).
    pub fn graph_mut(&mut self) -> &mut MemoryGraph {
        &mut self.graph
    }

    /// The chat template currently in use.
    pub fn template(&self) -> ChatTemplate {
        self.config.template
    }

    /// Switch the chat template (e.g. when loading a different model), keeping the stop
    /// tokens consistent. Extraction follows the same template (synced each turn).
    pub fn set_template(&mut self, template: ChatTemplate) {
        self.config.template = template;
        self.config.gen_config.stop = template
            .stop_tokens()
            .iter()
            .map(|s| s.to_string())
            .collect();
    }

    pub fn history(&self) -> &[Message] {
        &self.history
    }

    /// Current mood for the face.
    pub fn mood(&self) -> MoodVector {
        self.state.mood()
    }

    /// Run one turn: retrieve relevant memories, reply, learn from the exchange, and
    /// update the relationship state. `now` is the timestamp stamped on new memories.
    pub fn respond(
        &mut self,
        model: &mut dyn TextGenerator,
        embedder: &dyn Embedder,
        now: i64,
        user_input: &str,
    ) -> Result<Turn, AgentError> {
        self.history.push(Message::user(user_input));

        // Retrieve memories relevant to what the user just said.
        let query = embedder.embed(user_input);
        let hits = retrieve(&self.graph, &query, &self.config.retrieval);
        let memories: Vec<String> = hits
            .iter()
            .filter_map(|h| self.graph.node(h.id))
            .map(|node| node.content.clone())
            .collect();
        let retrieved = memories.len();

        // Build the prompt and generate the reply.
        let prompt =
            self.config
                .template
                .render(&assemble(&self.system_prompt(), &memories, &self.history));
        let raw = model
            .complete(&prompt, &self.config.gen_config)
            .map_err(|e| AgentError::Generation(e.to_string()))?;
        // Clean reasoning + chat-control tokens, then read Jaxson's expressed feeling
        // from its own *action* cues before removing them from the displayed reply.
        let cleaned = jaxson_llm::clean_output(&raw);
        let expressed = action_sentiment(&cleaned);
        let reply = jaxson_llm::strip_actions(&cleaned);
        self.history.push(Message::assistant(reply.clone()));

        // Learn from the latest exchange.
        let learned = self.learn(model, embedder, now)?;

        // Familiarity grows with each fact learned.
        for _ in 0..learned {
            self.state = self.state.apply(RelationshipEvent::LearnedFact);
        }

        // Mood: prefer Jaxson's expressed feeling (its action cues); otherwise fall back
        // to the sentiment of what the user said. Smoothing lives in the state machine.
        let sentiment = if expressed == Sentiment::NEUTRAL {
            analyze(user_input)
        } else {
            expressed
        };
        let target = self.affect.target_mood(&self.state, sentiment);
        self.state = self.state.apply(RelationshipEvent::MoodObserved {
            target,
            fraction: self.affect.responsiveness,
        });

        Ok(Turn {
            reply,
            mood: self.state.mood(),
            learned,
            retrieved,
        })
    }

    /// Extract memories from the recent window and merge them (with embeddings) into
    /// the graph. Returns how many nodes were learned.
    fn learn(
        &mut self,
        model: &mut dyn TextGenerator,
        embedder: &dyn Embedder,
        now: i64,
    ) -> Result<usize, AgentError> {
        // Extraction is best-effort: a model that returns malformed/empty JSON simply
        // means nothing was learned this turn, not a failed turn. (Logging this is F1.12.)
        // Extract with the same chat template the model is using.
        self.extractor.template = self.config.template;
        let window = self.recent_window();
        let extraction = match self.extractor.extract(model, &window) {
            Ok(extraction) => extraction,
            Err(_) => return Ok(0),
        };
        let (nodes, edges) =
            match extraction.into_graph(now, self.extractor.provenance, MemoryId::new) {
                Ok(graph) => graph,
                Err(_) => return Ok(0),
            };
        let mut learned = 0;
        for node in nodes {
            // Don't store a memory we already have (dedup by content).
            if self.graph.contains_content(&node.content) {
                continue;
            }
            let embedding = embedder.embed(&node.content);
            self.graph.insert_node(node.with_embedding(embedding));
            learned += 1;
        }
        for edge in edges {
            // Ignore edges whose endpoints aren't present (e.g. a deduped node).
            let _ = self.graph.insert_edge(edge);
        }
        Ok(learned)
    }

    /// Persona plus state-driven behavior hints. Proactive curiosity (when and what to
    /// ask) is gated by familiarity and what's already known — see [`crate::curiosity`].
    fn system_prompt(&self) -> String {
        let mut prompt = self.persona.clone();
        if let Some(hint) = curiosity::proactive_hint(&self.state, &self.graph) {
            prompt.push_str("\n\n");
            prompt.push_str(&hint);
        }
        prompt
    }

    /// The most recent messages, for the extraction pass.
    fn recent_window(&self) -> Vec<Message> {
        let window = self.config.history_window.max(1);
        let start = self.history.len().saturating_sub(window);
        self.history[start..].to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedder::HashEmbedder;
    use jaxson_llm::backends::ScriptedGenerator;

    /// Records the prompts and stop tokens it's asked to generate from (returns empty).
    #[derive(Default)]
    struct RecordingGenerator {
        prompts: Vec<String>,
        stops: Vec<Vec<String>>,
    }

    impl TextGenerator for RecordingGenerator {
        fn generate(
            &mut self,
            prompt: &str,
            config: &GenerationConfig,
            _on_token: &mut dyn FnMut(&str),
        ) -> Result<String, jaxson_llm::LlmError> {
            self.prompts.push(prompt.to_string());
            self.stops.push(config.stop.clone());
            Ok(String::new())
        }
    }

    #[test]
    fn set_template_drives_chat_extraction_and_stop() {
        let mut rec = RecordingGenerator::default();
        let embedder = HashEmbedder::default();
        let mut agent = Agent::new("persona");
        agent.set_template(ChatTemplate::Llama3);
        assert_eq!(agent.template(), ChatTemplate::Llama3);

        agent.respond(&mut rec, &embedder, 0, "hi").unwrap();

        // prompts[0] = chat, prompts[1] = extraction; both use the Llama-3 format.
        assert!(rec.prompts[0].starts_with("<|begin_of_text|>"));
        assert!(rec.prompts[1].contains("<|start_header_id|>"));
        // The chat config carries the Llama-3 stop token.
        assert!(rec.stops[0].iter().any(|s| s == "<|eot_id|>"));
    }

    fn extraction_json(content: &str) -> String {
        format!(
            r#"{{"memories":[{{"kind":"fact","content":"{content}","confidence":0.9}}],"relations":[]}}"#
        )
    }

    #[test]
    fn graph_mut_mutates_the_agents_own_graph() {
        use jaxson_memory::{MemoryKind, MemoryNode, Provenance};
        let mut agent = Agent::new("persona");
        agent.graph_mut().insert_node(MemoryNode::new(
            MemoryId::from_u128(1),
            MemoryKind::Fact,
            "x",
            0,
            Provenance::StatedByUser,
            0.5,
        ));
        assert_eq!(agent.graph().node_count(), 1);
    }

    #[test]
    fn fresh_agent_starts_blank_and_onboarding() {
        let agent = Agent::new("You are Jaxson.");
        assert!(agent.graph().is_empty());
        assert_eq!(agent.state().familiarity(), 0.0);
        // A brand-new Jaxson leads with a warm getting-to-know-you question.
        assert!(agent.system_prompt().contains("Warmly ask"));
    }

    #[test]
    fn a_turn_replies_learns_and_records_history() {
        let mut model =
            ScriptedGenerator::new(["Nice to meet you!", &extraction_json("User likes hiking")]);
        let embedder = HashEmbedder::default();
        let mut agent = Agent::new("You are Jaxson.");

        let turn = agent
            .respond(&mut model, &embedder, 100, "Hi, I love hiking")
            .unwrap();

        assert_eq!(turn.reply, "Nice to meet you!");
        assert_eq!(turn.learned, 1);
        assert_eq!(turn.retrieved, 0); // graph was empty when retrieval ran
        assert_eq!(agent.graph().node_count(), 1);
        assert_eq!(agent.history().len(), 2); // user + assistant
        assert!(agent.state().familiarity() > 0.0);
    }

    #[test]
    fn positive_input_brightens_mood() {
        let mut model = ScriptedGenerator::new(["yay", &extraction_json("nothing notable")]);
        let embedder = HashEmbedder::default();
        let mut agent = Agent::new("You are Jaxson.");
        let turn = agent
            .respond(&mut model, &embedder, 0, "I really love this!")
            .unwrap();
        assert!(turn.mood.valence() > 0.0);
        assert_eq!(turn.mood, agent.mood());
    }

    #[test]
    fn later_turns_retrieve_earlier_memories() {
        let embedder = HashEmbedder::default();
        let mut model = ScriptedGenerator::new([
            "Hello!".to_string(),
            extraction_json("User enjoys hiking on weekends"),
            "Hiking is wonderful!".to_string(),
            extraction_json("Nothing new"),
        ]);
        let mut agent = Agent::new("You are Jaxson.");

        agent
            .respond(&mut model, &embedder, 1, "I love hiking")
            .unwrap();
        let turn2 = agent
            .respond(&mut model, &embedder, 2, "tell me about hiking")
            .unwrap();

        // The hiking memory from turn 1 is retrieved for turn 2 (shared vocabulary).
        assert!(turn2.retrieved >= 1);
    }

    #[test]
    fn reasoning_blocks_are_stripped_from_replies() {
        let mut model =
            ScriptedGenerator::new(["<think>plan</think>Hello!", &extraction_json("nothing")]);
        let embedder = HashEmbedder::default();
        let mut agent = Agent::new("persona");
        let turn = agent.respond(&mut model, &embedder, 0, "hi").unwrap();
        assert_eq!(turn.reply, "Hello!");
    }

    #[test]
    fn does_not_relearn_a_duplicate_memory() {
        let json = extraction_json("User is named Ettore");
        let mut model = ScriptedGenerator::new(["hi", &json, "hi again", &json]);
        let embedder = HashEmbedder::default();
        let mut agent = Agent::new("persona");
        let first = agent.respond(&mut model, &embedder, 0, "a").unwrap();
        let second = agent.respond(&mut model, &embedder, 1, "b").unwrap();
        assert_eq!(first.learned, 1);
        assert_eq!(second.learned, 0); // same memory not stored again
        assert_eq!(agent.graph().node_count(), 1);
    }

    #[test]
    fn action_cues_are_stripped_and_drive_mood() {
        let mut model =
            ScriptedGenerator::new(["*ears perked up* Hi!", &extraction_json("nothing")]);
        let embedder = HashEmbedder::default();
        let mut agent = Agent::new("persona");
        let turn = agent.respond(&mut model, &embedder, 0, "hello").unwrap();
        assert_eq!(turn.reply, "Hi!"); // the *action* is removed from the text
                                       // The perked-up cue (not the neutral user input) drives a clearly positive mood.
        assert!(turn.mood.valence() > 0.2);
    }

    #[test]
    fn chat_control_tokens_are_stripped_from_replies() {
        let mut model = ScriptedGenerator::new(["Hello!<|im_end|>", &extraction_json("nothing")]);
        let embedder = HashEmbedder::default();
        let mut agent = Agent::new("persona");
        let turn = agent.respond(&mut model, &embedder, 0, "hi").unwrap();
        assert_eq!(turn.reply, "Hello!");
    }

    #[test]
    fn bad_extraction_json_is_non_fatal() {
        // The chat reply succeeds; the extraction step returns non-JSON. The turn must
        // still succeed, just with nothing learned.
        let mut model = ScriptedGenerator::new(["a reply", "not json at all"]);
        let embedder = HashEmbedder::default();
        let mut agent = Agent::new("persona");
        let turn = agent.respond(&mut model, &embedder, 0, "hello").unwrap();
        assert_eq!(turn.reply, "a reply");
        assert_eq!(turn.learned, 0);
        assert!(agent.graph().is_empty());
    }

    #[test]
    fn extraction_with_a_bad_relation_index_is_non_fatal() {
        // Valid JSON, but the relation points past the memory list — into_graph errors,
        // and the turn still succeeds with nothing learned (no partial graph).
        let json = r#"{"memories":[{"kind":"fact","content":"x","confidence":0.9}],"relations":[{"from":0,"to":9,"relation":"knows","weight":0.5}]}"#;
        let mut model = ScriptedGenerator::new(["a reply", json]);
        let embedder = HashEmbedder::default();
        let mut agent = Agent::new("persona");
        let turn = agent.respond(&mut model, &embedder, 0, "hello").unwrap();
        assert_eq!(turn.learned, 0);
        assert!(agent.graph().is_empty());
    }

    #[test]
    fn recent_window_returns_the_last_messages() {
        let mut agent = Agent::new("persona"); // default history_window = 4
        for i in 0..6 {
            agent.history.push(Message::user(format!("m{i}")));
        }
        let window = agent.recent_window();
        assert_eq!(window.len(), 4);
        assert_eq!(window[0].content, "m2");
        assert_eq!(window[3].content, "m5");
    }

    #[test]
    fn onboarding_lead_gives_way_to_casual_curiosity_as_familiarity_grows() {
        let embedder = HashEmbedder::default();
        let mut replies = Vec::new();
        for i in 0..10 {
            replies.push("ok".to_string());
            replies.push(extraction_json(&format!("fact number {i}")));
        }
        let mut model = ScriptedGenerator::new(replies);
        let mut agent = Agent::new("You are Jaxson.");

        // Fresh: leads every turn with a warm onboarding question.
        assert!(agent.system_prompt().contains("Warmly ask"));
        for i in 0..10 {
            agent
                .respond(&mut model, &embedder, i, "tell me more")
                .unwrap();
        }
        // Now acquainted; the lead is gone. Only Fact memories were learned, so other
        // topics remain open and Jaxson stays gently curious rather than going silent.
        assert!(agent.state().familiarity() > RelationshipState::ONBOARDING_FAMILIARITY_THRESHOLD);
        let prompt = agent.system_prompt();
        assert!(!prompt.contains("Warmly ask"));
        assert!(prompt.contains("gently ask"));
    }
}
