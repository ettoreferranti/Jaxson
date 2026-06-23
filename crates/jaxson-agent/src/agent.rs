use jaxson_affect::{analyze, AffectEngine};
use jaxson_core::{MoodVector, RelationshipEvent, RelationshipState};
use jaxson_extract::Extractor;
use jaxson_llm::{assemble, ChatTemplate, GenerationConfig, Message, TextGenerator};
use jaxson_memory::{retrieve, MemoryGraph, MemoryId, RetrievalParams};

use crate::embedder::Embedder;
use crate::error::AgentError;

/// Appended to the persona while Jaxson barely knows the user (behavior gating driven
/// by the relationship state machine — FR-M6).
const ONBOARDING_HINT: &str =
    "\n\nYou don't know this person well yet — warmly ask a question to learn \
     something new about them.";

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
        let reply = model
            .complete(&prompt, &self.config.gen_config)
            .map_err(|e| AgentError::Generation(e.to_string()))?;
        self.history.push(Message::assistant(reply.clone()));

        // Learn from the latest exchange.
        let learned = self.learn(model, embedder, now)?;

        // Familiarity grows with each fact learned.
        for _ in 0..learned {
            self.state = self.state.apply(RelationshipEvent::LearnedFact);
        }

        // Mood follows the sentiment of what the user said (affect engine, decoupled
        // from the model's wording). Smoothing lives in the state machine.
        let target = self.affect.target_mood(&self.state, analyze(user_input));
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
        let window = self.recent_window();
        let extraction = self.extractor.extract(model, &window)?;
        let (nodes, edges) =
            extraction.into_graph(now, self.extractor.provenance, MemoryId::new)?;
        let learned = nodes.len();
        for node in nodes {
            let embedding = embedder.embed(&node.content);
            self.graph.insert_node(node.with_embedding(embedding));
        }
        for edge in edges {
            self.graph
                .insert_edge(edge)
                .map_err(|e| AgentError::Graph(e.to_string()))?;
        }
        Ok(learned)
    }

    /// Persona plus state-driven behavior hints.
    fn system_prompt(&self) -> String {
        let mut prompt = self.persona.clone();
        if self.state.should_prioritize_onboarding() {
            prompt.push_str(ONBOARDING_HINT);
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

    fn extraction_json(content: &str) -> String {
        format!(
            r#"{{"memories":[{{"kind":"fact","content":"{content}","confidence":0.9}}],"relations":[]}}"#
        )
    }

    #[test]
    fn fresh_agent_starts_blank_and_onboarding() {
        let agent = Agent::new("You are Jaxson.");
        assert!(agent.graph().is_empty());
        assert_eq!(agent.state().familiarity(), 0.0);
        assert!(agent.system_prompt().contains(ONBOARDING_HINT));
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
    fn generation_failure_surfaces_as_error() {
        // No replies queued and an empty reply still generates; force a parse failure
        // instead by returning non-JSON for the extraction step.
        let mut model = ScriptedGenerator::new(["a reply", "not json at all"]);
        let embedder = HashEmbedder::default();
        let mut agent = Agent::new("persona");
        let err = agent
            .respond(&mut model, &embedder, 0, "hello")
            .unwrap_err();
        assert!(matches!(err, AgentError::Extraction(_)));
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
    fn onboarding_hint_drops_once_familiar() {
        let embedder = HashEmbedder::default();
        let mut replies = Vec::new();
        for i in 0..10 {
            replies.push("ok".to_string());
            replies.push(extraction_json(&format!("fact number {i}")));
        }
        let mut model = ScriptedGenerator::new(replies);
        let mut agent = Agent::new("You are Jaxson.");

        assert!(agent.system_prompt().contains(ONBOARDING_HINT));
        for i in 0..10 {
            agent
                .respond(&mut model, &embedder, i, "tell me more")
                .unwrap();
        }
        assert!(agent.state().familiarity() > RelationshipState::ONBOARDING_FAMILIARITY_THRESHOLD);
        assert!(!agent.system_prompt().contains(ONBOARDING_HINT));
    }
}
