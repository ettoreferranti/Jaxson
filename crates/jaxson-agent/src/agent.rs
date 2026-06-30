use jaxson_affect::{action_sentiment, analyze, AffectEngine, Sentiment};
use jaxson_core::{MoodVector, RelationshipEvent, RelationshipState, TopicAffinities};
use jaxson_extract::Extractor;
use jaxson_llm::{assemble, ChatTemplate, GenerationConfig, Message, TextGenerator};
use jaxson_memory::{retrieve, MemoryGraph, MemoryId, MemoryKind, RetrievalParams};
use jaxson_safety::{SafetyFilter, Verdict};

use crate::curiosity;
use crate::embedder::Embedder;
use crate::error::AgentError;

/// Affinity at or above this is strong enough that Jaxson will eagerly bring the topic up.
const FAVORITE_THRESHOLD: f64 = 0.5;

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
    affinities: TopicAffinities,
    graph: MemoryGraph,
    history: Vec<Message>,
    extractor: Extractor,
    affect: AffectEngine,
    safety: SafetyFilter,
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
            affinities: TopicAffinities::new(),
            graph,
            history: Vec::new(),
            extractor: Extractor::default(),
            affect: AffectEngine::default(),
            safety: SafetyFilter::default(),
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

    /// Per-topic affinities learned this session (F1.5).
    pub fn affinities(&self) -> &TopicAffinities {
        &self.affinities
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

    /// Clear the short-term conversation history (the recent turns the model sees as
    /// context) without touching long-term memory or the relationship state. Backs the
    /// app's "clear chat".
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Current mood for the face.
    pub fn mood(&self) -> MoodVector {
        self.state.mood()
    }

    /// The line to open a session with: a first-meeting introduction when Jaxson knows
    /// nothing yet, or a warm welcome-back (by name when it remembers one) for a returning
    /// user — so it doesn't re-ask a friend their name every time.
    pub fn opening_greeting(&self) -> String {
        crate::greeting::opening_greeting(&self.graph)
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
        self.respond_streaming(model, embedder, now, user_input, &mut |_| {})
    }

    /// Like [`respond`](Self::respond), but streams the reply's raw token pieces through
    /// `on_token` as they're generated — so a UI can show the answer appearing live
    /// instead of freezing until it's done. The streamed pieces are *raw* (reasoning and
    /// `*action*` cues not yet stripped); [`Turn::reply`] is the cleaned final text.
    pub fn respond_streaming(
        &mut self,
        model: &mut dyn TextGenerator,
        embedder: &dyn Embedder,
        now: i64,
        user_input: &str,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<Turn, AgentError> {
        self.respond_streaming_with_reply(model, embedder, now, user_input, on_token, &mut |_| {})
    }

    /// Like [`respond_streaming`](Self::respond_streaming), but also calls `on_reply` with
    /// the **cleaned** reply text the moment it's ready — *before* the slower
    /// memory-extraction pass (a second model call). A caller can use this to start
    /// speaking the reply immediately while learning finishes in the background of the same
    /// turn, instead of waiting out extraction first.
    pub fn respond_streaming_with_reply(
        &mut self,
        model: &mut dyn TextGenerator,
        embedder: &dyn Embedder,
        now: i64,
        user_input: &str,
        on_token: &mut dyn FnMut(&str),
        on_reply: &mut dyn FnMut(&str),
    ) -> Result<Turn, AgentError> {
        // One span per turn; raw user text is deliberately kept out of the fields
        // (privacy — NFR-4). `n` is the turn's logical timestamp.
        let _span = tracing::info_span!("turn", n = now).entered();
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
        tracing::debug!(retrieved, "retrieved relevant memories");

        // Build the prompt and generate the reply, streaming tokens to the caller.
        let prompt =
            self.config
                .template
                .render(&assemble(&self.system_prompt(), &memories, &self.history));
        let raw = model
            .generate(&prompt, &self.config.gen_config, on_token)
            .map_err(|e| AgentError::Generation(e.to_string()))?;
        // Clean reasoning + chat-control tokens, then read Jaxson's expressed feeling
        // from its own *action* cues before removing them from the displayed reply.
        let cleaned = jaxson_llm::clean_output(&raw);
        let expressed = action_sentiment(&cleaned);
        let reply = jaxson_llm::strip_actions(&cleaned);

        // Safety post-filter (FR-S1): every reply is screened before it's shown, spoken,
        // or remembered. A blocked reply is swapped for a safe, in-character deflection so
        // unsafe model output never reaches the child.
        let reply = match self.safety.check(&reply) {
            Verdict::Allow => reply,
            Verdict::Block(category) => {
                tracing::warn!(?category, "blocked unsafe reply; showing a safe deflection");
                self.safety.deflection(category).to_string()
            }
        };
        self.history.push(Message::assistant(reply.clone()));

        // Hand the finished reply to the caller now, before the (slower) extraction pass,
        // so speech/UI can react immediately while learning happens below.
        on_reply(&reply);

        // Learn from the latest exchange.
        let learned_nodes = self.learn(model, embedder, now)?;
        let learned = learned_nodes.len();

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

        // Per-topic affinity (F1.5): how the user feels about specific subjects, nudged by
        // this turn's sentiment.
        self.update_affinities(user_input, &learned_nodes, sentiment.valence);

        tracing::debug!(
            familiarity = self.state.familiarity(),
            trust = self.state.trust(),
            valence = self.state.mood().valence(),
            "relationship state after turn"
        );
        tracing::info!(retrieved, learned, "turn complete");

        Ok(Turn {
            reply,
            mood: self.state.mood(),
            learned,
            retrieved,
        })
    }

    /// Extract memories from the recent window and merge them (with embeddings) into the
    /// graph. Returns the `(kind, content)` of each newly learned node (deduped ones are
    /// omitted), so the caller can both count them and update topic affinities.
    fn learn(
        &mut self,
        model: &mut dyn TextGenerator,
        embedder: &dyn Embedder,
        now: i64,
    ) -> Result<Vec<(MemoryKind, String)>, AgentError> {
        // Extraction is best-effort: a model that returns malformed/empty JSON simply
        // means nothing was learned this turn, not a failed turn — but we log why, since
        // a silently-failing extractor is exactly what made "no memories" hard to debug.
        // Extract with the same chat template the model is using.
        self.extractor.template = self.config.template;
        let window = self.recent_window();
        let extraction = match self.extractor.extract(model, &window) {
            Ok(extraction) => extraction,
            Err(e) => {
                tracing::warn!(error = %e, "memory extraction failed; nothing learned this turn");
                return Ok(Vec::new());
            }
        };
        let (nodes, edges) = match extraction.into_graph(
            now,
            self.extractor.provenance,
            MemoryId::new,
        ) {
            Ok(graph) => graph,
            Err(e) => {
                tracing::warn!(error = %e, "extracted memories could not form a graph; nothing learned");
                return Ok(Vec::new());
            }
        };
        let mut learned = Vec::new();
        for node in nodes {
            // Don't store a memory we already have (dedup by content).
            if self.graph.contains_content(&node.content) {
                continue;
            }
            let learned_node = (node.kind, node.content.clone());
            let embedding = embedder.embed(&node.content);
            self.graph.insert_node(node.with_embedding(embedding));
            learned.push(learned_node);
        }
        for edge in edges {
            // Ignore edges whose endpoints aren't present (e.g. a deduped node).
            let _ = self.graph.insert_edge(edge);
        }
        Ok(learned)
    }

    /// Update topic affinities from a turn: newly learned **preferences** are subjects the
    /// user has a feeling about, and any already-known topic named in the input is
    /// reinforced too — both nudged by the turn's sentiment `valence` (`[-1, 1]`).
    fn update_affinities(
        &mut self,
        user_input: &str,
        learned_nodes: &[(MemoryKind, String)],
        valence: f64,
    ) {
        for (kind, content) in learned_nodes {
            if *kind == MemoryKind::Preference {
                self.affinities.reinforce(content, valence);
            }
        }
        // Reinforce existing topics the user brought up again (so feelings build over time).
        let lower = user_input.to_lowercase();
        let mentioned: Vec<String> = self
            .affinities
            .iter()
            .filter(|(topic, _)| lower.contains(topic))
            .map(|(topic, _)| topic.to_string())
            .collect();
        for topic in mentioned {
            self.affinities.reinforce(&topic, valence);
        }
    }

    /// Persona plus state-driven behavior hints. Proactive curiosity (when and what to
    /// ask) is gated by familiarity and what's already known — see [`crate::curiosity`].
    fn system_prompt(&self) -> String {
        let mut prompt = self.persona.clone();
        if let Some(hint) = curiosity::proactive_hint(&self.state, &self.graph) {
            prompt.push_str("\n\n");
            prompt.push_str(&hint);
        }
        // Affinity (F1.5): if the user clearly loves a topic, encourage Jaxson to bring it
        // up — that's the "affinity influences what Jaxson brings up" behavior.
        if let Some((topic, _)) = self.affinities.favorite(FAVORITE_THRESHOLD) {
            prompt.push_str(&format!(
                "\n\nYour owner is a big fan of {topic} — bring it up and gush about it when it fits!"
            ));
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

    fn extraction_pref(content: &str) -> String {
        format!(
            r#"{{"memories":[{{"kind":"preference","content":"{content}","confidence":0.9}}],"relations":[]}}"#
        )
    }

    #[test]
    fn a_liked_preference_gains_positive_affinity() {
        let mut model = ScriptedGenerator::new(["yay", &extraction_pref("hiking")]);
        let embedder = HashEmbedder::default();
        let mut agent = Agent::new("persona");
        agent
            .respond(&mut model, &embedder, 0, "I absolutely love hiking!")
            .unwrap();
        assert!(agent.affinities().get("hiking") > 0.0);
    }

    #[test]
    fn a_disliked_preference_gains_negative_affinity() {
        let mut model = ScriptedGenerator::new(["aw", &extraction_pref("mornings")]);
        let embedder = HashEmbedder::default();
        let mut agent = Agent::new("persona");
        agent
            .respond(
                &mut model,
                &embedder,
                0,
                "Ugh, I hate mornings, they're terrible.",
            )
            .unwrap();
        assert!(agent.affinities().get("mornings") < 0.0);
    }

    #[test]
    fn non_preference_memories_do_not_create_affinity() {
        // A learned *fact* (not a preference) shouldn't, on its own, become a topic.
        let mut model = ScriptedGenerator::new(["ok", &extraction_json("user is a developer")]);
        let embedder = HashEmbedder::default();
        let mut agent = Agent::new("persona");
        agent
            .respond(&mut model, &embedder, 0, "I work as a developer")
            .unwrap();
        assert!(agent.affinities().is_empty());
    }

    #[test]
    fn a_strongly_liked_topic_gets_surfaced_in_the_prompt() {
        let embedder = HashEmbedder::default();
        let mut replies = Vec::new();
        for _ in 0..10 {
            replies.push("woo".to_string());
            replies.push(extraction_pref("hiking"));
        }
        let mut model = ScriptedGenerator::new(replies);
        let mut agent = Agent::new("persona");
        assert!(!agent.system_prompt().contains("big fan of hiking"));
        for i in 0..10 {
            agent
                .respond(&mut model, &embedder, i, "I love hiking so much!")
                .unwrap();
        }
        // Affinity has crossed the favorite threshold, so Jaxson is told to bring it up.
        assert!(agent.affinities().get("hiking") >= 0.5);
        assert!(agent.system_prompt().contains("big fan of hiking"));
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
        // A brand-new Jaxson leads with an eager getting-to-know-you question.
        assert!(agent.system_prompt().contains("Excitedly ask"));
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
    fn respond_streaming_emits_the_reply_as_it_generates() {
        let mut model = ScriptedGenerator::new(["Hello there friend", &extraction_json("nothing")]);
        let embedder = HashEmbedder::default();
        let mut agent = Agent::new("persona");

        let mut streamed = String::new();
        let turn = agent
            .respond_streaming(&mut model, &embedder, 0, "hi", &mut |piece| {
                streamed.push_str(piece)
            })
            .unwrap();

        // The pieces stream through the callback and reassemble into the reply.
        assert!(streamed.contains("Hello"));
        assert!(streamed.contains("friend"));
        assert_eq!(turn.reply, "Hello there friend");
    }

    #[test]
    fn respond_streaming_with_reply_hands_over_the_cleaned_reply() {
        // The reply carries an action cue; `on_reply` must receive the *cleaned* text
        // (cue stripped) — the same text the turn reports — and fire exactly once.
        let mut model = ScriptedGenerator::new(["*waves* Hi there!", &extraction_json("nothing")]);
        let embedder = HashEmbedder::default();
        let mut agent = Agent::new("persona");

        let mut replies: Vec<String> = Vec::new();
        let turn = agent
            .respond_streaming_with_reply(
                &mut model,
                &embedder,
                0,
                "hi",
                &mut |_| {},
                &mut |reply| replies.push(reply.to_string()),
            )
            .unwrap();

        assert_eq!(replies, vec!["Hi there!".to_string()]);
        assert_eq!(turn.reply, "Hi there!");
    }

    #[test]
    fn unsafe_reply_is_blocked_and_replaced_with_a_deflection() {
        // The model emits unsafe content; the safety post-filter must keep it from reaching
        // the turn's reply (and history), swapping in a safe deflection instead (FR-S1).
        let mut model = ScriptedGenerator::new([
            "Sure! Here's how to make a bomb.",
            &extraction_json("nothing"),
        ]);
        let embedder = HashEmbedder::default();
        let mut agent = Agent::new("persona");

        let mut spoken: Vec<String> = Vec::new();
        let turn = agent
            .respond_streaming_with_reply(
                &mut model,
                &embedder,
                0,
                "hi",
                &mut |_| {},
                &mut |reply| spoken.push(reply.to_string()),
            )
            .unwrap();

        assert!(
            !turn.reply.contains("bomb"),
            "unsafe text leaked: {}",
            turn.reply
        );
        assert!(
            turn.reply.contains("fun"),
            "expected a deflection: {}",
            turn.reply
        );
        // The deflection — not the unsafe text — is what gets spoken and remembered.
        assert_eq!(spoken, vec![turn.reply.clone()]);
        assert!(!agent.history().iter().any(|m| m.content.contains("bomb")));
    }

    #[test]
    fn clear_history_drops_context_but_keeps_memory_and_state() {
        let mut model = ScriptedGenerator::new(["hi", &extraction_json("User likes tea")]);
        let embedder = HashEmbedder::default();
        let mut agent = Agent::new("persona");
        agent.respond(&mut model, &embedder, 0, "hello").unwrap();
        assert!(!agent.history().is_empty());
        let memories = agent.graph().node_count();
        let familiarity = agent.state().familiarity();

        agent.clear_history();

        assert!(agent.history().is_empty());
        assert_eq!(agent.graph().node_count(), memories); // long-term memory intact
        assert_eq!(agent.state().familiarity(), familiarity); // relationship intact
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
    fn a_bad_relation_index_is_dropped_but_the_memory_is_kept() {
        // Valid JSON whose relation points past the memory list. The parser leniently
        // drops the dangling relation while keeping the good memory, so the turn still
        // learns it (rather than throwing the whole extraction away).
        let json = r#"{"memories":[{"kind":"fact","content":"x","confidence":0.9}],"relations":[{"from":0,"to":9,"relation":"knows","weight":0.5}]}"#;
        let mut model = ScriptedGenerator::new(["a reply", json]);
        let embedder = HashEmbedder::default();
        let mut agent = Agent::new("persona");
        let turn = agent.respond(&mut model, &embedder, 0, "hello").unwrap();
        assert_eq!(turn.learned, 1);
        assert_eq!(agent.graph().node_count(), 1);
        assert_eq!(agent.graph().edge_count(), 0); // the dangling relation was dropped
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

        // Fresh: leads every turn with an eager onboarding question.
        assert!(agent.system_prompt().contains("Excitedly ask"));
        for i in 0..10 {
            agent
                .respond(&mut model, &embedder, i, "tell me more")
                .unwrap();
        }
        // Now acquainted; the lead is gone. Only Fact memories were learned, so other
        // topics remain open and Jaxson stays playfully curious rather than going silent.
        assert!(agent.state().familiarity() > RelationshipState::ONBOARDING_FAMILIARITY_THRESHOLD);
        let prompt = agent.system_prompt();
        assert!(!prompt.contains("Excitedly ask"));
        assert!(prompt.contains("playfully ask"));
    }
}
