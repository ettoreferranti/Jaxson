//! Jaxson's desktop shell: an egui window showing the animated face above a chat box.
//!
//! The face is the `jaxson-face` rasterizer's bitmap, refreshed every frame so it
//! blinks and drifts, and its expression follows the agent's mood. The chat is wired to
//! a mock-backed [`Agent`]: replies are canned (no model yet — that's `--features
//! llama`), but the *face reacts live* to the sentiment of whatever you type.
//!
//! Run: `cargo run --manifest-path crates/jaxson-app/Cargo.toml`

use std::sync::mpsc;
use std::time::Instant;

use eframe::egui;
use egui::Color32;

mod logging;
mod parental;
mod persist;

use jaxson_agent::{Agent, AgentConfig, Embedder, HashEmbedder, Turn};
use jaxson_core::MoodVector;
use jaxson_face::{face, face_with, rasterize, Activity, Bitmap};
use jaxson_llm::ollama::{self, OllamaModel};
use jaxson_llm::{ChatTemplate, GenerationConfig, LlmError, TextGenerator};
use jaxson_memory::{MemoryId, MemoryKind, MemoryNode};
use jaxson_safety::{PasscodeHash, Strictness};

/// Jaxson's character lives in `jaxson-agent` (single source of truth, shared with the
/// `persona_probe` tuning example).
const PERSONA: &str = jaxson_agent::DEFAULT_PERSONA;
const FACE_PIXELS: usize = 250;

/// A stand-in "model" for the UI: returns empty extractions and a canned chat reply,
/// chosen by sniffing the prompt. Lets the loop run without a real model.
struct DemoModel;

impl TextGenerator for DemoModel {
    fn generate(
        &mut self,
        prompt: &str,
        _config: &GenerationConfig,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<String, LlmError> {
        let reply = if prompt.contains("Extract memories as JSON") {
            r#"{"memories":[],"relations":[]}"#.to_string()
        } else {
            "I'm a demo brain for now (no model wired yet) — but watch my face react to \
             how you say things!"
                .to_string()
        };
        on_token(&reply);
        Ok(reply)
    }
}

/// Pick the chat template from `JAXSON_TEMPLATE` (chatml/llama3/plain), default ChatML.
fn select_template() -> ChatTemplate {
    match std::env::var("JAXSON_TEMPLATE").as_deref() {
        Ok("llama3") => ChatTemplate::Llama3,
        Ok("plain") => ChatTemplate::Plain,
        _ => ChatTemplate::ChatMl,
    }
}

/// What `boot` produces: the chat generator, the active embedder, a status label, and —
/// with a real model — the embedder that shares the chat model's weights (kept so "same
/// as chat" can be reselected later without reloading) plus the initial dedicated-embed
/// selection from `$JAXSON_EMBED_MODEL`. `selected`/`template` are set when the chat model
/// was resolved by name (so the picker reflects it and the chat format matches).
struct Boot {
    // `+ Send` so the brain can move to the generation worker thread (F1.9b).
    model: Box<dyn TextGenerator + Send>,
    embedder: Box<dyn Embedder + Send>,
    status: String,
    /// The discovered-model index the chat model resolved to (by name), if any.
    selected: Option<usize>,
    /// The chat template to use, when known from the model name; else `None` (fall back
    /// to `$JAXSON_TEMPLATE`).
    template: Option<ChatTemplate>,
    #[cfg(feature = "llama")]
    base_embedder: Option<jaxson_llm::backends::LlamaEmbedder>,
    #[cfg(feature = "llama")]
    embed_selected: Option<usize>,
}

/// The initial brain: the real model from `$JAXSON_MODEL` when built with `--features
/// llama`, otherwise the demo brain. `$JAXSON_MODEL` accepts a discovered model **name**
/// (e.g. `llama3.1`) — in which case the chat template is auto-selected — or a path to a
/// `.gguf`. `$JAXSON_EMBED_MODEL` (a discovered model name) optionally selects a separate
/// embedding model; otherwise embeddings come from the chat model.
#[cfg_attr(not(feature = "llama"), allow(unused_variables))]
fn boot(models: &[OllamaModel]) -> Boot {
    #[cfg(feature = "llama")]
    {
        use jaxson_llm::backends::{load_generator_and_embedder, LlamaConfig};
        if let Ok(arg) = std::env::var("JAXSON_MODEL") {
            // Resolve as a direct `.gguf` path, or a discovered model name.
            let resolved = {
                let path = std::path::PathBuf::from(&arg);
                if path.is_file() {
                    Some((None, path))
                } else {
                    resolve_model(models, &arg).map(|i| (Some(i), models[i].path.clone()))
                }
            };
            match resolved {
                Some((selected, path)) => match load_generator_and_embedder(&LlamaConfig {
                    model_path: path,
                    ..Default::default()
                }) {
                    Ok((generator, shared)) => {
                        let embed_selected = std::env::var("JAXSON_EMBED_MODEL")
                            .ok()
                            .and_then(|m| resolve_model(models, &m));
                        // When resolved by name, mirror the picker: show it and match the
                        // chat format to the model so it doesn't emit garbled tokens.
                        let name = selected.map(|i| models[i].name.clone());
                        let template = name.as_deref().map(ChatTemplate::for_model_name);
                        return Boot {
                            model: Box::new(generator),
                            embedder: active_embedder(models, embed_selected, &shared),
                            status: format!("model: {}", name.unwrap_or(arg)),
                            selected,
                            template,
                            base_embedder: Some(shared),
                            embed_selected,
                        };
                    }
                    Err(e) => eprintln!("Failed to load JAXSON_MODEL ({arg}): {e}"),
                },
                None => eprintln!(
                    "JAXSON_MODEL '{arg}' is not a .gguf file or a known model name; using demo brain"
                ),
            }
        }
    }
    Boot {
        model: Box::new(DemoModel),
        embedder: Box::new(HashEmbedder::default()),
        status: "demo brain".to_string(),
        selected: None,
        template: None,
        #[cfg(feature = "llama")]
        base_embedder: None,
        #[cfg(feature = "llama")]
        embed_selected: None,
    }
}

/// Build the active embedder: a dedicated model (its own weights) when one is selected,
/// otherwise a cheap clone of the chat-model-shared embedder. A dedicated model that
/// fails to load falls back to the chat model's embeddings.
#[cfg(feature = "llama")]
fn active_embedder(
    models: &[OllamaModel],
    embed_selected: Option<usize>,
    base: &jaxson_llm::backends::LlamaEmbedder,
) -> Box<dyn Embedder + Send> {
    use jaxson_llm::backends::{LlamaConfig, LlamaEmbedder};
    match embed_selected {
        Some(i) => match LlamaEmbedder::load(&LlamaConfig {
            model_path: models[i].path.clone(),
            // Memory snippets are short, so a small window keeps a dedicated embedder light.
            n_ctx: 512,
            ..Default::default()
        }) {
            Ok(dedicated) => Box::new(LlamaEmbed(dedicated)),
            Err(e) => {
                tracing::error!(model = %models[i].name, error = %e,
                    "failed to load embedding model; falling back to chat model");
                Box::new(LlamaEmbed(base.clone()))
            }
        },
        None => Box::new(LlamaEmbed(base.clone())),
    }
}

/// Find a discovered model by exact name, else by name prefix.
#[cfg(feature = "llama")]
fn resolve_model(models: &[OllamaModel], arg: &str) -> Option<usize> {
    models
        .iter()
        .position(|m| m.name == arg)
        .or_else(|| models.iter().position(|m| m.name.starts_with(arg)))
}

/// Adapts the fallible llama embedder to the agent's infallible [`Embedder`] seam: an
/// embedding error degrades to an empty vector (retrieval just finds no match) rather
/// than failing the turn.
#[cfg(feature = "llama")]
struct LlamaEmbed(jaxson_llm::backends::LlamaEmbedder);

#[cfg(feature = "llama")]
impl Embedder for LlamaEmbed {
    fn embed(&self, text: &str) -> Vec<f32> {
        match self.0.embed(text) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(error = %e, "embedding failed; using empty vector");
                Vec::new()
            }
        }
    }
}

/// A live microphone capture (push-to-talk): a cpal input stream filling a shared buffer
/// until it's stopped. Dropping it stops the stream.
#[cfg(feature = "whisper")]
struct Recorder {
    _stream: cpal::Stream,
    buffer: std::sync::Arc<std::sync::Mutex<Vec<f32>>>,
    sample_rate: u32,
    channels: u16,
}

/// Load the whisper speech-to-text model from `$JAXSON_WHISPER_MODEL`, if set and loadable.
#[cfg(feature = "whisper")]
fn load_stt() -> Option<Box<dyn jaxson_perception::SpeechToText>> {
    let path = std::env::var("JAXSON_WHISPER_MODEL").ok()?;
    // Scrub the username out of the path before logging (privacy / F2.6).
    let shown = jaxson_core::scrub::redact(&path);
    match jaxson_perception::backends::WhisperStt::load(&path) {
        Ok(stt) => {
            tracing::info!(model = %shown, "loaded whisper model");
            Some(Box::new(stt))
        }
        Err(e) => {
            tracing::error!(model = %shown, error = %e, "failed to load whisper model");
            None
        }
    }
}

/// Load the Piper text-to-speech voice from `$JAXSON_PIPER_VOICE`, if set and loadable.
/// Speech synthesis runs on the generation worker thread (it's bundled into [`Brain`]), so
/// the box must be `Send`.
#[cfg(feature = "piper")]
fn load_tts() -> Option<Box<dyn jaxson_perception::TextToSpeech + Send>> {
    let path = std::env::var("JAXSON_PIPER_VOICE").ok()?;
    // Scrub the username out of the path before logging (privacy / F2.6).
    let shown = jaxson_core::scrub::redact(&path);
    match jaxson_perception::backends::PiperTts::load(&path) {
        Ok(tts) => {
            // Piper voices tend to read fast and flat; slow the pace a touch for a calmer,
            // kid-friendly delivery. Override with $JAXSON_PIPER_LENGTH_SCALE (higher =
            // slower; the voice's own default is used if it's unset or unparseable).
            let length_scale = std::env::var("JAXSON_PIPER_LENGTH_SCALE")
                .ok()
                .and_then(|s| s.parse::<f32>().ok())
                .unwrap_or(1.2);
            tracing::info!(voice = %shown, length_scale, "loaded piper voice");
            Some(Box::new(tts.with_length_scale(Some(length_scale))))
        }
        Err(e) => {
            tracing::error!(voice = %shown, error = %e, "failed to load piper voice");
            None
        }
    }
}

/// The audio output for spoken replies: the rodio device sink (kept alive — dropping it
/// stops audio) plus a player that resamples and feeds it. Lives on the UI thread (the
/// underlying cpal stream isn't `Send`); only present with `--features piper`.
#[cfg(feature = "piper")]
struct AudioOut {
    // Field order matters for drop: the player is torn down before the sink it feeds.
    player: rodio::Player,
    _sink: rodio::MixerDeviceSink,
}

#[cfg(feature = "piper")]
impl AudioOut {
    /// Open the default output device, or `None` if there isn't one (playback degrades to
    /// silent — never fatal).
    fn open() -> Option<Self> {
        match rodio::DeviceSinkBuilder::open_default_sink() {
            Ok(mut sink) => {
                sink.log_on_drop(false);
                let player = rodio::Player::connect_new(sink.mixer());
                Some(AudioOut {
                    player,
                    _sink: sink,
                })
            }
            Err(e) => {
                tracing::error!(error = %e, "no audio output device; replies won't be spoken");
                None
            }
        }
    }

    /// Queue synthesized speech for playback. rodio resamples the voice's rate to the
    /// device and maps the mono signal to its channels. A no-op for empty audio.
    fn play(&self, audio: &jaxson_perception::Audio) {
        let Some(rate) = std::num::NonZero::new(audio.sample_rate) else {
            return; // rate 0 marks "nothing to say"
        };
        if audio.samples.is_empty() {
            return;
        }
        // rodio's default sample type is f32 — same as our PCM — so feed it directly.
        let mono = std::num::NonZero::new(1u16).expect("1 is nonzero");
        self.player.append(rodio::buffer::SamplesBuffer::new(
            mono,
            rate,
            audio.samples.clone(),
        ));
    }
}

/// Lip-sync driver (F2.3): turns the stream of spoken sentence chunks into the mouth's
/// current openness. Each chunk's loudness envelope is queued with the wall-clock duration
/// of one envelope frame; [`level`](Self::level) reads the envelope at the elapsed playback
/// time (audio plays at real time, so wall-clock tracks the playhead closely enough for a
/// stylized face). Chunks play back-to-back; the timeline resets when speech resumes after
/// a silence. Only present with `--features piper`.
#[cfg(feature = "piper")]
#[derive(Default)]
struct SpeechAnimator {
    /// `(loudness envelope, seconds per envelope frame)` for chunks not yet fully played.
    queue: std::collections::VecDeque<(Vec<f32>, f64)>,
    /// When the chunk at the front of the queue started playing.
    head_start: Option<Instant>,
}

#[cfg(feature = "piper")]
impl SpeechAnimator {
    /// Queue a spoken chunk's loudness envelope. `frame_secs` is the wall-clock duration
    /// each envelope sample represents. No-op for an empty envelope.
    fn push(&mut self, envelope: Vec<f32>, frame_secs: f64) {
        if envelope.is_empty() || frame_secs <= 0.0 {
            return;
        }
        // If nothing is playing, restart the timeline so the new chunk begins now rather
        // than partway through (the previous run's elapsed time is stale).
        if self.queue.is_empty() {
            self.head_start = None;
        }
        self.queue.push_back((envelope, frame_secs));
    }

    /// The current mouth level in `[0, 1]`, or `None` when nothing is playing. Drops chunks
    /// whose playback time has fully elapsed.
    fn level(&mut self) -> Option<f64> {
        let now = Instant::now();
        let mut start = *self.head_start.get_or_insert(now);
        loop {
            // Read the front chunk's timing, then drop the borrow before mutating the queue.
            let (dur, frame_secs, len) = match self.queue.front() {
                Some((env, frame_secs)) => (env.len() as f64 * frame_secs, *frame_secs, env.len()),
                None => {
                    self.head_start = None;
                    return None;
                }
            };
            let elapsed = now.duration_since(start).as_secs_f64();
            if elapsed >= dur {
                // This chunk finished; advance the playhead to the next one.
                start += std::time::Duration::from_secs_f64(dur);
                self.queue.pop_front();
                continue;
            }
            let idx = ((elapsed / frame_secs) as usize).min(len - 1);
            let level = self.queue.front().expect("front checked above").0[idx] as f64;
            self.head_start = Some(start);
            return Some(level);
        }
    }
}

/// The agent, model, embedder, and (with `--features piper`) the speech synthesizer
/// bundled together so a turn can be handed to a worker thread and handed back when done
/// (F1.9b — keeps the window responsive while the model generates and speaks). All are
/// `Send`.
struct Brain {
    agent: Agent,
    model: Box<dyn TextGenerator + Send>,
    embedder: Box<dyn Embedder + Send>,
    #[cfg(feature = "piper")]
    tts: Option<Box<dyn jaxson_perception::TextToSpeech + Send>>,
}

/// Messages the generation worker streams back to the UI thread.
enum TurnUpdate {
    /// A raw token piece, as the reply is generated.
    Token(String),
    /// A synthesized sentence of the reply, ready to play (piper feature only). Sent as
    /// each sentence finishes so speech starts before the whole reply is synthesized.
    #[cfg(feature = "piper")]
    Speak(jaxson_perception::Audio),
    /// The turn finished: the brain comes back (to be reinstalled) plus the result.
    Done(Box<Brain>, Result<Turn, String>),
}

struct JaxsonApp {
    /// The brain when idle; `None` while a turn is running on the worker thread.
    brain: Option<Brain>,
    /// Receiver for the in-flight turn's streamed tokens + completion.
    pending: Option<mpsc::Receiver<TurnUpdate>>,
    /// Accumulated raw reply text shown live while generating.
    streaming: String,
    /// A clone of the egui context, so the worker can wake the UI as tokens arrive.
    egui_ctx: egui::Context,
    /// Cached for display while the brain is away on the worker thread.
    mood: MoodVector,
    memory_count: usize,
    // The embedder sharing the chat model's weights — reused when the embedding model is
    // "same as chat", so toggling back doesn't reload. Only with a real model.
    #[cfg(feature = "llama")]
    base_embedder: Option<jaxson_llm::backends::LlamaEmbedder>,
    // Which discovered model to embed with; `None` means "same as the chat model".
    #[cfg(feature = "llama")]
    embed_selected: Option<usize>,
    models: Vec<OllamaModel>,
    selected: Option<usize>,
    status: String,
    transcript: Vec<(&'static str, String)>,
    input: String,
    start: Instant,
    turn: i64,
    face_tex: egui::TextureHandle,
    // Encrypted on-disk memory (Keychain-keyed); ephemeral without `--features sqlite`.
    persist: persist::Persistence,
    export_status: String,
    // Voice input (push-to-talk); only with `--features whisper`.
    #[cfg(feature = "whisper")]
    stt: Option<Box<dyn jaxson_perception::SpeechToText>>,
    #[cfg(feature = "whisper")]
    recorder: Option<Recorder>,
    #[cfg(feature = "whisper")]
    mic_status: String,
    // Voice output: plays synthesized replies; only with `--features piper`.
    #[cfg(feature = "piper")]
    audio: Option<AudioOut>,
    // Lip-sync: drives mouth openness from the playing speech (piper feature).
    #[cfg(feature = "piper")]
    speech: SpeechAnimator,
    // Memory inspector state.
    show_memories: bool,
    mem_search: String,
    editing: Option<MemoryId>,
    edit_buf: String,
    // Parental controls (FR-S3): persisted passcode + strictness, plus session unlock state.
    parental: parental::ParentalConfig,
    parent_open: bool,
    parent_unlocked: bool,
    parent_pin: String,
    parent_status: String,
}

impl JaxsonApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let neutral = to_image(&rasterize(&face(MoodVector::NEUTRAL, 0.0), FACE_PIXELS));
        let face_tex =
            cc.egui_ctx
                .load_texture("jaxson-face", neutral, egui::TextureOptions::NEAREST);
        let models = ollama::discover();
        let boot = boot(&models);
        // Prefer the template boot picked from the chat model's name; otherwise honor
        // $JAXSON_TEMPLATE (default ChatML).
        let template = boot.template.unwrap_or_else(select_template);
        // Load any previously persisted memory before the agent starts the session.
        let mut persist = persist::Persistence::open();
        let graph = persist.load();
        let mut agent = Agent::with_graph(PERSONA, graph).with_config(AgentConfig {
            template,
            // Stop generation at the template's end-of-turn token.
            gen_config: GenerationConfig {
                stop: template
                    .stop_tokens()
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
                ..Default::default()
            },
            ..Default::default()
        });
        // Apply the parent-chosen guardrail strictness (FR-S3) before the session starts.
        let parental = parental::load();
        agent.set_safety_strictness(parental.strictness);
        let mood = agent.mood();
        let memory_count = agent.graph().node_count();
        // A first-meeting intro, or a warm welcome-back (by name when remembered) for a
        // returning user — computed before the agent moves into the brain.
        let greeting = agent.opening_greeting();
        // Spoken replies (piper feature): load the voice, and only open the output device
        // if a voice actually loaded.
        #[cfg(feature = "piper")]
        let tts = load_tts();
        #[cfg(feature = "piper")]
        let audio = tts.is_some().then(AudioOut::open).flatten();
        JaxsonApp {
            brain: Some(Brain {
                agent,
                model: boot.model,
                embedder: boot.embedder,
                #[cfg(feature = "piper")]
                tts,
            }),
            pending: None,
            streaming: String::new(),
            egui_ctx: cc.egui_ctx.clone(),
            mood,
            memory_count,
            #[cfg(feature = "llama")]
            base_embedder: boot.base_embedder,
            #[cfg(feature = "llama")]
            embed_selected: boot.embed_selected,
            models,
            selected: boot.selected,
            status: boot.status,
            transcript: vec![("Jaxson", greeting)],
            input: String::new(),
            start: Instant::now(),
            turn: 0,
            face_tex,
            persist,
            export_status: String::new(),
            #[cfg(feature = "whisper")]
            stt: load_stt(),
            #[cfg(feature = "whisper")]
            recorder: None,
            #[cfg(feature = "whisper")]
            mic_status: String::new(),
            #[cfg(feature = "piper")]
            audio,
            #[cfg(feature = "piper")]
            speech: SpeechAnimator::default(),
            show_memories: false,
            mem_search: String::new(),
            editing: None,
            edit_buf: String::new(),
            parental,
            parent_open: false,
            parent_unlocked: false,
            parent_pin: String::new(),
            parent_status: String::new(),
        }
    }

    /// Set the guardrail strictness from the parent panel: apply it to the live agent and
    /// persist the choice so it sticks across restarts.
    fn set_strictness(&mut self, level: Strictness) {
        self.parental.strictness = level;
        if let Some(brain) = self.brain.as_mut() {
            brain.agent.set_safety_strictness(level);
        }
        self.parent_status = match parental::save(&self.parental) {
            Ok(()) => format!("Guardrails set to {level:?}."),
            Err(e) => format!("couldn't save settings: {e}"),
        };
    }

    /// The parental-control panel (FR-S3): first-run passcode setup, a locked passcode
    /// prompt, or — once unlocked — the guardrail-strictness picker plus memory review,
    /// which are otherwise hidden so a child can't reach them.
    fn parent_controls(&mut self, ui: &mut egui::Ui, busy: bool) {
        ui.separator();
        if self.parent_unlocked {
            ui.horizontal(|ui| {
                ui.label("Guardrails:");
                for level in [
                    Strictness::Lenient,
                    Strictness::Standard,
                    Strictness::Strict,
                ] {
                    let on = self.parental.strictness == level;
                    if ui.selectable_label(on, format!("{level:?}")).clicked() {
                        self.set_strictness(level);
                    }
                }
            });
            ui.horizontal(|ui| {
                if ui
                    .button(format!("🧠 Memories ({})", self.memory_count))
                    .clicked()
                {
                    self.show_memories = !self.show_memories;
                }
                if ui
                    .add_enabled(!busy, egui::Button::new("⬇ Export JSON"))
                    .clicked()
                {
                    if let Some(brain) = self.brain.as_ref() {
                        self.export_status = match persist::export_json(brain.agent.graph()) {
                            Ok(path) => format!("exported to {}", path.display()),
                            Err(e) => format!("export failed: {e}"),
                        };
                    }
                }
                if ui.button("🔒 Lock").clicked() {
                    self.parent_unlocked = false;
                    self.show_memories = false;
                    self.parent_open = false;
                    self.parent_status.clear();
                }
            });
        } else if self.parental.has_passcode() {
            ui.horizontal(|ui| {
                ui.label("Parent passcode:");
                ui.add(
                    egui::TextEdit::singleline(&mut self.parent_pin)
                        .password(true)
                        .desired_width(120.0),
                );
                if ui.button("Unlock").clicked() {
                    if self.parental.unlocks(&self.parent_pin) {
                        self.parent_unlocked = true;
                        self.parent_status.clear();
                    } else {
                        self.parent_status = "Wrong passcode.".to_string();
                    }
                    self.parent_pin.clear();
                }
            });
        } else {
            ui.label("Set a parent passcode to lock guardrails and memory review:");
            ui.horizontal(|ui| {
                ui.add(
                    egui::TextEdit::singleline(&mut self.parent_pin)
                        .password(true)
                        .desired_width(120.0),
                );
                let can_set = !self.parent_pin.trim().is_empty();
                if ui
                    .add_enabled(can_set, egui::Button::new("Set passcode"))
                    .clicked()
                {
                    self.parental.passcode = Some(PasscodeHash::new(self.parent_pin.trim()));
                    self.parent_unlocked = true;
                    self.parent_status = match parental::save(&self.parental) {
                        Ok(()) => "Passcode set.".to_string(),
                        Err(e) => format!("couldn't save: {e}"),
                    };
                    self.parent_pin.clear();
                }
            });
        }
        if !self.parent_status.is_empty() {
            ui.small(self.parent_status.as_str());
        }
    }

    /// The memory inspector window: browse, search, edit, and delete what Jaxson knows.
    fn memory_window(&mut self, ctx: &egui::Context) {
        if !self.show_memories {
            return;
        }
        // The graph lives in the brain; while a turn is generating it's away on the
        // worker, so the inspector is unavailable for that brief window.
        let Some(brain) = self.brain.as_mut() else {
            return;
        };
        // Snapshot the (filtered) memories as owned data so we can mutate the graph after.
        let items: Vec<(MemoryId, String, MemoryKind, f32)> = brain
            .agent
            .graph()
            .search(&self.mem_search)
            .iter()
            .map(|n| (n.id, n.content.clone(), n.kind, n.confidence))
            .collect();

        let mut to_delete: Option<MemoryId> = None;
        let mut to_save: Option<(MemoryId, String)> = None;
        let mut open = true;

        egui::Window::new("Memories")
            .open(&mut open)
            .default_width(340.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label("search:");
                    ui.text_edit_singleline(&mut self.mem_search);
                });
                ui.separator();
                if items.is_empty() {
                    ui.label("(nothing remembered yet)");
                }
                egui::ScrollArea::vertical()
                    .max_height(320.0)
                    .show(ui, |ui| {
                        for (id, content, kind, confidence) in &items {
                            ui.group(|ui| {
                                if self.editing == Some(*id) {
                                    ui.text_edit_singleline(&mut self.edit_buf);
                                    ui.horizontal(|ui| {
                                        if ui.button("Save").clicked() {
                                            to_save = Some((*id, self.edit_buf.clone()));
                                        }
                                        if ui.button("Cancel").clicked() {
                                            self.editing = None;
                                        }
                                    });
                                } else {
                                    ui.label(format!("[{kind:?}] {content}"));
                                    ui.horizontal(|ui| {
                                        ui.small(format!("confidence {confidence:.2}"));
                                        if ui.button("Edit").clicked() {
                                            self.editing = Some(*id);
                                            self.edit_buf = content.clone();
                                        }
                                        if ui.button("Delete").clicked() {
                                            to_delete = Some(*id);
                                        }
                                    });
                                }
                            });
                        }
                    });
            });
        if !open {
            self.show_memories = false;
        }

        let changed = to_delete.is_some() || to_save.is_some();
        if let Some(id) = to_delete {
            brain.agent.graph_mut().remove_node(id);
            if self.editing == Some(id) {
                self.editing = None;
            }
        }
        if let Some((id, new_content)) = to_save {
            // Rebuild the node with the new content and a fresh embedding.
            let fields = brain
                .agent
                .graph()
                .node(id)
                .map(|n| (n.kind, n.created_at, n.provenance, n.confidence));
            if let Some((kind, created_at, provenance, confidence)) = fields {
                let embedding = brain.embedder.embed(&new_content);
                let updated =
                    MemoryNode::new(id, kind, new_content, created_at, provenance, confidence)
                        .with_embedding(embedding);
                brain.agent.graph_mut().insert_node(updated);
            }
            self.editing = None;
        }
        // Curation edits change what Jaxson knows — persist them immediately.
        if changed {
            self.memory_count = brain.agent.graph().node_count();
            self.persist.save(brain.agent.graph());
        }
    }

    /// Load the Ollama model at `index` as the active brain (needs the `llama` feature).
    fn load_selected(&mut self, index: usize) {
        let name = self.models[index].name.clone();
        let path = self.models[index].path.clone();
        #[cfg(feature = "llama")]
        {
            use jaxson_llm::backends::{load_generator_and_embedder, LlamaConfig};
            match load_generator_and_embedder(&LlamaConfig {
                model_path: path.clone(),
                ..Default::default()
            }) {
                Ok((generator, shared)) => {
                    if let Some(brain) = self.brain.as_mut() {
                        brain.model = Box::new(generator);
                        // Match the chat format to the model so it doesn't emit garbled
                        // control tokens (e.g. llama3.1 needs the Llama-3 template).
                        brain
                            .agent
                            .set_template(ChatTemplate::for_model_name(&name));
                    } else {
                        return; // a turn is generating; the picker is disabled anyway
                    }
                    self.base_embedder = Some(shared);
                    self.selected = Some(index);
                    self.status = format!("model: {name}");
                    tracing::info!(model = %name, "loaded model");
                    // Rebuild the active embedder against the new chat model (reusing it
                    // for "same as chat", or keeping the dedicated embedding model).
                    self.apply_embed_selection();
                }
                Err(e) => {
                    tracing::error!(model = %name, error = %e, "failed to load model");
                    self.status = format!("failed to load {name}: {e}");
                }
            }
        }
        #[cfg(not(feature = "llama"))]
        {
            let _ = path;
            self.status = format!("rebuild with --features llama to load {name}");
        }
    }

    /// Pick the embedding model (`None` = same as the chat model), then rebuild the
    /// active embedder. Switching never reloads the chat model.
    #[cfg(feature = "llama")]
    fn select_embed(&mut self, sel: Option<usize>) {
        self.embed_selected = sel;
        self.apply_embed_selection();
    }

    /// Rebuild the active embedder from the current selection and the chat model's shared
    /// embedder. Memories stored under a different embedder won't match this one's vector
    /// space, but cosine handles that gracefully (no match, no crash).
    #[cfg(feature = "llama")]
    fn apply_embed_selection(&mut self) {
        // Clone the shared embedder (a cheap Arc bump) so we don't hold a borrow of
        // `base_embedder` while also mutably borrowing `brain`.
        let Some(base) = self.base_embedder.clone() else {
            return;
        };
        let embedder = active_embedder(&self.models, self.embed_selected, &base);
        if let Some(brain) = self.brain.as_mut() {
            brain.embedder = embedder;
        }
        let label = self
            .embed_selected
            .map(|i| self.models[i].name.as_str())
            .unwrap_or("same as chat");
        tracing::info!(embed = %label, "embedder set");
    }

    /// Push-to-talk toggle: start recording, or stop and transcribe what was captured.
    #[cfg(feature = "whisper")]
    fn toggle_recording(&mut self) {
        if self.recorder.is_some() {
            self.stop_and_transcribe();
        } else if let Err(e) = self.start_recording() {
            tracing::error!(error = %e, "could not start recording");
            self.mic_status = format!("mic error: {e}");
        }
    }

    /// Open the default microphone and stream samples into a shared buffer until stopped.
    #[cfg(feature = "whisper")]
    fn start_recording(&mut self) -> Result<(), String> {
        use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

        let device = cpal::default_host()
            .default_input_device()
            .ok_or("no microphone found")?;
        let config = device.default_input_config().map_err(|e| e.to_string())?;
        let sample_rate = config.sample_rate().0;
        let channels = config.channels();
        let buffer = std::sync::Arc::new(std::sync::Mutex::new(Vec::<f32>::new()));
        let sink = buffer.clone();
        let stream_config: cpal::StreamConfig = config.clone().into();
        let err_fn = |e| tracing::error!(error = %e, "audio input stream error");

        // Append incoming samples (as f32) to the buffer on the audio thread.
        let stream = match config.sample_format() {
            cpal::SampleFormat::F32 => device.build_input_stream(
                &stream_config,
                move |data: &[f32], _: &_| sink.lock().unwrap().extend_from_slice(data),
                err_fn,
                None,
            ),
            cpal::SampleFormat::I16 => device.build_input_stream(
                &stream_config,
                move |data: &[i16], _: &_| {
                    sink.lock()
                        .unwrap()
                        .extend(data.iter().map(|s| *s as f32 / 32768.0));
                },
                err_fn,
                None,
            ),
            other => return Err(format!("unsupported mic sample format: {other:?}")),
        }
        .map_err(|e| e.to_string())?;
        stream.play().map_err(|e| e.to_string())?;

        self.recorder = Some(Recorder {
            _stream: stream,
            buffer,
            sample_rate,
            channels,
        });
        self.mic_status = "🔴 recording… (click again to stop)".to_string();
        Ok(())
    }

    /// Stop recording, convert the captured audio to 16 kHz mono, transcribe it, and send
    /// the text as the user's turn.
    #[cfg(feature = "whisper")]
    fn stop_and_transcribe(&mut self) {
        use jaxson_perception::{downmix_stereo, Audio, WHISPER_SAMPLE_RATE};

        let Some(rec) = self.recorder.take() else {
            return;
        };
        let raw = std::mem::take(&mut *rec.buffer.lock().unwrap());
        let (sample_rate, channels) = (rec.sample_rate, rec.channels);
        drop(rec); // stops the stream

        // Collapse to mono (downmix stereo, or take the first of >2 channels), then resample.
        let mono = match channels {
            1 => Audio::new(raw, sample_rate),
            2 => downmix_stereo(&raw, sample_rate),
            n => Audio::new(
                raw.iter().step_by(n as usize).copied().collect(),
                sample_rate,
            ),
        };
        let audio = mono.resample_to(WHISPER_SAMPLE_RATE);

        let result = self.stt.as_mut().map(|stt| stt.transcribe(&audio));
        match result {
            Some(Ok(t)) if !t.is_empty() => {
                self.mic_status.clear();
                self.input = t.text;
                self.send();
            }
            Some(Ok(_)) => self.mic_status = "(didn't catch that — try again)".to_string(),
            Some(Err(e)) => {
                tracing::error!(error = %e, "transcription failed");
                self.mic_status = format!("transcription error: {e}");
            }
            None => {}
        }
    }

    /// Dispatch a turn: hand the brain to a worker thread that generates the reply
    /// (streaming tokens back), so the UI thread stays responsive. The brain comes back
    /// via the channel when the turn finishes (see [`poll_turn`](Self::poll_turn)). A no-op
    /// if the input is blank or a turn is already running.
    fn send(&mut self) {
        let input = std::mem::take(&mut self.input);
        let input = input.trim().to_string();
        if input.is_empty() {
            return;
        }
        let Some(mut brain) = self.brain.take() else {
            self.input = input; // busy — keep the text for when we're free
            return;
        };
        self.transcript.push(("You", input.clone()));
        self.turn += 1;
        let now = self.turn;
        self.streaming.clear();

        let (tx, rx) = mpsc::channel::<TurnUpdate>();
        self.pending = Some(rx);
        let ctx = self.egui_ctx.clone();
        std::thread::spawn(move || {
            let started = Instant::now();
            // Take the synthesizer out of the brain so `on_reply` can borrow it without
            // colliding with the `&mut brain.agent` the turn needs; restored afterwards.
            #[cfg(feature = "piper")]
            let mut tts = brain.tts.take();
            let result = {
                // Stream raw token pieces back to the UI as they're produced.
                let mut on_token = |piece: &str| {
                    let _ = tx.send(TurnUpdate::Token(piece.to_string()));
                    ctx.request_repaint();
                };
                // The reply is ready *before* memory extraction (a second model pass that
                // can take many seconds). Synthesize + send it to the player here so speech
                // starts right after the text, not after extraction. Sentence by sentence,
                // so playback begins on the first short sentence. `split_sentences` /
                // `synthesize` strip `*action*` cues so they aren't read aloud.
                let mut on_reply = |_reply: &str| {
                    #[cfg(feature = "piper")]
                    if let Some(tts) = tts.as_mut() {
                        for sentence in jaxson_perception::split_sentences(_reply) {
                            match tts.synthesize(&sentence) {
                                Ok(audio) if !audio.samples.is_empty() => {
                                    let _ = tx.send(TurnUpdate::Speak(audio));
                                    ctx.request_repaint();
                                }
                                Ok(_) => {}
                                Err(e) => tracing::error!(error = %e, "speech synthesis failed"),
                            }
                        }
                    }
                };
                brain.agent.respond_streaming_with_reply(
                    brain.model.as_mut(),
                    brain.embedder.as_ref(),
                    now,
                    &input,
                    &mut on_token,
                    &mut on_reply,
                )
            };
            #[cfg(feature = "piper")]
            {
                brain.tts = tts;
            }
            tracing::info!(elapsed_ms = started.elapsed().as_millis(), "turn complete");
            let _ = tx.send(TurnUpdate::Done(
                Box::new(brain),
                result.map_err(|e| e.to_string()),
            ));
            ctx.request_repaint();
        });
    }

    /// Drain the worker channel: accumulate streamed tokens, and when the turn finishes,
    /// reinstall the brain, record the reply, and persist.
    fn poll_turn(&mut self) {
        let mut done: Option<(Box<Brain>, Result<Turn, String>)> = None;
        if let Some(rx) = &self.pending {
            loop {
                match rx.try_recv() {
                    Ok(TurnUpdate::Token(piece)) => self.streaming.push_str(&piece),
                    // A synthesized sentence arrived — play it now (it queues after anything
                    // still playing) and feed its loudness envelope to the lip-sync driver.
                    #[cfg(feature = "piper")]
                    Ok(TurnUpdate::Speak(audio)) => {
                        if let Some(out) = self.audio.as_ref() {
                            out.play(&audio);
                        }
                        // ~45 ms envelope frames: smooth enough to track speech, coarse
                        // enough to flap the mouth visibly.
                        let frame = (audio.sample_rate as f64 * 0.045) as usize;
                        if frame > 0 {
                            let frame_secs = frame as f64 / audio.sample_rate as f64;
                            self.speech.push(audio.envelope(frame), frame_secs);
                        }
                    }
                    Ok(TurnUpdate::Done(brain, result)) => {
                        done = Some((brain, result));
                        break;
                    }
                    Err(_) => break, // nothing more this frame (or worker gone)
                }
            }
        }
        let Some((brain, result)) = done else {
            return;
        };
        self.brain = Some(*brain);
        self.pending = None;
        self.streaming.clear();
        match result {
            Ok(turn) => {
                tracing::info!(
                    learned = turn.learned,
                    retrieved = turn.retrieved,
                    "reply ready"
                );
                self.transcript.push(("Jaxson", turn.reply));
            }
            Err(e) => {
                tracing::error!(error = %e, "turn failed");
                self.transcript.push(("Jaxson", format!("(error: {e})")));
            }
        }
        if let Some(brain) = &self.brain {
            self.mood = brain.agent.mood();
            self.memory_count = brain.agent.graph().node_count();
            // The turn may have learned new memories — persist the updated graph.
            self.persist.save(brain.agent.graph());
        }
    }

    /// Whether a turn is currently generating (the brain is away on the worker thread).
    fn is_busy(&self) -> bool {
        self.brain.is_none()
    }
}

impl eframe::App for JaxsonApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Pull in any streamed tokens / completion from the generation worker.
        self.poll_turn();
        let busy = self.is_busy();

        // Refresh the face every frame so it animates (blink/gaze) and reflects mood. The
        // mood is cached (updated when a turn finishes), so the face keeps moving even
        // while the brain is away on the worker thread.
        let t = self.start.elapsed().as_secs_f64();
        // Layer in what Jaxson is doing (F2.3): listening (mic open) takes priority over
        // speaking (lip-sync to the playing reply); otherwise idle mood animation.
        #[allow(unused_mut)]
        let mut activity = Activity::Idle;
        #[cfg(feature = "piper")]
        if let Some(level) = self.speech.level() {
            activity = Activity::Speaking { level };
        }
        #[cfg(feature = "whisper")]
        if self.recorder.is_some() {
            activity = Activity::Listening;
        }
        let bitmap = rasterize(&face_with(self.mood, t, activity), FACE_PIXELS);
        self.face_tex
            .set(to_image(&bitmap), egui::TextureOptions::NEAREST);

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add_space(8.0);
            ui.vertical_centered(|ui| {
                ui.image(egui::load::SizedTexture::new(
                    self.face_tex.id(),
                    egui::vec2(250.0, 250.0),
                ));
                ui.label(format!("mood: {:?}", self.mood.dominant_emotion()));
            });

            // Model picker — Ollama models, plus the built-in demo brain.
            ui.horizontal(|ui| {
                let current = match self.selected {
                    Some(i) => self.models[i].name.as_str(),
                    None => "demo brain",
                };
                let mut to_load = None;
                ui.add_enabled_ui(!busy, |ui| {
                    egui::ComboBox::from_label("model")
                        .selected_text(current)
                        .show_ui(ui, |ui| {
                            for (i, model) in self.models.iter().enumerate() {
                                if ui
                                    .selectable_label(self.selected == Some(i), &model.name)
                                    .clicked()
                                {
                                    to_load = Some(i);
                                }
                            }
                        });
                });
                if let Some(i) = to_load {
                    self.load_selected(i);
                }
            });

            // Embedding-model picker — defaults to the chat model, or a separate one
            // (e.g. nomic-embed-text) for better/cheaper retrieval (F1.4b).
            #[cfg(feature = "llama")]
            ui.horizontal(|ui| {
                let current = match self.embed_selected {
                    Some(i) => self.models[i].name.as_str(),
                    None => "(same as chat)",
                };
                let mut choice: Option<Option<usize>> = None;
                ui.add_enabled_ui(!busy, |ui| {
                    egui::ComboBox::from_label("embed")
                        .selected_text(current)
                        .show_ui(ui, |ui| {
                            if ui
                                .selectable_label(self.embed_selected.is_none(), "(same as chat)")
                                .clicked()
                            {
                                choice = Some(None);
                            }
                            for (i, model) in self.models.iter().enumerate() {
                                if ui
                                    .selectable_label(self.embed_selected == Some(i), &model.name)
                                    .clicked()
                                {
                                    choice = Some(Some(i));
                                }
                            }
                        });
                });
                if let Some(sel) = choice {
                    self.select_embed(sel);
                }
            });
            ui.small(self.status.as_str());

            ui.separator();

            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .max_height(180.0)
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    for (who, text) in &self.transcript {
                        let color = if *who == "You" {
                            Color32::from_rgb(0x66, 0xb2, 0xff) // you: blue
                        } else {
                            Color32::from_rgb(0x8f, 0xe3, 0x88) // jaxson: green
                        };
                        // Wrap long messages and keep the text selectable (so replies can
                        // be read fully and copied). The speaker name is the colored tag.
                        ui.horizontal_wrapped(|ui| {
                            ui.spacing_mut().item_spacing.x = 4.0;
                            ui.colored_label(
                                color,
                                egui::RichText::new(format!("{who}:")).strong(),
                            );
                            ui.label(text);
                        });
                        ui.add_space(4.0);
                    }
                    // While a turn runs, show the reply streaming in (or "thinking…" before
                    // the first token).
                    if self.pending.is_some() {
                        let green = Color32::from_rgb(0x8f, 0xe3, 0x88);
                        let preview = if self.streaming.is_empty() {
                            "💭 thinking…"
                        } else {
                            self.streaming.as_str()
                        };
                        ui.horizontal_wrapped(|ui| {
                            ui.spacing_mut().item_spacing.x = 4.0;
                            ui.colored_label(green, egui::RichText::new("Jaxson:").strong());
                            ui.label(preview);
                        });
                    }
                });

            ui.separator();

            ui.horizontal(|ui| {
                // Push-to-talk mic (whisper feature, model loaded): click to record, again
                // to stop + transcribe + send.
                #[cfg(feature = "whisper")]
                if self.stt.is_some() {
                    let label = if self.recorder.is_some() {
                        "⏹"
                    } else {
                        "🎤"
                    };
                    if ui
                        .add_enabled(!busy, egui::Button::new(label))
                        .on_hover_text("Push to talk")
                        .clicked()
                    {
                        self.toggle_recording();
                    }
                }
                let response = ui.add(
                    egui::TextEdit::singleline(&mut self.input)
                        .desired_width(f32::INFINITY)
                        .hint_text("Talk to Jaxson…"),
                );
                let entered =
                    response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                // Can't send while a turn is still generating.
                let can_send = !busy && !self.input.trim().is_empty();
                let clicked = ui
                    .add_enabled(can_send, egui::Button::new("Send"))
                    .clicked();
                if clicked || (entered && can_send) {
                    self.send();
                    ui.memory_mut(|m| m.request_focus(response.id));
                }
            });
            #[cfg(feature = "whisper")]
            if !self.mic_status.is_empty() {
                ui.small(self.mic_status.as_str());
            }

            ui.horizontal(|ui| {
                // Clear the visible chat and the model's short-term context — long-term
                // memory (the graph) is kept. Always available to the child.
                if ui
                    .add_enabled(!busy, egui::Button::new("🧹 Clear chat"))
                    .clicked()
                {
                    if let Some(brain) = self.brain.as_mut() {
                        brain.agent.clear_history();
                    }
                    self.transcript =
                        vec![("Jaxson", "Fresh start! What's on your mind?".to_string())];
                }
                // Parent mode (FR-S3): gates guardrail tuning + memory review.
                let parent_label = if self.parent_unlocked {
                    "🔓 Parent"
                } else {
                    "🔒 Parent"
                };
                if ui.button(parent_label).clicked() {
                    self.parent_open = !self.parent_open;
                    self.parent_status.clear();
                    self.parent_pin.clear();
                }
            });
            // Memory review + guardrail settings live behind the parent passcode, so they're
            // only shown when the panel is open.
            if self.parent_open {
                self.parent_controls(ui, busy);
            }
            ui.small(self.persist.status());
            if !self.export_status.is_empty() {
                ui.small(self.export_status.as_str());
            }
        });

        self.memory_window(ctx);

        // Keep animating between interactions.
        ctx.request_repaint();
    }
}

/// Convert a 1-bit [`Bitmap`] into an egui image: ink = black, background = white.
fn to_image(bitmap: &Bitmap) -> egui::ColorImage {
    let n = bitmap.size();
    let mut pixels = Vec::with_capacity(n * n);
    for y in 0..n {
        for x in 0..n {
            pixels.push(if bitmap.get(x, y) {
                Color32::BLACK
            } else {
                Color32::WHITE
            });
        }
    }
    egui::ColorImage {
        size: [n, n],
        pixels,
    }
}

fn main() -> eframe::Result<()> {
    // Keep the guard alive for the whole run so the file writer flushes on exit.
    let _log_guard = logging::init();
    tracing::info!(
        llama = cfg!(feature = "llama"),
        sqlite = cfg!(feature = "sqlite"),
        whisper = cfg!(feature = "whisper"),
        piper = cfg!(feature = "piper"),
        "Jaxson starting"
    );
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([360.0, 560.0])
            .with_title("Jaxson"),
        ..Default::default()
    };
    eframe::run_native(
        "Jaxson",
        native_options,
        Box::new(|cc| Ok(Box::new(JaxsonApp::new(cc)))),
    )
}
