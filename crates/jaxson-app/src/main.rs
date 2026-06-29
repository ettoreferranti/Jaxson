//! Jaxson's desktop shell: an egui window showing the animated face above a chat box.
//!
//! The face is the `jaxson-face` rasterizer's bitmap, refreshed every frame so it
//! blinks and drifts, and its expression follows the agent's mood. The chat is wired to
//! a mock-backed [`Agent`]: replies are canned (no model yet — that's `--features
//! llama`), but the *face reacts live* to the sentiment of whatever you type.
//!
//! Run: `cargo run --manifest-path crates/jaxson-app/Cargo.toml`

use std::time::Instant;

use eframe::egui;
use egui::Color32;

mod logging;
mod persist;

use jaxson_agent::{Agent, AgentConfig, Embedder, HashEmbedder};
use jaxson_core::MoodVector;
use jaxson_face::{face, rasterize, Bitmap};
use jaxson_llm::ollama::{self, OllamaModel};
use jaxson_llm::{ChatTemplate, GenerationConfig, LlmError, TextGenerator};
use jaxson_memory::{MemoryId, MemoryKind, MemoryNode};

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
    model: Box<dyn TextGenerator>,
    embedder: Box<dyn Embedder>,
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
) -> Box<dyn Embedder> {
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
    match jaxson_perception::backends::WhisperStt::load(&path) {
        Ok(stt) => {
            tracing::info!(model = %path, "loaded whisper model");
            Some(Box::new(stt))
        }
        Err(e) => {
            tracing::error!(model = %path, error = %e, "failed to load whisper model");
            None
        }
    }
}

struct JaxsonApp {
    agent: Agent,
    model: Box<dyn TextGenerator>,
    embedder: Box<dyn Embedder>,
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
    // Memory inspector state.
    show_memories: bool,
    mem_search: String,
    editing: Option<MemoryId>,
    edit_buf: String,
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
        JaxsonApp {
            agent: Agent::with_graph(PERSONA, graph).with_config(AgentConfig {
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
            }),
            model: boot.model,
            embedder: boot.embedder,
            #[cfg(feature = "llama")]
            base_embedder: boot.base_embedder,
            #[cfg(feature = "llama")]
            embed_selected: boot.embed_selected,
            models,
            selected: boot.selected,
            status: boot.status,
            transcript: vec![("Jaxson", "Hi! I'm Jaxson. What's your name?".to_string())],
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
            show_memories: false,
            mem_search: String::new(),
            editing: None,
            edit_buf: String::new(),
        }
    }

    /// The memory inspector window: browse, search, edit, and delete what Jaxson knows.
    fn memory_window(&mut self, ctx: &egui::Context) {
        if !self.show_memories {
            return;
        }
        // Snapshot the (filtered) memories as owned data so we can mutate the graph after.
        let items: Vec<(MemoryId, String, MemoryKind, f32)> = self
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
            self.agent.graph_mut().remove_node(id);
            if self.editing == Some(id) {
                self.editing = None;
            }
        }
        if let Some((id, new_content)) = to_save {
            // Rebuild the node with the new content and a fresh embedding.
            let fields = self
                .agent
                .graph()
                .node(id)
                .map(|n| (n.kind, n.created_at, n.provenance, n.confidence));
            if let Some((kind, created_at, provenance, confidence)) = fields {
                let embedding = self.embedder.embed(&new_content);
                let updated =
                    MemoryNode::new(id, kind, new_content, created_at, provenance, confidence)
                        .with_embedding(embedding);
                self.agent.graph_mut().insert_node(updated);
            }
            self.editing = None;
        }
        // Curation edits change what Jaxson knows — persist them immediately.
        if changed {
            self.persist.save(self.agent.graph());
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
                    self.model = Box::new(generator);
                    self.base_embedder = Some(shared);
                    // Match the chat format to the model so it doesn't emit garbled
                    // control tokens (e.g. llama3.1 needs the Llama-3 template).
                    self.agent.set_template(ChatTemplate::for_model_name(&name));
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
        if let Some(base) = &self.base_embedder {
            self.embedder = active_embedder(&self.models, self.embed_selected, base);
            let label = self
                .embed_selected
                .map(|i| self.models[i].name.as_str())
                .unwrap_or("same as chat");
            tracing::info!(embed = %label, "embedder set");
        }
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
        let config = device
            .default_input_config()
            .map_err(|e| e.to_string())?;
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
            n => Audio::new(raw.iter().step_by(n as usize).copied().collect(), sample_rate),
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

    fn send(&mut self) {
        let input = std::mem::take(&mut self.input);
        let input = input.trim().to_string();
        if input.is_empty() {
            return;
        }
        self.transcript.push(("You", input.clone()));
        self.turn += 1;
        let started = Instant::now();
        let result =
            self.agent
                .respond(self.model.as_mut(), self.embedder.as_ref(), self.turn, &input);
        let elapsed_ms = started.elapsed().as_millis();
        match result {
            Ok(turn) => {
                tracing::info!(
                    elapsed_ms,
                    learned = turn.learned,
                    retrieved = turn.retrieved,
                    reply_chars = turn.reply.chars().count(),
                    "reply generated"
                );
                self.transcript.push(("Jaxson", turn.reply));
            }
            Err(e) => {
                tracing::error!(elapsed_ms, error = %e, "turn failed");
                self.transcript.push(("Jaxson", format!("(error: {e})")));
            }
        }
        // The turn may have learned new memories — persist the updated graph.
        self.persist.save(self.agent.graph());
    }
}

impl eframe::App for JaxsonApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Refresh the face every frame so it animates (blink/gaze) and reflects mood.
        let t = self.start.elapsed().as_secs_f64();
        let bitmap = rasterize(&face(self.agent.mood(), t), FACE_PIXELS);
        self.face_tex
            .set(to_image(&bitmap), egui::TextureOptions::NEAREST);

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add_space(8.0);
            ui.vertical_centered(|ui| {
                ui.image(egui::load::SizedTexture::new(
                    self.face_tex.id(),
                    egui::vec2(250.0, 250.0),
                ));
                ui.label(format!("mood: {:?}", self.agent.mood().dominant_emotion()));
            });

            // Model picker — Ollama models, plus the built-in demo brain.
            ui.horizontal(|ui| {
                let current = match self.selected {
                    Some(i) => self.models[i].name.as_str(),
                    None => "demo brain",
                };
                let mut to_load = None;
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
                            ui.colored_label(color, egui::RichText::new(format!("{who}:")).strong());
                            ui.label(text);
                        });
                        ui.add_space(4.0);
                    }
                });

            ui.separator();

            ui.horizontal(|ui| {
                // Push-to-talk mic (whisper feature, model loaded): click to record, again
                // to stop + transcribe + send.
                #[cfg(feature = "whisper")]
                if self.stt.is_some() {
                    let label = if self.recorder.is_some() { "⏹" } else { "🎤" };
                    if ui.button(label).on_hover_text("Push to talk").clicked() {
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
                let can_send = !self.input.trim().is_empty();
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
                let label = format!("🧠 Memories ({})", self.agent.graph().node_count());
                if ui.button(label).clicked() {
                    self.show_memories = !self.show_memories;
                }
                // Clear the visible chat and the model's short-term context — long-term
                // memory (the graph) is kept.
                if ui.button("🧹 Clear chat").clicked() {
                    self.agent.clear_history();
                    self.transcript =
                        vec![("Jaxson", "Fresh start! What's on your mind?".to_string())];
                }
                // Debug dump: the DB is encrypted, so this is the readable view.
                if ui.button("⬇ Export JSON").clicked() {
                    self.export_status = match persist::export_json(self.agent.graph()) {
                        Ok(path) => format!("exported to {}", path.display()),
                        Err(e) => format!("export failed: {e}"),
                    };
                }
            });
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
