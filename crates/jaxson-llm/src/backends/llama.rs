//! `llama.cpp` backend with Metal offload, via the `llama-cpp-2` crate.
//!
//! Behind the `llama` feature. Building it needs cmake + a C/C++ toolchain; running
//! it needs a local GGUF model. Everything here is a thin, imperative wrapper around
//! llama.cpp — the *testable* logic (prompt assembly, config) lives in the pure
//! modules, so this file intentionally stays small.
//!
//! The backend can only be initialized once per process, so it's held in a shared
//! [`OnceLock`]; the loaded model is shared behind an [`Arc`] so generation and
//! embedding (F1.4b) run off the same weights — loaded once — via separate contexts.

use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;

use crate::config::GenerationConfig;
use crate::error::LlmError;
use crate::generator::TextGenerator;

/// How to load the model.
#[derive(Debug, Clone)]
pub struct LlamaConfig {
    /// Path to a local GGUF model file.
    pub model_path: PathBuf,
    /// Layers to offload to the GPU (Metal). Use a large value (e.g. `999`) to offload
    /// the whole model on Apple Silicon; `0` keeps it on the CPU.
    pub n_gpu_layers: u32,
    /// Context window size in tokens.
    pub n_ctx: u32,
}

impl Default for LlamaConfig {
    fn default() -> Self {
        LlamaConfig {
            model_path: PathBuf::new(),
            n_gpu_layers: 999,
            n_ctx: 4096,
        }
    }
}

/// The process-wide llama.cpp backend. `LlamaBackend::init` flips a global flag and
/// errors if called twice, so it's initialized once and shared.
static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();

/// Get (initializing once) the shared backend, with llama.cpp's verbose logs silenced.
fn shared_backend() -> Result<&'static LlamaBackend, LlmError> {
    if let Some(backend) = BACKEND.get() {
        return Ok(backend);
    }
    let mut backend = LlamaBackend::init().map_err(|e| LlmError::Backend(e.to_string()))?;
    backend.void_logs();
    // If another caller raced us, keep whichever landed first — both are equivalent.
    let _ = BACKEND.set(backend);
    Ok(BACKEND.get().expect("backend was just initialized"))
}

/// Load a GGUF's weights once into a shared [`Arc`], offloading to the GPU per `config`.
fn load_model(config: &LlamaConfig) -> Result<Arc<LlamaModel>, LlmError> {
    let backend = shared_backend()?;
    let model_params = LlamaModelParams::default().with_n_gpu_layers(config.n_gpu_layers);
    let model = LlamaModel::load_from_file(backend, &config.model_path, &model_params)
        .map_err(|e| LlmError::ModelLoad(e.to_string()))?;
    Ok(Arc::new(model))
}

/// A loaded `llama.cpp` model ready to generate text.
pub struct LlamaGenerator {
    model: Arc<LlamaModel>,
    n_ctx: u32,
}

impl LlamaGenerator {
    /// Initialize the backend and load the model from disk. This is the expensive
    /// step; reuse the returned generator across turns.
    pub fn load(config: &LlamaConfig) -> Result<Self, LlmError> {
        Ok(LlamaGenerator {
            model: load_model(config)?,
            n_ctx: config.n_ctx,
        })
    }
}

impl TextGenerator for LlamaGenerator {
    fn generate(
        &mut self,
        prompt: &str,
        config: &GenerationConfig,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<String, LlmError> {
        let cfg = config.clone().validated();

        let backend = shared_backend()?;
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(self.n_ctx));
        let mut ctx = self
            .model
            .new_context(backend, ctx_params)
            .map_err(|e| LlmError::Backend(e.to_string()))?;

        let tokens = self
            .model
            .str_to_token(prompt, AddBos::Always)
            .map_err(|e| LlmError::Generation(e.to_string()))?;

        let mut batch = LlamaBatch::new(self.n_ctx as usize, 1);
        let last = tokens.len() as i32 - 1;
        for (i, token) in tokens.iter().enumerate() {
            batch
                .add(*token, i as i32, &[0], i as i32 == last)
                .map_err(|e| LlmError::Generation(e.to_string()))?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| LlmError::Generation(e.to_string()))?;

        let mut sampler = build_sampler(&cfg);
        let mut out = String::new();
        // Position of the next token to append = prompt length, then one per step.
        let prompt_len = batch.n_tokens();
        // A single decoder across the loop so multibyte UTF-8 spanning token
        // boundaries is reassembled correctly.
        let mut decoder = encoding_rs::UTF_8.new_decoder();

        for step in 0..cfg.max_tokens {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);

            if self.model.is_eog_token(token) {
                break;
            }

            let piece = self
                .model
                .token_to_piece(token, &mut decoder, false, None)
                .map_err(|e| LlmError::Generation(e.to_string()))?;
            on_token(&piece);
            out.push_str(&piece);

            // Honor stop strings: trim the matched suffix and finish.
            if let Some(stop) = cfg.stop.iter().find(|s| out.ends_with(s.as_str())) {
                out.truncate(out.len() - stop.len());
                break;
            }

            batch.clear();
            batch
                .add(token, prompt_len + step as i32, &[0], true)
                .map_err(|e| LlmError::Generation(e.to_string()))?;
            ctx.decode(&mut batch)
                .map_err(|e| LlmError::Generation(e.to_string()))?;
        }

        Ok(out)
    }
}

/// Produces real semantic embeddings from the local model (F1.4b): mean-pooled hidden
/// states over the input, L2-normalized. Shares the model's weights with a
/// [`LlamaGenerator`] when built via [`load_generator_and_embedder`].
pub struct LlamaEmbedder {
    model: Arc<LlamaModel>,
    /// Context window for an embedding pass. Memory snippets are short, so this can be
    /// much smaller than the generator's window.
    n_ctx: u32,
}

impl LlamaEmbedder {
    /// Load a model dedicated to embeddings. Prefer [`load_generator_and_embedder`] to
    /// share weights with the chat model instead of loading them twice.
    pub fn load(config: &LlamaConfig) -> Result<Self, LlmError> {
        Ok(LlamaEmbedder {
            model: load_model(config)?,
            n_ctx: config.n_ctx,
        })
    }

    /// Embed `text` into a unit-length vector (mean pooling). Empty/whitespace input
    /// yields an empty vector.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, LlmError> {
        let backend = shared_backend()?;
        let params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(self.n_ctx))
            .with_embeddings(true)
            .with_pooling_type(LlamaPoolingType::Mean);
        let mut ctx = self
            .model
            .new_context(backend, params)
            .map_err(|e| LlmError::Backend(e.to_string()))?;

        let mut tokens = self
            .model
            .str_to_token(text, AddBos::Always)
            .map_err(|e| LlmError::Generation(e.to_string()))?;
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        // Never exceed the context window.
        tokens.truncate(self.n_ctx as usize);

        let mut batch = LlamaBatch::new(tokens.len(), 1);
        batch
            .add_sequence(&tokens, 0, false)
            .map_err(|e| LlmError::Generation(e.to_string()))?;
        ctx.decode(&mut batch)
            .map_err(|e| LlmError::Generation(e.to_string()))?;

        let embedding = ctx
            .embeddings_seq_ith(0)
            .map_err(|e| LlmError::Generation(e.to_string()))?;
        Ok(l2_normalize(embedding))
    }
}

/// Load a GGUF once and return a generator and embedder that share its weights (one copy
/// in memory; separate contexts). This is how the app gets both from a single model.
pub fn load_generator_and_embedder(
    config: &LlamaConfig,
) -> Result<(LlamaGenerator, LlamaEmbedder), LlmError> {
    let model = load_model(config)?;
    Ok((
        LlamaGenerator {
            model: Arc::clone(&model),
            n_ctx: config.n_ctx,
        },
        LlamaEmbedder {
            model,
            n_ctx: config.n_ctx,
        },
    ))
}

/// L2-normalize a vector so cosine similarity is just a dot product. A zero vector is
/// returned unchanged (it can't be normalized).
fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

/// Build a sampler chain from the decoding config: greedy when temperature is zero,
/// otherwise top-p + temperature + a seeded distribution.
fn build_sampler(cfg: &GenerationConfig) -> LlamaSampler {
    if cfg.temperature <= 0.0 {
        LlamaSampler::greedy()
    } else {
        let seed = cfg.seed.unwrap_or_else(time_seed);
        LlamaSampler::chain_simple([
            LlamaSampler::top_p(cfg.top_p, 1),
            LlamaSampler::temp(cfg.temperature),
            LlamaSampler::dist(seed),
        ])
    }
}

/// A best-effort non-deterministic seed when the caller didn't supply one.
fn time_seed() -> u32 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0)
}
