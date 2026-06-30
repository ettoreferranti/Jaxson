//! whisper.cpp speech-to-text backend (Metal offload), via the `whisper-rs` crate.
//!
//! Behind the `whisper` feature. Building needs cmake + a C/C++ toolchain; running needs
//! a local whisper model (GGML, e.g. `ggml-base.en.bin`). A thin imperative wrapper around
//! whisper.cpp — the *testable* logic (audio helpers, transcript normalization) lives in
//! the pure modules, so this file intentionally stays small.

use std::path::Path;

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::audio::Audio;
use crate::error::PerceptionError;
use crate::stt::{SpeechToText, Transcript};
use crate::WHISPER_SAMPLE_RATE;

/// A loaded whisper model ready to transcribe 16 kHz mono audio.
pub struct WhisperStt {
    ctx: WhisperContext,
    language: Option<String>,
}

impl WhisperStt {
    /// Load a whisper model from disk. This is the expensive step; reuse the returned
    /// transcriber across utterances.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, PerceptionError> {
        // whisper.cpp + GGML print init banners and per-token decode traces straight to
        // stderr. Route them into whisper-rs's logging hooks — with no `log`/`tracing`
        // backend feature enabled, that effectively silences them — so they don't drown the
        // app's own logs. Idempotent; only the first call takes effect.
        whisper_rs::install_logging_hooks();

        let path = path
            .as_ref()
            .to_str()
            .ok_or_else(|| PerceptionError::ModelLoad("model path is not valid UTF-8".into()))?;
        let ctx = WhisperContext::new_with_params(path, WhisperContextParameters::default())
            .map_err(|e| PerceptionError::ModelLoad(e.to_string()))?;
        Ok(WhisperStt {
            ctx,
            language: Some("en".to_string()),
        })
    }

    /// Set the spoken-language hint (e.g. `"en"`); `None` lets whisper auto-detect.
    pub fn with_language(mut self, language: Option<impl Into<String>>) -> Self {
        self.language = language.map(Into::into);
        self
    }
}

impl SpeechToText for WhisperStt {
    fn transcribe(&mut self, audio: &Audio) -> Result<Transcript, PerceptionError> {
        if audio.sample_rate != WHISPER_SAMPLE_RATE {
            return Err(PerceptionError::Audio(format!(
                "whisper needs {WHISPER_SAMPLE_RATE} Hz mono audio, got {} Hz",
                audio.sample_rate
            )));
        }
        if audio.samples.is_empty() {
            return Ok(Transcript::new(""));
        }

        let mut state = self
            .ctx
            .create_state()
            .map_err(|e| PerceptionError::Backend(e.to_string()))?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(self.language.as_deref());
        // Keep whisper.cpp quiet — we only want the text.
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        state
            .full(params, &audio.samples)
            .map_err(|e| PerceptionError::Backend(e.to_string()))?;

        let mut text = String::new();
        for i in 0..state.full_n_segments() {
            if let Some(segment) = state.get_segment(i) {
                let piece = segment
                    .to_str_lossy()
                    .map_err(|e| PerceptionError::Backend(e.to_string()))?;
                text.push_str(&piece);
            }
        }
        Ok(Transcript::new(text))
    }
}
