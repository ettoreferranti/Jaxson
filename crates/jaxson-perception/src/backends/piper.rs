//! Piper neural text-to-speech backend (VITS via ONNX Runtime), through the `piper-rs`
//! crate.
//!
//! Behind the `piper` feature. Building needs a C/C++ toolchain (espeak-ng is compiled for
//! phonemization); running needs a local Piper voice: an `*.onnx` model plus its
//! `*.onnx.json` config (e.g. from the `rhasspy/piper-voices` set). A thin imperative
//! wrapper around `piper-rs` — the *testable* logic (cue-stripping, whitespace) lives in
//! the pure [`speakable_text`](crate::speakable_text), so this file stays small.
//!
//! Cross-platform by design (ONNX Runtime), so it ports to the future hardware bot — only
//! the seam ([`TextToSpeech`]) is depended on elsewhere, leaving room for other backends.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use piper_rs::Piper;

use crate::audio::Audio;
use crate::error::PerceptionError;
use crate::tts::{speakable_text, TextToSpeech};

/// A loaded Piper voice ready to synthesize speech.
pub struct PiperTts {
    piper: Piper,
    speaker_id: Option<i64>,
    length_scale: Option<f32>,
}

impl PiperTts {
    /// Load a voice from its ONNX model path, deriving the config as `<model>.json` — the
    /// standard Piper layout (`voice.onnx` + `voice.onnx.json`). Use
    /// [`load_with_config`](Self::load_with_config) when the two live apart.
    ///
    /// This is the expensive step; reuse the returned synthesizer across utterances.
    pub fn load(model_path: impl AsRef<Path>) -> Result<Self, PerceptionError> {
        let model_path = model_path.as_ref();
        let mut config = OsString::from(model_path.as_os_str());
        config.push(".json");
        Self::load_with_config(model_path, PathBuf::from(config))
    }

    /// Load a voice from an explicit ONNX model + JSON config path.
    pub fn load_with_config(
        model_path: impl AsRef<Path>,
        config_path: impl AsRef<Path>,
    ) -> Result<Self, PerceptionError> {
        let piper = Piper::new(model_path.as_ref(), config_path.as_ref())
            .map_err(|e| PerceptionError::ModelLoad(e.to_string()))?;
        Ok(PiperTts {
            piper,
            speaker_id: None,
            length_scale: None,
        })
    }

    /// Select a speaker for multi-speaker voices; `None` (the default) uses speaker 0.
    pub fn with_speaker(mut self, speaker_id: Option<i64>) -> Self {
        self.speaker_id = speaker_id;
        self
    }

    /// Override the speaking pace. `length_scale` stretches each phoneme's duration:
    /// `> 1.0` slows speech down (calmer, clearer), `< 1.0` speeds it up. `None` (the
    /// default) uses the voice's own configured value.
    pub fn with_length_scale(mut self, length_scale: Option<f32>) -> Self {
        self.length_scale = length_scale;
        self
    }
}

impl TextToSpeech for PiperTts {
    fn synthesize(&mut self, text: &str) -> Result<Audio, PerceptionError> {
        let spoken = speakable_text(text);
        // Nothing to say (text was empty or only stage-cues): skip espeak/ONNX, which would
        // otherwise choke on empty phonemes. Rate 0 marks the clip as carrying no audio.
        if spoken.is_empty() {
            return Ok(Audio::new(Vec::new(), 0));
        }
        let (samples, sample_rate) = self
            .piper
            .create(
                &spoken,
                false,
                self.speaker_id,
                self.length_scale,
                None,
                None,
            )
            .map_err(|e| PerceptionError::Backend(e.to_string()))?;
        Ok(Audio::new(samples, sample_rate))
    }
}
