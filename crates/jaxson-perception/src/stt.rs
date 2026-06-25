//! The speech-to-text seam: a [`SpeechToText`] trait, a [`Transcript`] result, and a
//! deterministic [`MockStt`] so the loop runs (and tests) without a model or a mic.

use crate::audio::Audio;
use crate::error::PerceptionError;

/// The text recognized from a block of speech.
#[derive(Debug, Clone, PartialEq)]
pub struct Transcript {
    /// The recognized text, whitespace-normalized.
    pub text: String,
}

impl Transcript {
    /// Build a transcript from raw backend text, collapsing runs of whitespace and
    /// trimming the ends — whisper emits per-segment text with leading spaces and
    /// newlines that we don't want surfaced as the user's message.
    pub fn new(raw: impl AsRef<str>) -> Self {
        Transcript {
            text: raw
                .as_ref()
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" "),
        }
    }

    /// Whether nothing was recognized.
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }
}

/// Turns spoken audio into text. Object-safe, so the rest of Jaxson can depend on
/// `dyn SpeechToText` and swap the deterministic [`MockStt`] for the real whisper backend.
pub trait SpeechToText {
    /// Transcribe a block of audio. Implementations expect **16 kHz mono** input (see
    /// [`WHISPER_SAMPLE_RATE`](crate::WHISPER_SAMPLE_RATE)).
    fn transcribe(&mut self, audio: &Audio) -> Result<Transcript, PerceptionError>;
}

/// A deterministic [`SpeechToText`] that ignores the audio and returns canned text — for
/// tests, demos, and running the loop without a whisper model or microphone.
#[derive(Debug, Clone)]
pub struct MockStt {
    text: String,
}

impl MockStt {
    pub fn new(text: impl Into<String>) -> Self {
        MockStt { text: text.into() }
    }
}

impl SpeechToText for MockStt {
    fn transcribe(&mut self, _audio: &Audio) -> Result<Transcript, PerceptionError> {
        Ok(Transcript::new(&self.text))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transcript_normalizes_whitespace() {
        assert_eq!(Transcript::new("  hello   there \n").text, "hello there");
        assert_eq!(Transcript::new("one\n\ttwo").text, "one two");
    }

    #[test]
    fn transcript_is_empty_only_when_blank() {
        assert!(Transcript::new("   \n ").is_empty());
        assert!(!Transcript::new("hi").is_empty());
    }

    #[test]
    fn mock_returns_its_canned_text_regardless_of_audio() {
        let mut stt = MockStt::new("  my name is Ettore ");
        let out = stt.transcribe(&Audio::new(vec![0.0; 4], 16_000)).unwrap();
        assert_eq!(out.text, "my name is Ettore");
    }
}
