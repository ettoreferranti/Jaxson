//! The text-to-speech seam: a [`TextToSpeech`] trait, the pure [`speakable_text`] cleanup
//! that keeps Jaxson's `*action*` stage-cues out of the spoken audio, and a deterministic
//! [`MockTts`] so the loop runs (and tests) without an ONNX voice model.

use crate::audio::Audio;
use crate::error::PerceptionError;

/// Sample rate the deterministic [`MockTts`] tags its silent clips with — a typical Piper
/// voice rate (22.05 kHz), so downstream playback code sees a realistic value in tests.
pub const MOCK_TTS_SAMPLE_RATE: u32 = 22_050;

/// Strip Jaxson's `*action*` stage-cues and collapse whitespace, leaving only the words to
/// be spoken. The face renders cues like `*waves*` / `*tilts head*` from the reply text;
/// the synthesizer must not read them aloud.
///
/// Cues are the spans between paired `*`: splitting on `*` yields alternating outside /
/// inside segments, and we keep only the outside (even-indexed) ones. An unterminated
/// trailing `*…` is treated as an unclosed cue and dropped. The surviving text is
/// whitespace-normalized (runs collapsed, ends trimmed).
pub fn speakable_text(raw: &str) -> String {
    let spoken: String = raw.split('*').step_by(2).collect::<Vec<_>>().join(" ");
    spoken.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Turns text into spoken audio. Object-safe, so the rest of Jaxson can depend on
/// `dyn TextToSpeech` and swap the deterministic [`MockTts`] for the real Piper backend.
pub trait TextToSpeech {
    /// Synthesize `text` into mono PCM [`Audio`]. Implementations speak only
    /// [`speakable_text`] of the input (stage-cues stripped); empty speakable text yields
    /// empty audio.
    fn synthesize(&mut self, text: &str) -> Result<Audio, PerceptionError>;
}

/// A deterministic [`TextToSpeech`] that returns silence whose length scales with the
/// spoken text — for tests, demos, and running the loop without a voice model. Useful for
/// asserting that stage-cues were stripped (shorter clip) without invoking ONNX/espeak.
#[derive(Debug, Clone)]
pub struct MockTts {
    sample_rate: u32,
    samples_per_char: usize,
}

impl MockTts {
    /// A mock at [`MOCK_TTS_SAMPLE_RATE`] emitting `256` silent samples per spoken char.
    pub fn new() -> Self {
        MockTts {
            sample_rate: MOCK_TTS_SAMPLE_RATE,
            samples_per_char: 256,
        }
    }

    /// Override the tagged sample rate.
    pub fn with_sample_rate(mut self, sample_rate: u32) -> Self {
        self.sample_rate = sample_rate;
        self
    }
}

impl Default for MockTts {
    fn default() -> Self {
        Self::new()
    }
}

impl TextToSpeech for MockTts {
    fn synthesize(&mut self, text: &str) -> Result<Audio, PerceptionError> {
        let chars = speakable_text(text).chars().count();
        Ok(Audio::new(
            vec![0.0; chars * self.samples_per_char],
            self.sample_rate,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speakable_strips_action_cues() {
        assert_eq!(
            speakable_text("Hi there *waves happily* friend"),
            "Hi there friend"
        );
    }

    #[test]
    fn speakable_strips_multiple_cues_and_normalizes_whitespace() {
        assert_eq!(
            speakable_text("*beep*  Hello\n\t*boop* world *spins*"),
            "Hello world"
        );
    }

    #[test]
    fn speakable_drops_unterminated_trailing_cue() {
        // A lone opening `*` starts a cue that never closes — everything after it goes.
        assert_eq!(speakable_text("real words *unclosed cue"), "real words");
    }

    #[test]
    fn speakable_keeps_text_with_no_cues() {
        assert_eq!(speakable_text("  just   words \n"), "just words");
    }

    #[test]
    fn speakable_of_only_a_cue_is_empty() {
        assert_eq!(speakable_text("*shrugs*"), "");
    }

    #[test]
    fn mock_length_scales_with_spoken_chars() {
        let mut tts = MockTts::new();
        // 5 spoken chars ("hello"), cue stripped → 5 * 256 samples.
        let audio = tts.synthesize("hello *waves*").unwrap();
        assert_eq!(audio.samples.len(), 5 * 256);
        assert_eq!(audio.sample_rate, MOCK_TTS_SAMPLE_RATE);
    }

    #[test]
    fn mock_of_cue_only_text_is_empty_audio() {
        let mut tts = MockTts::new();
        assert!(tts.synthesize("*blinks*").unwrap().samples.is_empty());
    }

    #[test]
    fn mock_honors_overridden_sample_rate() {
        let mut tts = MockTts::new().with_sample_rate(16_000);
        assert_eq!(tts.synthesize("hi").unwrap().sample_rate, 16_000);
    }
}
