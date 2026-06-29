//! # jaxson-perception
//!
//! Local speech for Jaxson (v0.2) — both directions. The pure layer always compiles and is
//! mutation-graded: in ([`SpeechToText`] + [`Transcript`]) and out ([`TextToSpeech`] +
//! [`speakable_text`], which keeps `*action*` cues out of the audio), plus audio helpers
//! ([`Audio`], [`downmix_stereo`]) and deterministic mocks ([`MockStt`], [`MockTts`]). The
//! native backends live behind cargo features — whisper.cpp STT ([`backends::WhisperStt`],
//! `whisper`) and Piper neural TTS ([`backends::PiperTts`], `piper`) — so default builds,
//! tests, and CI need no C toolchain or model.
//!
//! Everything runs **on-device**; no audio ever leaves the machine.
//!
//! ```
//! use jaxson_perception::{Audio, MockStt, MockTts, SpeechToText, TextToSpeech};
//!
//! let mut stt = MockStt::new("hello jaxson");
//! let audio = Audio::new(vec![0.0; 16_000], 16_000);
//! assert_eq!(stt.transcribe(&audio).unwrap().text, "hello jaxson");
//!
//! let mut tts = MockTts::new();
//! assert!(!tts.synthesize("hi there").unwrap().samples.is_empty());
//! ```

mod audio;
pub mod backends;
mod error;
mod stt;
mod tts;

pub use audio::{downmix_stereo, Audio, WHISPER_SAMPLE_RATE};
pub use error::PerceptionError;
pub use stt::{MockStt, SpeechToText, Transcript};
pub use tts::{speakable_text, split_sentences, MockTts, TextToSpeech, MOCK_TTS_SAMPLE_RATE};
