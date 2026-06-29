//! # jaxson-perception
//!
//! Local speech perception for Jaxson (v0.2). The pure layer — a [`SpeechToText`] seam,
//! a whitespace-normalizing [`Transcript`], audio helpers ([`Audio`], [`downmix_stereo`]),
//! and a deterministic [`MockStt`] — always compiles and is mutation-graded. The native
//! whisper.cpp (Metal) backend ([`backends::WhisperStt`]) lives behind the `whisper`
//! cargo feature, so default builds, tests, and CI need no C toolchain or model.
//!
//! Everything runs **on-device**; no audio ever leaves the machine.
//!
//! ```
//! use jaxson_perception::{Audio, MockStt, SpeechToText};
//!
//! let mut stt = MockStt::new("hello jaxson");
//! let audio = Audio::new(vec![0.0; 16_000], 16_000);
//! assert_eq!(stt.transcribe(&audio).unwrap().text, "hello jaxson");
//! ```

mod audio;
pub mod backends;
mod error;
mod stt;

pub use audio::{downmix_stereo, Audio, WHISPER_SAMPLE_RATE};
pub use error::PerceptionError;
pub use stt::{MockStt, SpeechToText, Transcript};
