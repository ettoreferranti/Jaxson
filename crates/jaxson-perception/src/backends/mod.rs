//! Speech backends. The native whisper.cpp STT backend (`whisper` feature) and the Piper
//! neural TTS backend (`piper` feature) each pull a C/C++ build and need a model at
//! runtime; the pure seams and the deterministic [`MockStt`](crate::MockStt) /
//! [`MockTts`](crate::MockTts) are always available.

#[cfg(feature = "whisper")]
mod whisper;
#[cfg(feature = "whisper")]
pub use whisper::WhisperStt;

#[cfg(feature = "piper")]
mod piper;
#[cfg(feature = "piper")]
pub use piper::PiperTts;
