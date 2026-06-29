//! Speech backends. The native whisper.cpp STT backend lives behind the `whisper`
//! feature (it pulls a C/C++ build and needs a model at runtime); the pure seam and the
//! deterministic [`MockStt`](crate::MockStt) are always available.

#[cfg(feature = "whisper")]
mod whisper;
#[cfg(feature = "whisper")]
pub use whisper::WhisperStt;
