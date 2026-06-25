//! Headless smoke-test for the whisper STT backend (F2.1): transcribe a WAV file. Use it
//! to confirm a whisper model works before wiring the microphone in.
//!
//! ```text
//! cargo run -p jaxson-perception --example whisper_transcribe --features whisper -- \
//!     /path/to/ggml-base.en.bin /path/to/speech.wav
//! ```
//!
//! Get a model from whisper.cpp's releases (e.g. `ggml-base.en.bin`). The WAV should be
//! **16 kHz mono** (16-bit PCM or 32-bit float); stereo is down-mixed automatically.
//! Needs `cmake` + a C/C++ toolchain to build and a model to run.

#[cfg(feature = "whisper")]
fn main() {
    use jaxson_perception::backends::WhisperStt;
    use jaxson_perception::{downmix_stereo, Audio, SpeechToText, WHISPER_SAMPLE_RATE};

    let mut args = std::env::args().skip(1);
    let model = args
        .next()
        .expect("usage: whisper_transcribe <model.bin> <audio.wav>");
    let wav = args
        .next()
        .expect("usage: whisper_transcribe <model.bin> <audio.wav>");

    // Read the WAV into interleaved f32 samples at its native rate.
    let mut reader = hound::WavReader::open(&wav).expect("open wav");
    let spec = reader.spec();
    let samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .map(|s| s.expect("read sample") as f32 / 32768.0)
            .collect(),
        (hound::SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .map(|s| s.expect("read sample"))
            .collect(),
        (fmt, bits) => panic!("unsupported WAV: {fmt:?} {bits}-bit (use 16-bit PCM or f32)"),
    };

    let audio = if spec.channels == 2 {
        downmix_stereo(&samples, spec.sample_rate)
    } else {
        Audio::new(samples, spec.sample_rate)
    };
    eprintln!(
        "audio: {:.1}s, {} Hz, rms {:.3}",
        audio.duration_secs(),
        audio.sample_rate,
        audio.rms()
    );
    if audio.sample_rate != WHISPER_SAMPLE_RATE {
        eprintln!(
            "warning: whisper wants {WHISPER_SAMPLE_RATE} Hz mono — convert first, e.g.\n  \
             afconvert -f WAVE -d LEI16@16000 -c 1 in.wav out.wav"
        );
    }

    eprintln!("Loading {model} …");
    let mut stt = WhisperStt::load(&model).expect("load model");
    let transcript = stt.transcribe(&audio).expect("transcribe");
    println!("\nTRANSCRIPT: {}", transcript.text);
}

#[cfg(not(feature = "whisper"))]
fn main() {
    eprintln!(
        "This example needs the `whisper` feature:\n  \
         cargo run -p jaxson-perception --example whisper_transcribe --features whisper -- <model.bin> <audio.wav>"
    );
}
