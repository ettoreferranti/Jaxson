//! Headless smoke-test for the Piper TTS backend (F2.2): synthesize text to a WAV file.
//! Use it to confirm a Piper voice works before wiring spoken replies into the app.
//!
//! ```text
//! cargo run -p jaxson-perception --example piper_speak --features piper -- \
//!     /path/to/en_US-libritts_r-medium.onnx "Hi, I'm Jaxson!" out.wav
//! ```
//!
//! Get a voice from the `rhasspy/piper-voices` set — you need BOTH files, side by side:
//! `<voice>.onnx` and `<voice>.onnx.json` (the example derives the `.json` path). A
//! child-friendly English voice (e.g. an `en_US` "high"/"medium" voice) suits Jaxson.
//! Needs a C/C++ toolchain to build (espeak-ng is compiled) and the voice to run.

#[cfg(feature = "piper")]
fn main() {
    use jaxson_perception::backends::PiperTts;
    use jaxson_perception::TextToSpeech;

    let mut args = std::env::args().skip(1);
    let model = args
        .next()
        .expect("usage: piper_speak <voice.onnx> <text> [out.wav]");
    let text = args
        .next()
        .expect("usage: piper_speak <voice.onnx> <text> [out.wav]");
    let out = args.next().unwrap_or_else(|| "out.wav".to_string());

    eprintln!("Loading {model} …");
    let mut tts = PiperTts::load(&model).expect("load voice");
    let audio = tts.synthesize(&text).expect("synthesize");
    eprintln!(
        "synthesized {:.1}s at {} Hz ({} samples)",
        audio.duration_secs(),
        audio.sample_rate,
        audio.samples.len()
    );

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: audio.sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&out, spec).expect("create wav");
    for &s in &audio.samples {
        let clamped = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        writer.write_sample(clamped).expect("write sample");
    }
    writer.finalize().expect("finalize wav");
    println!("Saved to {out}");
}

#[cfg(not(feature = "piper"))]
fn main() {
    eprintln!(
        "This example needs the `piper` feature:\n  \
         cargo run -p jaxson-perception --example piper_speak --features piper -- <voice.onnx> <text> [out.wav]"
    );
}
