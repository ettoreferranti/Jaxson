//! Mono PCM audio plus small pure helpers for the speech pipeline. These run before the
//! native backend (downmix the mic's stereo to mono, gauge loudness for push-to-talk,
//! trim dead air), so they're kept here — pure and mutation-graded.

/// The sample rate whisper.cpp expects: 16 kHz, mono, 32-bit float.
pub const WHISPER_SAMPLE_RATE: u32 = 16_000;

/// A block of mono 32-bit-float PCM samples at a known sample rate.
#[derive(Debug, Clone, PartialEq)]
pub struct Audio {
    /// Mono samples, nominally in `[-1.0, 1.0]`.
    pub samples: Vec<f32>,
    /// Samples per second.
    pub sample_rate: u32,
}

impl Audio {
    /// Wrap mono `samples` recorded at `sample_rate`.
    pub fn new(samples: Vec<f32>, sample_rate: u32) -> Self {
        Audio {
            samples,
            sample_rate,
        }
    }

    /// Length in seconds (`0.0` if the sample rate is zero, avoiding a divide-by-zero).
    pub fn duration_secs(&self) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.samples.len() as f64 / self.sample_rate as f64
    }

    /// Root-mean-square level — a cheap "how loud is this" measure for voice activity and
    /// level meters. `0.0` for an empty clip.
    pub fn rms(&self) -> f32 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let sum_sq: f32 = self.samples.iter().map(|s| s * s).sum();
        (sum_sq / self.samples.len() as f32).sqrt()
    }

    /// Whether the clip's RMS is below `threshold` — effectively silence.
    pub fn is_silent(&self, threshold: f32) -> bool {
        self.rms() < threshold
    }

    /// Drop leading and trailing samples whose magnitude is below `threshold`, keeping the
    /// spoken middle (less audio for whisper to chew on). An all-quiet clip becomes empty.
    pub fn trim_silence(&self, threshold: f32) -> Audio {
        let Some(start) = self.samples.iter().position(|s| s.abs() >= threshold) else {
            return Audio::new(Vec::new(), self.sample_rate);
        };
        let end = self
            .samples
            .iter()
            .rposition(|s| s.abs() >= threshold)
            .unwrap_or(start);
        Audio::new(self.samples[start..=end].to_vec(), self.sample_rate)
    }
}

/// Down-mix interleaved stereo (`L, R, L, R, …`) to mono by averaging each L/R pair. A
/// trailing unpaired sample (odd length) is carried through unchanged.
pub fn downmix_stereo(interleaved: &[f32], sample_rate: u32) -> Audio {
    let mut mono = Vec::with_capacity(interleaved.len().div_ceil(2));
    let mut pairs = interleaved.chunks_exact(2);
    for pair in pairs.by_ref() {
        mono.push((pair[0] + pair[1]) / 2.0);
    }
    if let Some(&last) = pairs.remainder().first() {
        mono.push(last);
    }
    Audio::new(mono, sample_rate)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-6
    }

    #[test]
    fn duration_is_samples_over_rate() {
        let audio = Audio::new(vec![0.0; 16_000], WHISPER_SAMPLE_RATE);
        assert!((audio.duration_secs() - 1.0).abs() < 1e-9);
        // Half a second.
        let half = Audio::new(vec![0.0; 8_000], WHISPER_SAMPLE_RATE);
        assert!((half.duration_secs() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn duration_handles_zero_sample_rate() {
        assert_eq!(Audio::new(vec![0.0; 10], 0).duration_secs(), 0.0);
    }

    #[test]
    fn rms_of_full_scale_square_is_one() {
        let audio = Audio::new(vec![1.0, -1.0, 1.0, -1.0], 16_000);
        assert!(approx(audio.rms(), 1.0));
    }

    #[test]
    fn rms_of_known_signal() {
        // RMS of [0.5, -0.5] = sqrt((0.25 + 0.25)/2) = 0.5.
        assert!(approx(Audio::new(vec![0.5, -0.5], 16_000).rms(), 0.5));
    }

    #[test]
    fn rms_of_empty_is_zero() {
        assert_eq!(Audio::new(Vec::new(), 16_000).rms(), 0.0);
    }

    #[test]
    fn is_silent_compares_rms_to_threshold() {
        let quiet = Audio::new(vec![0.01, -0.01], 16_000); // rms 0.01
        assert!(quiet.is_silent(0.05));
        assert!(!quiet.is_silent(0.005));
        // Exactly at threshold is NOT silent (strict `<`).
        assert!(!quiet.is_silent(0.01));
    }

    #[test]
    fn trim_silence_drops_leading_and_trailing_quiet() {
        let audio = Audio::new(vec![0.0, 0.0, 0.8, -0.7, 0.0, 0.0], 16_000);
        let trimmed = audio.trim_silence(0.1);
        assert_eq!(trimmed.samples, vec![0.8, -0.7]);
        assert_eq!(trimmed.sample_rate, 16_000);
    }

    #[test]
    fn trim_silence_threshold_is_inclusive() {
        // A sample exactly at the threshold magnitude is kept.
        let audio = Audio::new(vec![0.0, 0.1, 0.0], 16_000);
        assert_eq!(audio.trim_silence(0.1).samples, vec![0.1]);
    }

    #[test]
    fn trim_silence_of_all_quiet_is_empty() {
        let audio = Audio::new(vec![0.0, 0.01, -0.02], 16_000);
        assert!(audio.trim_silence(0.1).samples.is_empty());
    }

    #[test]
    fn downmix_averages_each_stereo_pair() {
        let audio = downmix_stereo(&[1.0, 3.0, 2.0, 4.0], 48_000);
        assert_eq!(audio.samples, vec![2.0, 3.0]);
        assert_eq!(audio.sample_rate, 48_000);
    }

    #[test]
    fn downmix_keeps_a_trailing_unpaired_sample() {
        let audio = downmix_stereo(&[1.0, 3.0, 9.0], 48_000);
        assert_eq!(audio.samples, vec![2.0, 9.0]);
    }

    #[test]
    fn downmix_of_empty_is_empty() {
        assert!(downmix_stereo(&[], 48_000).samples.is_empty());
    }
}
