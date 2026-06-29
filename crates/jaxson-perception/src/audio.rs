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

    /// A loudness envelope: the RMS of each consecutive `frame`-sample window, then
    /// **peak-normalized** to `[0, 1]` so the loudest moment maps to `1.0`. Sampling this
    /// at playback time drives lip-sync (the mouth opens with the voice). Within a clip the
    /// relative dynamics (pauses dip toward `0`) are preserved; normalizing means a quiet
    /// clip still animates the mouth fully. Empty for an empty clip or `frame == 0`; an
    /// all-silent clip yields all-zero frames.
    pub fn envelope(&self, frame: usize) -> Vec<f32> {
        if frame == 0 || self.samples.is_empty() {
            return Vec::new();
        }
        let mut env: Vec<f32> = self
            .samples
            .chunks(frame)
            .map(|w| {
                let sum_sq: f32 = w.iter().map(|s| s * s).sum();
                (sum_sq / w.len() as f32).sqrt()
            })
            .collect();
        let peak = env.iter().copied().fold(0.0_f32, f32::max);
        if peak > 0.0 {
            for v in &mut env {
                *v /= peak;
            }
        }
        env
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

    /// Resample to `target_rate` by linear interpolation — microphones usually capture at
    /// 44.1/48 kHz, but whisper wants [`WHISPER_SAMPLE_RATE`]. Linear interpolation is
    /// cheap and good enough for speech (whisper is robust to the mild aliasing); a
    /// higher-quality resampler can replace this later. A no-op when already at the target
    /// rate (or for empty/zero-rate input).
    pub fn resample_to(&self, target_rate: u32) -> Audio {
        // Without both rates there's nothing meaningful to do — keep the samples as-is.
        if target_rate == 0 || self.sample_rate == 0 {
            return self.clone();
        }
        // Fewer than two samples: nothing to interpolate between (and same-rate input
        // falls out of the general path unchanged, so it needs no special case).
        if self.samples.len() < 2 {
            return Audio::new(self.samples.clone(), target_rate);
        }
        let ratio = target_rate as f64 / self.sample_rate as f64;
        let out_len = ((self.samples.len() as f64) * ratio).round() as usize;
        let last = self.samples.len() - 1;
        let mut out = Vec::with_capacity(out_len);
        for i in 0..out_len {
            // Where this output sample falls in the source timeline.
            let src_pos = i as f64 / ratio;
            let left = (src_pos.floor() as usize).min(last);
            let right = (left + 1).min(last);
            let frac = (src_pos - left as f64) as f32;
            out.push(self.samples[left] + (self.samples[right] - self.samples[left]) * frac);
        }
        Audio::new(out, target_rate)
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
    fn envelope_is_peak_normalized_rms_per_frame() {
        // Frames of 2: [1,-1] rms 1, [0.5,-0.5] rms 0.5, [0,0] rms 0 → peak 1 → unchanged.
        let audio = Audio::new(vec![1.0, -1.0, 0.5, -0.5, 0.0, 0.0], 16_000);
        let env = audio.envelope(2);
        assert_eq!(env.len(), 3);
        assert!(approx(env[0], 1.0));
        assert!(approx(env[1], 0.5));
        assert!(approx(env[2], 0.0));
    }

    #[test]
    fn envelope_normalizes_so_the_loudest_frame_is_one() {
        // All quiet-ish: rms 0.5 then 0.25 → peak 0.5 → normalized to 1.0 and 0.5.
        let audio = Audio::new(vec![0.5, -0.5, 0.25, -0.25], 16_000);
        let env = audio.envelope(2);
        assert!(approx(env[0], 1.0));
        assert!(approx(env[1], 0.5));
    }

    #[test]
    fn envelope_of_silence_is_all_zero() {
        let env = Audio::new(vec![0.0; 6], 16_000).envelope(2);
        assert_eq!(env, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn envelope_handles_a_short_final_frame() {
        // 5 samples at a constant 0.2, frame 2 → 3 frames (the last is a single sample).
        // Every frame's RMS is 0.2, so all normalize to 1.0 — which only holds if each
        // frame divides by its *own* length (the short frame included), not a fixed one.
        let env = Audio::new(vec![0.2, 0.2, 0.2, 0.2, 0.2], 16_000).envelope(2);
        assert_eq!(env.len(), 3);
        for v in env {
            assert!(approx(v, 1.0), "{v}");
        }
    }

    #[test]
    fn envelope_is_empty_for_empty_clip_or_zero_frame() {
        assert!(Audio::new(Vec::new(), 16_000).envelope(2).is_empty());
        assert!(Audio::new(vec![0.1, 0.2], 16_000).envelope(0).is_empty());
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

    #[test]
    fn resample_to_same_rate_is_a_noop() {
        let audio = Audio::new(vec![0.1, 0.2, 0.3], 16_000);
        assert_eq!(audio.resample_to(16_000), audio);
    }

    #[test]
    fn resample_downsamples_length_by_the_ratio() {
        // 48 kHz → 16 kHz is a 1/3 ratio: 9 samples → 3.
        let audio = Audio::new(vec![0.0; 9], 48_000);
        let out = audio.resample_to(16_000);
        assert_eq!(out.sample_rate, 16_000);
        assert_eq!(out.samples.len(), 3);
    }

    #[test]
    fn resample_upsamples_and_interpolates_between_points() {
        // 1 kHz → 2 kHz doubles the count; the inserted samples interpolate linearly.
        let audio = Audio::new(vec![0.0, 1.0], 1_000);
        let out = audio.resample_to(2_000);
        assert_eq!(out.samples.len(), 4);
        // src positions: 0, 0.5, 1.0, 1.5 → 0.0, 0.5, 1.0, 1.0 (clamped at the end).
        assert!(approx(out.samples[0], 0.0));
        assert!(approx(out.samples[1], 0.5));
        assert!(approx(out.samples[2], 1.0));
    }

    #[test]
    fn resample_interpolation_is_exact_between_nonzero_points() {
        // [0, 2, 6] @ 1 kHz → 2 kHz. Output src positions: 0, .5, 1, 1.5, 2, 2.5.
        let out = Audio::new(vec![0.0, 2.0, 6.0], 1_000).resample_to(2_000);
        assert_eq!(out.samples.len(), 6);
        assert!(approx(out.samples[1], 1.0)); // between 0 and 2 at 0.5
        assert!(approx(out.samples[2], 2.0)); // exactly sample 1
                                              // between samples 1 (=2) and 2 (=6) at frac 0.5 → 2 + (6-2)*0.5 = 4.
                                              // Pins both the interpolation subtraction and the fractional offset.
        assert!(approx(out.samples[3], 4.0));
    }

    #[test]
    fn resample_of_too_few_samples_just_relabels_the_rate() {
        // Empty and single-sample inputs have nothing to interpolate.
        assert_eq!(
            Audio::new(Vec::new(), 48_000).resample_to(16_000),
            Audio::new(Vec::new(), 16_000)
        );
        assert_eq!(
            Audio::new(vec![0.5], 48_000).resample_to(16_000),
            Audio::new(vec![0.5], 16_000)
        );
    }

    #[test]
    fn resample_with_a_zero_rate_keeps_the_original() {
        // Zero target rate → unchanged (can't target nothing).
        let a = Audio::new(vec![0.1, 0.2], 48_000);
        assert_eq!(a.resample_to(0), a);
        // Zero source rate → unchanged (don't divide by it).
        let b = Audio::new(vec![0.1, 0.2], 0);
        assert_eq!(b.resample_to(16_000), b);
    }
}
