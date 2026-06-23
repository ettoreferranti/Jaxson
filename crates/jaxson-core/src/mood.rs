use serde::{Deserialize, Serialize};

/// A discrete emotion the face can render, derived from a [`MoodVector`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Emotion {
    Neutral,
    Happy,
    Content,
    Sad,
    Upset,
}

/// A point in a circumplex model of affect.
///
/// Jaxson's affect engine produces a `MoodVector`; the face renders from it. Using a
/// continuous valence/arousal space (rather than a fixed list of emotions) lets the
/// face blend and transition smoothly, and lets us *derive* a discrete dominant
/// [`Emotion`] when one is wanted.
///
/// - `valence`: unpleasant (`-1.0`) … pleasant (`+1.0`)
/// - `arousal`: calm/sleepy (`-1.0`) … excited/alert (`+1.0`)
///
/// Both components are always clamped to `[-1.0, 1.0]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MoodVector {
    valence: f64,
    arousal: f64,
}

impl MoodVector {
    /// Neutral, calm, alert resting mood (the origin).
    pub const NEUTRAL: MoodVector = MoodVector {
        valence: 0.0,
        arousal: 0.0,
    };

    /// Create a mood, clamping both components into `[-1.0, 1.0]`.
    pub fn new(valence: f64, arousal: f64) -> Self {
        MoodVector {
            valence: valence.clamp(-1.0, 1.0),
            arousal: arousal.clamp(-1.0, 1.0),
        }
    }

    /// Pleasantness, in `[-1.0, 1.0]`.
    pub fn valence(&self) -> f64 {
        self.valence
    }

    /// Activation, in `[-1.0, 1.0]`.
    pub fn arousal(&self) -> f64 {
        self.arousal
    }

    /// Linear interpolation toward `target` by `fraction` (clamped to `[0.0, 1.0]`).
    ///
    /// The affect engine uses this to smooth mood transitions so the face never
    /// snaps between expressions.
    pub fn blended(&self, target: &MoodVector, fraction: f64) -> MoodVector {
        let f = fraction.clamp(0.0, 1.0);
        MoodVector::new(
            self.valence + (target.valence - self.valence) * f,
            self.arousal + (target.arousal - self.arousal) * f,
        )
    }

    /// The discrete emotion this mood most closely expresses.
    pub fn dominant_emotion(&self) -> Emotion {
        // Near the origin we read as neutral rather than forcing a quadrant.
        let magnitude = (self.valence * self.valence + self.arousal * self.arousal).sqrt();
        if magnitude < 0.2 {
            return Emotion::Neutral;
        }

        match (self.valence >= 0.0, self.arousal >= 0.0) {
            (true, true) => Emotion::Happy,    // pleasant + activated
            (true, false) => Emotion::Content, // pleasant + calm
            (false, true) => Emotion::Upset,   // unpleasant + activated
            (false, false) => Emotion::Sad,    // unpleasant + low energy
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    #[test]
    fn components_are_clamped() {
        let m = MoodVector::new(5.0, -9.0);
        assert_eq!(m.valence(), 1.0);
        assert_eq!(m.arousal(), -1.0);
    }

    #[test]
    fn neutral_is_origin() {
        assert_eq!(MoodVector::NEUTRAL.valence(), 0.0);
        assert_eq!(MoodVector::NEUTRAL.arousal(), 0.0);
    }

    #[test]
    fn blend_moves_fraction_toward_target() {
        let start = MoodVector::new(0.0, 0.0);
        let target = MoodVector::new(1.0, -1.0);
        let mid = start.blended(&target, 0.5);
        assert!(approx(mid.valence(), 0.5));
        assert!(approx(mid.arousal(), -0.5));
    }

    #[test]
    fn blend_from_nonzero_start_uses_difference_not_sum() {
        // From a non-zero start, `target - self` and `target + self` diverge, so this
        // pins the interpolation formula rather than just its direction.
        let start = MoodVector::new(0.4, -0.2);
        let target = MoodVector::new(0.8, 0.6);
        let mid = start.blended(&target, 0.5);
        // valence: 0.4 + (0.8 - 0.4) * 0.5 = 0.6
        assert!(approx(mid.valence(), 0.6));
        // arousal: -0.2 + (0.6 - (-0.2)) * 0.5 = 0.2
        assert!(approx(mid.arousal(), 0.2));
    }

    #[test]
    fn blend_fraction_is_clamped() {
        let start = MoodVector::new(0.0, 0.0);
        let target = MoodVector::new(1.0, 1.0);
        assert_eq!(start.blended(&target, 5.0), target);
        assert_eq!(start.blended(&target, -3.0), start);
    }

    #[test]
    fn dominant_emotion_is_neutral_near_origin() {
        assert_eq!(
            MoodVector::new(0.1, 0.1).dominant_emotion(),
            Emotion::Neutral
        );
    }

    #[test]
    fn dominant_emotion_maps_each_quadrant() {
        let cases = [
            (0.8, 0.8, Emotion::Happy),
            (0.8, -0.8, Emotion::Content),
            (-0.8, 0.8, Emotion::Upset),
            (-0.8, -0.8, Emotion::Sad),
        ];
        for (v, a, expected) in cases {
            assert_eq!(
                MoodVector::new(v, a).dominant_emotion(),
                expected,
                "v={v} a={a}"
            );
        }
    }

    #[test]
    fn just_outside_origin_is_not_neutral() {
        // A mood just past the 0.2 magnitude ring should snap to a quadrant.
        let m = MoodVector::new(0.2, 0.2);
        assert_ne!(m.dominant_emotion(), Emotion::Neutral);
        assert_eq!(m.dominant_emotion(), Emotion::Happy);
    }

    #[test]
    fn neutral_ring_boundary_is_exclusive() {
        // magnitude exactly 0.2 must NOT read as neutral (the check is `< 0.2`).
        let m = MoodVector::new(0.2, 0.0);
        assert_eq!(m.dominant_emotion(), Emotion::Happy);
    }

    #[test]
    fn round_trips_through_serde() {
        let original = MoodVector::new(0.42, -0.17);
        let json = serde_json::to_string(&original).unwrap();
        let decoded: MoodVector = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, original);
    }
}
