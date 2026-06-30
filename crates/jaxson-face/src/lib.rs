//! # jaxson-face
//!
//! The *geometry* of Jaxson's deliberately minimal face — two eyes, a mouth, and a
//! pair of reactive ears —
//! as a pure function of [`mood`](jaxson_core::MoodVector) and time. No rendering: a
//! UI shell (the egui app, F1.8b) draws the [`Face`] this produces, and a future
//! hardware bot can drive servos from the same numbers.
//!
//! Keeping it pure means the mapping from feeling to expression — smile from valence,
//! wide eyes from arousal, idle blinks and gaze drift from time — is deterministic and
//! mutation-tested, rather than buried in draw calls.
//!
//! ```
//! use jaxson_core::MoodVector;
//! use jaxson_face::face;
//!
//! let happy = face(MoodVector::new(0.8, 0.5), 1.0);
//! assert!(happy.mouth.curve > 0.0); // smiling
//! assert_eq!(happy.left_eye.openness, happy.right_eye.openness); // symmetric
//! ```

use jaxson_core::MoodVector;

mod raster;
pub use raster::{rasterize, Bitmap};

/// Seconds between idle blinks.
const BLINK_PERIOD: f64 = 4.0;
/// How long a blink takes.
const BLINK_DURATION: f64 = 0.18;
/// Maximum horizontal pupil drift (in eye-radius units).
const GAZE_AMPLITUDE: f64 = 0.15;

/// One eye. `openness` is `0.0` (shut) … `1.0` (wide); pupil offsets are in
/// eye-radius units, roughly `[-GAZE_AMPLITUDE, GAZE_AMPLITUDE]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Eye {
    pub openness: f64,
    pub pupil_dx: f64,
    pub pupil_dy: f64,
}

/// The mouth. `curve` is `-1.0` (frown) … `+1.0` (smile); `openness` is `0.0` … `1.0`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mouth {
    pub curve: f64,
    pub openness: f64,
}

/// Both ears (symmetric). `perk` is `-1.0` (drooped) … `0.0` (neutral) … `+1.0`
/// (perked up, alert/excited).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ears {
    pub perk: f64,
}

/// A full facial pose. The two eyes and ears move together (the face is symmetric).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Face {
    pub left_eye: Eye,
    pub right_eye: Eye,
    pub mouth: Mouth,
    pub ears: Ears,
}

/// Resting eye openness from arousal: calm is relaxed, alert is wide. Maps arousal
/// `[-1, 1]` to openness `[0.4, 1.0]`.
pub fn base_eye_openness(arousal: f64) -> f64 {
    (0.7 + 0.3 * arousal).clamp(0.0, 1.0)
}

/// Blink multiplier at time `t`: `1.0` (open) most of the time, dipping to `0.0`
/// (shut) briefly once per [`BLINK_PERIOD`]. A symmetric close-then-open.
pub fn blink_factor(t: f64) -> f64 {
    let phase = t.rem_euclid(BLINK_PERIOD);
    // `x` runs 0 -> 1 across the blink window and stays at 1 between blinks (the
    // clamp), so the factor traces 1 -> 0 -> 1 during a blink and rests at 1 after.
    let x = (phase / BLINK_DURATION).min(1.0);
    (2.0 * x - 1.0).abs()
}

/// Actual eye openness: resting openness modulated by blinking.
pub fn eye_openness(arousal: f64, t: f64) -> f64 {
    base_eye_openness(arousal) * blink_factor(t)
}

/// Mouth curvature from valence: pleasant smiles up, unpleasant frowns down.
pub fn mouth_curve(valence: f64) -> f64 {
    valence.clamp(-1.0, 1.0)
}

/// How far the mouth opens — only activated (positive-arousal) moods open it. Maps to
/// `[0.0, 0.5]`.
pub fn mouth_openness(arousal: f64) -> f64 {
    arousal.max(0.0) * 0.5
}

/// How perked the ears are from mood: alert/excited (high arousal, pleasant) perks them
/// up, sadness droops them. Result in `[-1.0, 1.0]`.
pub fn ear_perk(mood: MoodVector) -> f64 {
    (0.7 * mood.arousal() + 0.3 * mood.valence()).clamp(-1.0, 1.0)
}

/// Slow idle gaze drift at time `t`, as `(dx, dy)` pupil offsets in eye-radius units.
/// Bounded by `GAZE_AMPLITUDE` horizontally and half that vertically.
pub fn gaze(t: f64) -> (f64, f64) {
    let dx = GAZE_AMPLITUDE * (0.7 * t).sin();
    let dy = 0.5 * GAZE_AMPLITUDE * (0.9 * t + 1.0).sin();
    (dx, dy)
}

/// Widest the mouth opens at peak speech loudness (lip-sync). Bigger than the mood-driven
/// [`mouth_openness`] range so talking reads clearly.
const SPEAKING_MOUTH_MAX: f64 = 0.6;
/// How much wider than resting the eyes go while listening — attentive, "I'm all ears".
const LISTENING_EYE_BOOST: f64 = 0.25;
/// Floor on ear perk while listening, so Jaxson looks alert even in a low mood.
const LISTENING_EAR_PERK: f64 = 0.5;

/// What Jaxson is doing right now, layered on top of mood-driven expression so the face
/// follows the conversation (F2.3): talking flaps the mouth, listening widens the eyes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activity {
    /// Just being — mood plus idle blink/gaze only.
    Idle,
    /// Speaking aloud; `level` is the current speech loudness in `[0, 1]` (e.g. from an
    /// audio envelope), driving how far the mouth opens.
    Speaking { level: f64 },
    /// Listening to the user (microphone open): attentive, steady, wide-eyed.
    Listening,
}

/// Mouth openness while speaking, from the current speech `level` (`[0, 1]`, clamped).
pub fn speaking_mouth_openness(level: f64) -> f64 {
    level.clamp(0.0, 1.0) * SPEAKING_MOUTH_MAX
}

/// Eye openness while listening: resting openness widened a touch for an attentive look,
/// still modulated by blinking so Jaxson doesn't stare unnervingly.
pub fn listening_eye_openness(arousal: f64, t: f64) -> f64 {
    (base_eye_openness(arousal) + LISTENING_EYE_BOOST).clamp(0.0, 1.0) * blink_factor(t)
}

/// Assemble the full face for a `mood` at time `t`.
pub fn face(mood: MoodVector, t: f64) -> Face {
    let openness = eye_openness(mood.arousal(), t);
    let (dx, dy) = gaze(t);
    let eye = Eye {
        openness,
        pupil_dx: dx,
        pupil_dy: dy,
    };
    Face {
        left_eye: eye,
        right_eye: eye,
        mouth: Mouth {
            curve: mouth_curve(mood.valence()),
            openness: mouth_openness(mood.arousal()),
        },
        ears: Ears {
            perk: ear_perk(mood),
        },
    }
}

/// Assemble the face for a `mood` at time `t`, layered with what Jaxson is [`Activity`]
/// doing. [`face`] is exactly the [`Activity::Idle`] case.
pub fn face_with(mood: MoodVector, t: f64, activity: Activity) -> Face {
    let mut f = face(mood, t);
    match activity {
        Activity::Idle => {}
        Activity::Speaking { level } => {
            // The mouth tracks the voice; the mood still sets the smile/frown curve.
            f.mouth.openness = speaking_mouth_openness(level);
        }
        Activity::Listening => {
            // Attentive: wide eyes looking straight at the user (no idle drift), ears up.
            let eye = Eye {
                openness: listening_eye_openness(mood.arousal(), t),
                pupil_dx: 0.0,
                pupil_dy: 0.0,
            };
            f.left_eye = eye;
            f.right_eye = eye;
            f.ears.perk = f.ears.perk.max(LISTENING_EAR_PERK);
        }
    }
    f
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn base_eye_openness_rises_with_arousal() {
        assert!(approx(base_eye_openness(0.0), 0.7));
        assert!(approx(base_eye_openness(1.0), 1.0));
        assert!(approx(base_eye_openness(-1.0), 0.4));
    }

    #[test]
    fn base_eye_openness_is_clamped() {
        assert_eq!(base_eye_openness(5.0), 1.0);
        assert_eq!(base_eye_openness(-5.0), 0.0);
    }

    #[test]
    fn eyes_are_open_outside_a_blink() {
        // t = 1.0s is between blinks (period 4s, duration 0.18s).
        assert!(approx(blink_factor(1.0), 1.0));
    }

    #[test]
    fn eyes_are_open_at_the_blink_window_edges() {
        // Start of the window opens fully (also guards the `.abs()`).
        assert!(approx(blink_factor(0.0), 1.0));
        assert!(approx(blink_factor(BLINK_DURATION), 1.0));
    }

    #[test]
    fn eyes_shut_at_the_middle_of_a_blink() {
        // Halfway through the blink window the eyes are fully shut.
        assert!(approx(blink_factor(BLINK_DURATION / 2.0), 0.0));
    }

    #[test]
    fn blink_repeats_each_period() {
        assert!(approx(
            blink_factor(BLINK_PERIOD + BLINK_DURATION / 2.0),
            0.0
        ));
        assert!(approx(blink_factor(1.0), blink_factor(1.0 + BLINK_PERIOD)));
    }

    #[test]
    fn eye_openness_combines_arousal_and_blink() {
        // Mid-blink shuts the eye regardless of arousal.
        assert!(approx(eye_openness(1.0, BLINK_DURATION / 2.0), 0.0));
        // Open phase shows the resting openness.
        assert!(approx(eye_openness(0.0, 1.0), 0.7));
    }

    #[test]
    fn mouth_curve_tracks_valence_and_clamps() {
        assert!(approx(mouth_curve(0.5), 0.5));
        assert!(approx(mouth_curve(-0.5), -0.5));
        assert_eq!(mouth_curve(9.0), 1.0);
        assert_eq!(mouth_curve(-9.0), -1.0);
    }

    #[test]
    fn mouth_opens_only_for_positive_arousal() {
        assert!(approx(mouth_openness(1.0), 0.5));
        assert!(approx(mouth_openness(0.4), 0.2));
        assert_eq!(mouth_openness(-1.0), 0.0);
    }

    #[test]
    fn gaze_is_deterministic_and_bounded() {
        assert_eq!(gaze(2.5), gaze(2.5));
        for &t in &[0.0, 1.3, 7.7, 100.0] {
            let (dx, dy) = gaze(t);
            assert!(dx.abs() <= GAZE_AMPLITUDE + 1e-9);
            assert!(dy.abs() <= 0.5 * GAZE_AMPLITUDE + 1e-9);
        }
    }

    #[test]
    fn gaze_dx_is_zero_at_origin() {
        assert!(approx(gaze(0.0).0, 0.0));
    }

    #[test]
    fn gaze_matches_the_exact_curve() {
        // Pins both axes' coefficients and the dy phase offset.
        let (dx, dy) = gaze(1.0);
        assert!(approx(dx, 0.15 * (0.7_f64).sin()));
        assert!(approx(dy, 0.5 * 0.15 * (0.9_f64 + 1.0).sin()));
    }

    #[test]
    fn ear_perk_rises_with_arousal_and_droops_when_sad() {
        assert!(ear_perk(MoodVector::new(0.8, 0.8)) > 0.5); // excited -> perked
        assert!(ear_perk(MoodVector::new(-0.8, -0.6)) < 0.0); // sad -> drooped
        assert!(approx(ear_perk(MoodVector::NEUTRAL), 0.0));
    }

    #[test]
    fn ear_perk_is_clamped() {
        assert_eq!(ear_perk(MoodVector::new(1.0, 1.0)), 1.0);
        assert_eq!(ear_perk(MoodVector::new(-1.0, -1.0)), -1.0);
    }

    #[test]
    fn face_is_symmetric_and_expresses_mood() {
        let f = face(MoodVector::new(0.8, 0.5), 1.0);
        assert_eq!(f.left_eye, f.right_eye);
        assert!(approx(f.mouth.curve, 0.8));
        assert!(f.mouth.openness > 0.0);
        assert!(f.ears.perk > 0.0); // happy+alert perks the ears
    }

    #[test]
    fn sad_mood_frowns_and_droops_ears() {
        let f = face(MoodVector::new(-0.7, -0.5), 1.0);
        assert!(f.mouth.curve < 0.0);
        assert_eq!(f.mouth.openness, 0.0); // low arousal keeps the mouth closed
        assert!(f.ears.perk < 0.0);
    }

    #[test]
    fn speaking_mouth_openness_scales_level_and_clamps() {
        assert!(approx(speaking_mouth_openness(0.0), 0.0));
        assert!(approx(speaking_mouth_openness(1.0), SPEAKING_MOUTH_MAX));
        assert!(approx(
            speaking_mouth_openness(0.5),
            SPEAKING_MOUTH_MAX * 0.5
        ));
        assert_eq!(speaking_mouth_openness(9.0), SPEAKING_MOUTH_MAX); // clamp high
        assert_eq!(speaking_mouth_openness(-9.0), 0.0); // clamp low
    }

    #[test]
    fn listening_widens_the_eyes_beyond_resting() {
        // Outside a blink, listening is wider than the same mood's resting openness.
        assert!(listening_eye_openness(0.0, 1.0) > base_eye_openness(0.0));
        // A blink still shuts them (no creepy stare).
        assert!(approx(
            listening_eye_openness(0.0, BLINK_DURATION / 2.0),
            0.0
        ));
        // Clamped at fully open.
        assert_eq!(listening_eye_openness(1.0, 1.0), 1.0);
    }

    #[test]
    fn idle_activity_equals_plain_face() {
        let mood = MoodVector::new(0.3, 0.4);
        assert_eq!(face_with(mood, 1.0, Activity::Idle), face(mood, 1.0));
    }

    #[test]
    fn speaking_drives_mouth_from_level_not_mood() {
        // A calm (low-arousal) mood would keep the mouth shut; speaking opens it anyway,
        // and the mood's curve is preserved.
        let mood = MoodVector::new(0.6, -0.5);
        let f = face_with(mood, 1.0, Activity::Speaking { level: 1.0 });
        assert!(approx(f.mouth.openness, SPEAKING_MOUTH_MAX));
        assert!(approx(f.mouth.curve, 0.6));
        // Zero level closes the mouth even though we're "speaking".
        let quiet = face_with(mood, 1.0, Activity::Speaking { level: 0.0 });
        assert!(approx(quiet.mouth.openness, 0.0));
    }

    #[test]
    fn listening_widens_eyes_centers_gaze_and_perks_ears() {
        // A sad mood would droop the ears and drift the gaze; listening overrides both.
        let mood = MoodVector::new(-0.6, -0.4);
        let f = face_with(mood, 1.0, Activity::Listening);
        assert!(f.left_eye.openness > face(mood, 1.0).left_eye.openness);
        assert_eq!(f.left_eye.pupil_dx, 0.0);
        assert_eq!(f.left_eye.pupil_dy, 0.0);
        assert_eq!(f.left_eye, f.right_eye);
        assert!(f.ears.perk >= LISTENING_EAR_PERK);
    }
}
