//! A lightweight, deterministic read of the emotional tone of a piece of text.
//!
//! This is a lexicon stand-in (like the agent's `HashEmbedder`): good enough to make
//! the affect loop run and test without a model, and to be replaced later by a richer
//! analyzer. It is intentionally decoupled from the LLM.

/// The affective reading of an utterance, on the circumplex axes.
///
/// - `valence`: unpleasant (`-1.0`) … pleasant (`+1.0`)
/// - `arousal`: calm (`0.0`) … activated (`1.0`)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sentiment {
    pub valence: f64,
    pub arousal: f64,
}

impl Sentiment {
    /// Neutral, calm.
    pub const NEUTRAL: Sentiment = Sentiment {
        valence: 0.0,
        arousal: 0.0,
    };

    /// Create a sentiment, clamping `valence` to `[-1, 1]` and `arousal` to `[0, 1]`.
    pub fn new(valence: f64, arousal: f64) -> Self {
        Sentiment {
            valence: valence.clamp(-1.0, 1.0),
            arousal: arousal.clamp(0.0, 1.0),
        }
    }
}

const POSITIVE: &[&str] = &[
    "love",
    "like",
    "great",
    "happy",
    "good",
    "wonderful",
    "enjoy",
    "fun",
    "yes",
    "thanks",
    "thank",
    "awesome",
    "nice",
    "glad",
    "excited",
    "amazing",
    "best",
];

const NEGATIVE: &[&str] = &[
    "hate", "sad", "bad", "angry", "no", "awful", "terrible", "annoyed", "upset", "worried",
    "scared", "sorry", "hurt", "afraid", "lonely", "tired", "worst",
];

const INTENSIFIERS: &[&str] = &["really", "so", "very", "super", "totally", "absolutely"];

fn contains(set: &[&str], word: &str) -> bool {
    set.contains(&word)
}

/// Estimate the sentiment of `text` from a small lexicon.
///
/// Valence is the balance of positive vs negative words; arousal rises with
/// exclamation marks, intensifiers, and the sheer amount of emotional language.
pub fn analyze(text: &str) -> Sentiment {
    let lower = text.to_lowercase();
    let tokens: Vec<&str> = lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .collect();

    let positive = tokens.iter().filter(|t| contains(POSITIVE, t)).count();
    let negative = tokens.iter().filter(|t| contains(NEGATIVE, t)).count();
    let emotional = positive + negative;

    let valence = if emotional == 0 {
        0.0
    } else {
        (positive as f64 - negative as f64) / emotional as f64
    };

    let exclamations = text.matches('!').count();
    let intensifiers = tokens.iter().filter(|t| contains(INTENSIFIERS, t)).count();
    let arousal = (exclamations as f64 * 0.3 + intensifiers as f64 * 0.2 + emotional as f64 * 0.1)
        .clamp(0.0, 1.0);

    Sentiment::new(valence, arousal)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn new_clamps_axes() {
        let s = Sentiment::new(5.0, 5.0);
        assert_eq!(s.valence, 1.0);
        assert_eq!(s.arousal, 1.0);
        let s = Sentiment::new(-5.0, -5.0);
        assert_eq!(s.valence, -1.0);
        assert_eq!(s.arousal, 0.0); // arousal floored at 0
    }

    #[test]
    fn neutral_text_reads_neutral() {
        assert_eq!(analyze("the sky is blue today"), Sentiment::NEUTRAL);
    }

    #[test]
    fn positive_text_is_positive_valence() {
        let s = analyze("I love this");
        assert!(approx(s.valence, 1.0));
        assert!(s.arousal > 0.0);
    }

    #[test]
    fn negative_text_is_negative_valence() {
        let s = analyze("I hate this");
        assert!(approx(s.valence, -1.0));
    }

    #[test]
    fn mixed_text_balances_valence() {
        // one positive ("love"), one negative ("hate") => 0.
        assert!(approx(analyze("love and hate").valence, 0.0));
    }

    #[test]
    fn exclamations_and_intensifiers_raise_arousal() {
        let calm = analyze("good");
        let excited = analyze("really good!!");
        assert!(excited.arousal > calm.arousal);
    }

    #[test]
    fn arousal_follows_the_exact_weighting() {
        // "really good!" => 1 exclamation, 1 intensifier, 1 emotional word.
        // arousal = 1*0.3 + 1*0.2 + 1*0.1 = 0.6
        assert!(approx(analyze("really good!").arousal, 0.6));
    }

    #[test]
    fn arousal_is_capped_at_one() {
        assert_eq!(analyze("love love love so very really!!!!!!").arousal, 1.0);
    }

    #[test]
    fn valence_magnitude_reflects_balance() {
        // two positive, one negative => (2-1)/3.
        assert!(approx(analyze("good nice bad").valence, 1.0 / 3.0));
    }
}
