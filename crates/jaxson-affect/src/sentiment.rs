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

/// Keywords found inside `*action*` cues, with their (valence, arousal) contribution.
/// Matched as substrings so `perk`/`perked`, `excit`/`excited`/`excitement` all hit.
const ACTION_LEXICON: &[(&str, f64, f64)] = &[
    ("perk", 0.6, 0.9),
    ("excit", 0.8, 0.9),
    ("wag", 0.9, 0.8),
    ("bounce", 0.7, 0.9),
    ("eager", 0.6, 0.8),
    ("curious", 0.4, 0.7),
    ("curiosity", 0.4, 0.7),
    ("lean", 0.3, 0.5),
    ("grin", 0.9, 0.6),
    ("beam", 0.9, 0.6),
    ("smil", 0.7, 0.4),
    ("delight", 0.9, 0.7),
    ("glee", 0.9, 0.8),
    ("droop", -0.5, -0.3),
    ("sigh", -0.4, -0.2),
    ("frown", -0.7, 0.2),
    ("slump", -0.6, -0.4),
    ("whimper", -0.7, 0.3),
    ("tuck", -0.6, -0.2),
    ("downcast", -0.6, -0.3),
];

/// The text inside each `*…*` cue.
fn action_spans(text: &str) -> Vec<&str> {
    let mut spans = Vec::new();
    let mut rest = text;
    while let Some(open) = rest.find('*') {
        let after = &rest[open + 1..];
        match after.find('*') {
            Some(close) => {
                spans.push(&after[..close]);
                rest = &after[close + 1..];
            }
            None => break,
        }
    }
    spans
}

/// Read Jaxson's expressed feeling from the `*action*` cues in its own reply (e.g.
/// `*ears perked up with excitement*` → bright and alert). Returns
/// [`Sentiment::NEUTRAL`] when there are no recognized cues.
pub fn action_sentiment(text: &str) -> Sentiment {
    let mut valence = 0.0;
    let mut arousal = 0.0;
    let mut hits = 0;
    for span in action_spans(text) {
        let lower = span.to_lowercase();
        for (keyword, v, a) in ACTION_LEXICON {
            if lower.contains(keyword) {
                valence += v;
                arousal += a;
                hits += 1;
            }
        }
    }
    if hits == 0 {
        return Sentiment::NEUTRAL;
    }
    Sentiment::new(valence / hits as f64, arousal / hits as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn action_sentiment_reads_excited_cues_as_bright_and_alert() {
        let s = action_sentiment("*ears perked up with excitement*");
        assert!(s.valence > 0.0);
        assert!(s.arousal > 0.5);
    }

    #[test]
    fn action_sentiment_reads_sad_cues_as_negative() {
        let s = action_sentiment("*tail tucked, a quiet whimper*");
        assert!(s.valence < 0.0);
    }

    #[test]
    fn action_sentiment_is_neutral_without_cues() {
        assert_eq!(action_sentiment("just plain text"), Sentiment::NEUTRAL);
    }

    #[test]
    fn action_sentiment_ignores_unrecognized_actions() {
        // A cue with no lexicon keyword contributes nothing.
        assert_eq!(action_sentiment("*waves a hand*"), Sentiment::NEUTRAL);
    }

    #[test]
    fn every_negative_action_keyword_reads_negative() {
        for word in [
            "droop", "sigh", "frown", "slump", "whimper", "tuck", "downcast",
        ] {
            assert!(
                action_sentiment(&format!("*{word}*")).valence < 0.0,
                "{word} should be negative valence"
            );
        }
    }

    #[test]
    fn calm_negative_actions_have_zero_arousal() {
        // These have negative arousal, which Sentiment clamps to 0 — guards the arousal
        // signs in the lexicon.
        for word in ["droop", "sigh", "slump", "tuck", "downcast"] {
            assert_eq!(
                action_sentiment(&format!("*{word}*")).arousal,
                0.0,
                "{word}"
            );
        }
    }

    #[test]
    fn action_sentiment_averages_multiple_keywords() {
        // "*excited grin*" => excit (0.8, 0.9) + grin (0.9, 0.6), averaged over 2 hits.
        let s = action_sentiment("*excited grin*");
        assert!(approx(s.valence, (0.8 + 0.9) / 2.0));
        assert!(approx(s.arousal, (0.9 + 0.6) / 2.0));
    }

    #[test]
    fn action_sentiment_reads_two_separate_cues() {
        // Needs correct span advancement to see the *second* cue.
        let s = action_sentiment("*grin* then *droop*");
        // grin (0.9) + droop (-0.5) over 2 hits => 0.2
        assert!(approx(s.valence, (0.9 - 0.5) / 2.0));
    }

    #[test]
    fn action_spans_extracts_each_cue_exactly() {
        assert_eq!(action_spans("*grin* then *droop*"), vec!["grin", "droop"]);
        assert_eq!(action_spans("no cues here"), Vec::<&str>::new());
        assert_eq!(action_spans("trailing *unclosed"), Vec::<&str>::new());
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
