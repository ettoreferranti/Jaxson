use jaxson_core::{MoodVector, RelationshipState};

use crate::sentiment::Sentiment;

/// Computes Jaxson's mood from its relationship state and the sentiment of the latest
/// exchange, smoothing over time so the face transitions naturally.
///
/// Decoupled from the LLM's wording (FR-E4): mood comes from internal state + measured
/// sentiment, not from the tokens the model happened to choose.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AffectEngine {
    /// How fast mood moves toward its target each update, in `[0.0, 1.0]`
    /// (0 = never changes, 1 = snaps instantly).
    pub responsiveness: f64,
}

impl Default for AffectEngine {
    fn default() -> Self {
        AffectEngine {
            responsiveness: 0.5,
        }
    }
}

impl AffectEngine {
    /// The mood the latest exchange *points toward*, before smoothing.
    ///
    /// Valence is the exchange's pleasantness lifted by a warmth baseline that grows
    /// with trust and familiarity (a close relationship rests a little brighter);
    /// arousal follows the exchange's activation.
    pub fn target_mood(&self, state: &RelationshipState, sentiment: Sentiment) -> MoodVector {
        let warmth = 0.3 * state.trust() + 0.1 * state.familiarity();
        MoodVector::new(sentiment.valence + warmth, sentiment.arousal)
    }

    /// Smooth `current` mood toward the target implied by `state` + `sentiment`.
    ///
    /// Equivalent to applying `RelationshipEvent::MoodObserved { target, fraction:
    /// responsiveness }` to the state machine — the orchestrator does exactly that so
    /// the relationship state stays the single source of truth.
    pub fn update(
        &self,
        current: MoodVector,
        state: &RelationshipState,
        sentiment: Sentiment,
    ) -> MoodVector {
        current.blended(&self.target_mood(state, sentiment), self.responsiveness)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn target_is_neutral_for_neutral_sentiment_and_fresh_state() {
        let engine = AffectEngine::default();
        let target = engine.target_mood(&RelationshipState::INITIAL, Sentiment::NEUTRAL);
        assert_eq!(target, MoodVector::NEUTRAL);
    }

    #[test]
    fn trust_adds_warmth_to_neutral_exchanges() {
        let engine = AffectEngine::default();
        let state = RelationshipState::new(0.5, 0.0, MoodVector::NEUTRAL);
        // warmth = 0.3 * 0.5 = 0.15
        let target = engine.target_mood(&state, Sentiment::NEUTRAL);
        assert!(approx(target.valence(), 0.15));
    }

    #[test]
    fn familiarity_adds_a_smaller_warmth() {
        let engine = AffectEngine::default();
        let state = RelationshipState::new(0.0, 0.5, MoodVector::NEUTRAL);
        // warmth = 0.1 * 0.5 = 0.05
        assert!(approx(
            engine.target_mood(&state, Sentiment::NEUTRAL).valence(),
            0.05
        ));
    }

    #[test]
    fn sentiment_and_warmth_combine() {
        let engine = AffectEngine::default();
        let state = RelationshipState::new(0.5, 0.0, MoodVector::NEUTRAL); // warmth 0.15
        let target = engine.target_mood(&state, Sentiment::new(0.2, 0.4));
        // valence = 0.2 + 0.15 = 0.35, arousal passes through
        assert!(approx(target.valence(), 0.35));
        assert!(approx(target.arousal(), 0.4));
    }

    #[test]
    fn strong_sentiment_clamps_valence() {
        let engine = AffectEngine::default();
        let state = RelationshipState::new(1.0, 1.0, MoodVector::NEUTRAL); // warmth 0.4
        let target = engine.target_mood(&state, Sentiment::new(1.0, 0.0));
        assert_eq!(target.valence(), 1.0);
    }

    #[test]
    fn update_snaps_to_target_at_full_responsiveness() {
        let engine = AffectEngine {
            responsiveness: 1.0,
        };
        let state = RelationshipState::INITIAL;
        let sentiment = Sentiment::new(1.0, 0.5);
        let mood = engine.update(MoodVector::NEUTRAL, &state, sentiment);
        assert_eq!(mood, engine.target_mood(&state, sentiment));
    }

    #[test]
    fn update_moves_partway_at_half_responsiveness() {
        let engine = AffectEngine {
            responsiveness: 0.5,
        };
        let state = RelationshipState::INITIAL;
        // target valence = 1.0; from 0.0, halfway => 0.5
        let mood = engine.update(MoodVector::NEUTRAL, &state, Sentiment::new(1.0, 0.0));
        assert!(approx(mood.valence(), 0.5));
    }
}
