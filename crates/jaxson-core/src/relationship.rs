use serde::{Deserialize, Serialize};

use crate::mood::MoodVector;

/// Events that mutate a [`RelationshipState`]. Memories and interactions are
/// translated into these so all state transitions flow through one well-tested path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RelationshipEvent {
    /// Jaxson learned a new fact about the user.
    LearnedFact,
    /// A positive exchange of the given strength `[0.0, 1.0]`.
    PositiveInteraction { strength: f64 },
    /// A negative exchange of the given strength `[0.0, 1.0]`.
    NegativeInteraction { strength: f64 },
    /// Conversation sentiment pulls mood toward `target` by `fraction`.
    MoodObserved { target: MoodVector, fraction: f64 },
    /// A memory was deleted by the user.
    MemoryForgotten,
}

/// The relationship state machine at the core of Jaxson.
///
/// Memories and interactions emit [`RelationshipEvent`]s; [`apply`](Self::apply) is a
/// pure, clamped transition that mutates the scalars below.
///
/// - `trust`: how safe/open the user is with Jaxson, `[0.0, 1.0]`. Gates sensitive
///   topics.
/// - `familiarity`: how much Jaxson knows the user, `[0.0, 1.0]`. Low familiarity
///   biases the agent toward asking onboarding questions.
/// - `mood`: the current affective state that drives the face.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RelationshipState {
    trust: f64,
    familiarity: f64,
    mood: MoodVector,
}

impl RelationshipState {
    /// Familiarity below this threshold means Jaxson should prioritize learning about
    /// the user by asking questions.
    pub const ONBOARDING_FAMILIARITY_THRESHOLD: f64 = 0.3;

    /// Trust below this threshold keeps sensitive topics locked.
    pub const SENSITIVE_TOPIC_TRUST_THRESHOLD: f64 = 0.5;

    /// A fresh relationship: no trust earned, nothing known, neutral mood — Jaxson
    /// starts knowing nothing about the user (the "Ron's defect" premise).
    pub const INITIAL: RelationshipState = RelationshipState {
        trust: 0.0,
        familiarity: 0.0,
        mood: MoodVector::NEUTRAL,
    };

    /// Create a state, clamping `trust` and `familiarity` into `[0.0, 1.0]`.
    pub fn new(trust: f64, familiarity: f64, mood: MoodVector) -> Self {
        RelationshipState {
            trust: trust.clamp(0.0, 1.0),
            familiarity: familiarity.clamp(0.0, 1.0),
            mood,
        }
    }

    /// How safe/open the user is with Jaxson, `[0.0, 1.0]`.
    pub fn trust(&self) -> f64 {
        self.trust
    }

    /// How much Jaxson knows the user, `[0.0, 1.0]`.
    pub fn familiarity(&self) -> f64 {
        self.familiarity
    }

    /// The current affective state.
    pub fn mood(&self) -> MoodVector {
        self.mood
    }

    /// Whether the agent should currently lead with getting-to-know-you questions.
    pub fn should_prioritize_onboarding(&self) -> bool {
        self.familiarity < Self::ONBOARDING_FAMILIARITY_THRESHOLD
    }

    /// Whether sensitive topics are currently unlocked.
    pub fn allows_sensitive_topics(&self) -> bool {
        self.trust >= Self::SENSITIVE_TOPIC_TRUST_THRESHOLD
    }

    /// Apply an event, returning the resulting state (value semantics; deterministic).
    pub fn apply(&self, event: RelationshipEvent) -> RelationshipState {
        match event {
            RelationshipEvent::LearnedFact => {
                // Each new fact nudges familiarity up with diminishing returns.
                RelationshipState::new(
                    self.trust,
                    self.familiarity + (1.0 - self.familiarity) * 0.05,
                    self.mood,
                )
            }
            RelationshipEvent::PositiveInteraction { strength } => {
                let s = strength.clamp(0.0, 1.0);
                RelationshipState::new(
                    self.trust + (1.0 - self.trust) * 0.1 * s,
                    self.familiarity,
                    self.mood.blended(&MoodVector::new(1.0, 0.5), 0.3 * s),
                )
            }
            RelationshipEvent::NegativeInteraction { strength } => {
                let s = strength.clamp(0.0, 1.0);
                RelationshipState::new(
                    self.trust * (1.0 - 0.15 * s),
                    self.familiarity,
                    self.mood.blended(&MoodVector::new(-1.0, 0.3), 0.3 * s),
                )
            }
            RelationshipEvent::MoodObserved { target, fraction } => RelationshipState::new(
                self.trust,
                self.familiarity,
                self.mood.blended(&target, fraction),
            ),
            RelationshipEvent::MemoryForgotten => {
                // Deleting a memory should pull familiarity back (deletion is real).
                RelationshipState::new(self.trust, self.familiarity * 0.95, self.mood)
            }
        }
    }

    /// Apply a sequence of events in order.
    pub fn apply_all(
        &self,
        events: impl IntoIterator<Item = RelationshipEvent>,
    ) -> RelationshipState {
        events
            .into_iter()
            .fold(*self, |state, event| state.apply(event))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    #[test]
    fn initial_state_knows_nothing() {
        let s = RelationshipState::INITIAL;
        assert_eq!(s.trust(), 0.0);
        assert_eq!(s.familiarity(), 0.0);
        assert_eq!(s.mood(), MoodVector::NEUTRAL);
    }

    #[test]
    fn scalars_are_clamped() {
        let s = RelationshipState::new(2.0, -1.0, MoodVector::NEUTRAL);
        assert_eq!(s.trust(), 1.0);
        assert_eq!(s.familiarity(), 0.0);
    }

    #[test]
    fn fresh_agent_prioritizes_onboarding() {
        assert!(RelationshipState::INITIAL.should_prioritize_onboarding());
        let known = RelationshipState::new(0.0, 0.5, MoodVector::NEUTRAL);
        assert!(!known.should_prioritize_onboarding());
    }

    #[test]
    fn sensitive_topics_locked_until_trust_threshold() {
        assert!(!RelationshipState::INITIAL.allows_sensitive_topics());
        let trusted = RelationshipState::new(0.5, 0.0, MoodVector::NEUTRAL);
        assert!(trusted.allows_sensitive_topics());
        let almost = RelationshipState::new(0.49, 0.0, MoodVector::NEUTRAL);
        assert!(!almost.allows_sensitive_topics());
    }

    #[test]
    fn onboarding_boundary_is_exclusive() {
        // At exactly the threshold, onboarding is NOT prioritized (the check is `<`).
        let at = RelationshipState::new(
            0.0,
            RelationshipState::ONBOARDING_FAMILIARITY_THRESHOLD,
            MoodVector::NEUTRAL,
        );
        assert!(!at.should_prioritize_onboarding());
    }

    #[test]
    fn learned_fact_uses_exact_formula() {
        // From 0: 0 + (1 - 0) * 0.05 = 0.05. Pins the formula, not just the sign.
        let once = RelationshipState::INITIAL.apply(RelationshipEvent::LearnedFact);
        assert!(approx(once.familiarity(), 0.05));
    }

    #[test]
    fn learned_fact_raises_familiarity_with_diminishing_returns() {
        let once = RelationshipState::INITIAL.apply(RelationshipEvent::LearnedFact);
        assert!(once.familiarity() > 0.0);
        let high = RelationshipState::new(0.0, 0.9, MoodVector::NEUTRAL);
        let gain_low = once.familiarity() - RelationshipState::INITIAL.familiarity();
        let gain_high =
            high.apply(RelationshipEvent::LearnedFact).familiarity() - high.familiarity();
        assert!(gain_high < gain_low);
    }

    #[test]
    fn familiarity_converges_below_one() {
        let events = std::iter::repeat_n(RelationshipEvent::LearnedFact, 200);
        let s = RelationshipState::INITIAL.apply_all(events);
        assert!(s.familiarity() <= 1.0);
        assert!(!s.should_prioritize_onboarding());
    }

    #[test]
    fn positive_interaction_raises_trust_and_brightens_mood() {
        let s = RelationshipState::INITIAL
            .apply(RelationshipEvent::PositiveInteraction { strength: 1.0 });
        assert!(s.trust() > 0.0);
        assert!(s.mood().valence() > 0.0);
    }

    #[test]
    fn negative_interaction_lowers_trust_and_darkens_mood() {
        let trusting = RelationshipState::new(0.8, 0.5, MoodVector::NEUTRAL);
        let s = trusting.apply(RelationshipEvent::NegativeInteraction { strength: 1.0 });
        assert!(s.trust() < 0.8);
        assert!(s.mood().valence() < 0.0);
        // Familiarity is unaffected by interaction valence.
        assert_eq!(s.familiarity(), 0.5);
    }

    #[test]
    fn positive_interaction_uses_exact_formula() {
        // strength 0.5 from trust 0.5, neutral mood.
        let s = RelationshipState::new(0.5, 0.0, MoodVector::NEUTRAL)
            .apply(RelationshipEvent::PositiveInteraction { strength: 0.5 });
        // trust: 0.5 + (1 - 0.5) * 0.1 * 0.5 = 0.525
        assert!(approx(s.trust(), 0.525));
        // mood: neutral blended toward (1, 0.5) by 0.3 * 0.5 = 0.15
        assert!(approx(s.mood().valence(), 0.15)); // 0 + (1 - 0) * 0.15
        assert!(approx(s.mood().arousal(), 0.075)); // 0 + (0.5 - 0) * 0.15
    }

    #[test]
    fn negative_interaction_uses_exact_formula() {
        // strength 0.5 from trust 0.8, neutral mood.
        let s = RelationshipState::new(0.8, 0.5, MoodVector::NEUTRAL)
            .apply(RelationshipEvent::NegativeInteraction { strength: 0.5 });
        // trust: 0.8 * (1 - 0.15 * 0.5) = 0.8 * 0.925 = 0.74
        assert!(approx(s.trust(), 0.74));
        // mood: neutral blended toward (-1, 0.3) by 0.3 * 0.5 = 0.15
        assert!(approx(s.mood().valence(), -0.15)); // 0 + (-1 - 0) * 0.15
        assert!(approx(s.mood().arousal(), 0.045)); // 0 + (0.3 - 0) * 0.15
    }

    #[test]
    fn interaction_strength_is_clamped() {
        let huge = RelationshipState::INITIAL
            .apply(RelationshipEvent::PositiveInteraction { strength: 99.0 });
        let one = RelationshipState::INITIAL
            .apply(RelationshipEvent::PositiveInteraction { strength: 1.0 });
        assert_eq!(huge, one);
        let neg = RelationshipState::INITIAL
            .apply(RelationshipEvent::PositiveInteraction { strength: -5.0 });
        assert_eq!(neg.trust(), 0.0);
    }

    #[test]
    fn forgetting_lowers_familiarity() {
        let known = RelationshipState::new(0.5, 0.8, MoodVector::NEUTRAL);
        let after = known.apply(RelationshipEvent::MemoryForgotten);
        assert!(after.familiarity() < 0.8);
        assert_eq!(after.trust(), 0.5);
    }

    #[test]
    fn mood_observed_blends_toward_target() {
        let happy = MoodVector::new(1.0, 1.0);
        let s = RelationshipState::INITIAL.apply(RelationshipEvent::MoodObserved {
            target: happy,
            fraction: 0.5,
        });
        assert!(approx(s.mood().valence(), 0.5));
        assert!(approx(s.mood().arousal(), 0.5));
    }

    #[test]
    fn apply_all_equals_ordered_fold() {
        let events = [
            RelationshipEvent::LearnedFact,
            RelationshipEvent::PositiveInteraction { strength: 0.5 },
            RelationshipEvent::LearnedFact,
            RelationshipEvent::NegativeInteraction { strength: 0.2 },
        ];
        let folded = events
            .iter()
            .fold(RelationshipState::INITIAL, |s, e| s.apply(*e));
        assert_eq!(RelationshipState::INITIAL.apply_all(events), folded);
    }

    #[test]
    fn round_trips_through_serde() {
        let original = RelationshipState::new(0.3, 0.6, MoodVector::new(0.2, -0.4));
        let json = serde_json::to_string(&original).unwrap();
        let decoded: RelationshipState = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, original);
    }
}
