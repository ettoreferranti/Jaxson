//! Per-topic affinity: how much the user likes or dislikes each subject, in `[-1, 1]`.
//!
//! This is the "richer transitions" half of the relationship state machine (F1.5). Where
//! [`RelationshipState`](crate::RelationshipState) tracks scalars about the relationship
//! as a whole (trust, familiarity, mood), [`TopicAffinities`] tracks a *per-subject*
//! feeling: hiking `+0.8`, mornings `-0.6`. Affinity drives what Jaxson brings up and
//! colors the baseline mood (see `docs/ARCHITECTURE.md` §4.3).
//!
//! Reinforcement is a clamped, diminishing-returns transition (the same shape as trust /
//! familiarity): each observation nudges the score toward `±1` by a fraction of the
//! remaining headroom, so repeated signals saturate rather than overflow. It's a heap
//! map, so unlike `RelationshipState` it isn't `Copy` and has no `const` initial value.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// How fast a single observation moves an affinity toward its bound.
const LEARNING_RATE: f64 = 0.3;

/// Per-topic affinity scores in `[-1, 1]`, keyed by a normalized topic string.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TopicAffinities {
    scores: BTreeMap<String, f64>,
}

impl TopicAffinities {
    /// An empty set of affinities (nothing felt about anything yet).
    pub fn new() -> Self {
        TopicAffinities::default()
    }

    /// Current affinity for `topic` in `[-1, 1]` — `0.0` if Jaxson has no feeling about
    /// it yet. Case- and whitespace-insensitive.
    pub fn get(&self, topic: &str) -> f64 {
        self.scores.get(&normalize(topic)).copied().unwrap_or(0.0)
    }

    /// Reinforce `topic` by `strength` (clamped to `[-1, 1]`; positive = liked). The score
    /// moves toward `±1` by a fraction of the remaining headroom, so it approaches but
    /// never crosses the bound. A blank topic is ignored.
    pub fn reinforce(&mut self, topic: &str, strength: f64) {
        let key = normalize(topic);
        if key.is_empty() {
            return;
        }
        let s = strength.clamp(-1.0, 1.0);
        let current = self.scores.get(&key).copied().unwrap_or(0.0);
        // Headroom is the distance to the bound we're moving toward (+1 for s>=0, -1 for
        // s<0), so the step shrinks as the score saturates.
        let headroom = if s >= 0.0 {
            1.0 - current
        } else {
            1.0 + current
        };
        let next = (current + headroom * LEARNING_RATE * s).clamp(-1.0, 1.0);
        self.scores.insert(key, next);
    }

    /// Decay every affinity toward `0` by `rate` (clamped to `[0, 1]`): `0` keeps them as
    /// is, `1` wipes them to zero. Models feelings fading when a topic goes quiet.
    pub fn decay(&mut self, rate: f64) {
        let keep = 1.0 - rate.clamp(0.0, 1.0);
        for score in self.scores.values_mut() {
            *score *= keep;
        }
    }

    /// The most-liked topic whose affinity is at least `threshold`, if any. Ties are
    /// broken by topic name (smallest wins) for determinism.
    pub fn favorite(&self, threshold: f64) -> Option<(&str, f64)> {
        self.scores
            .iter()
            .filter(|(_, &v)| v >= threshold)
            .max_by(|(ak, av), (bk, bv)| {
                av.partial_cmp(bv)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| bk.cmp(ak))
            })
            .map(|(k, &v)| (k.as_str(), v))
    }

    /// How many topics have a recorded affinity.
    pub fn len(&self) -> usize {
        self.scores.len()
    }

    /// Whether nothing is felt about anything yet.
    pub fn is_empty(&self) -> bool {
        self.scores.is_empty()
    }

    /// Iterate over `(topic, affinity)` pairs, ordered by topic name.
    pub fn iter(&self) -> impl Iterator<Item = (&str, f64)> {
        self.scores.iter().map(|(k, &v)| (k.as_str(), v))
    }
}

/// Normalize a topic key: trimmed and lowercased, so "Hiking", " hiking " collapse.
fn normalize(topic: &str) -> String {
    topic.trim().to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    #[test]
    fn unknown_topic_is_neutral() {
        assert_eq!(TopicAffinities::new().get("hiking"), 0.0);
        assert!(TopicAffinities::new().is_empty());
    }

    #[test]
    fn reinforce_uses_the_exact_diminishing_formula() {
        let mut a = TopicAffinities::new();
        a.reinforce("hiking", 1.0);
        // 0 + (1 - 0) * 0.3 * 1 = 0.3
        assert!(approx(a.get("hiking"), 0.3));
        a.reinforce("hiking", 1.0);
        // 0.3 + (1 - 0.3) * 0.3 * 1 = 0.51
        assert!(approx(a.get("hiking"), 0.51));
    }

    #[test]
    fn negative_reinforcement_is_symmetric() {
        let mut a = TopicAffinities::new();
        a.reinforce("mornings", -1.0);
        // 0 + (1 + 0) * 0.3 * -1 = -0.3
        assert!(approx(a.get("mornings"), -0.3));
        a.reinforce("mornings", -1.0);
        // -0.3 + (1 + -0.3) * 0.3 * -1 = -0.51
        assert!(approx(a.get("mornings"), -0.51));
    }

    #[test]
    fn affinity_saturates_but_never_crosses_the_bound() {
        let mut a = TopicAffinities::new();
        for _ in 0..200 {
            a.reinforce("dogs", 1.0);
        }
        let v = a.get("dogs");
        assert!(v < 1.0, "must not reach or exceed +1, got {v}");
        assert!(v > 0.99, "should be close to +1, got {v}");
    }

    #[test]
    fn fractional_strength_scales_the_step() {
        let mut a = TopicAffinities::new();
        a.reinforce("x", 0.5);
        // 0 + (1 - 0) * 0.3 * 0.5 = 0.15 — pins that `strength` multiplies (not divides).
        assert!(approx(a.get("x"), 0.15));
    }

    #[test]
    fn reinforcement_strength_is_clamped() {
        let mut huge = TopicAffinities::new();
        huge.reinforce("x", 9.0);
        let mut one = TopicAffinities::new();
        one.reinforce("x", 1.0);
        assert_eq!(huge.get("x"), one.get("x"));
    }

    #[test]
    fn len_counts_distinct_topics() {
        let mut a = TopicAffinities::new();
        assert_eq!(a.len(), 0);
        a.reinforce("hiking", 1.0);
        a.reinforce("coffee", 1.0);
        assert_eq!(a.len(), 2);
        assert!(!a.is_empty());
    }

    #[test]
    fn topic_keys_are_case_and_whitespace_insensitive() {
        let mut a = TopicAffinities::new();
        a.reinforce("Hiking", 1.0);
        a.reinforce("  hiking ", 1.0);
        // Both updates landed on the same key.
        assert_eq!(a.len(), 1);
        assert!(approx(a.get("HIKING"), 0.51));
    }

    #[test]
    fn blank_topics_are_ignored() {
        let mut a = TopicAffinities::new();
        a.reinforce("   ", 1.0);
        a.reinforce("", 1.0);
        assert!(a.is_empty());
    }

    #[test]
    fn flipping_sign_moves_back_toward_zero_and_across() {
        let mut a = TopicAffinities::new();
        a.reinforce("topic", 1.0); // 0.3
        a.reinforce("topic", -1.0); // 0.3 + (1 + 0.3)*0.3*-1 = 0.3 - 0.39 = -0.09
        assert!(approx(a.get("topic"), -0.09));
    }

    #[test]
    fn decay_pulls_scores_toward_zero() {
        let mut a = TopicAffinities::new();
        a.reinforce("a", 1.0); // 0.3
        a.decay(0.5);
        assert!(approx(a.get("a"), 0.15));
        a.decay(1.0);
        assert!(approx(a.get("a"), 0.0));
    }

    #[test]
    fn decay_zero_and_clamping_leave_scores_intact() {
        let mut a = TopicAffinities::new();
        a.reinforce("a", 1.0); // 0.3
        a.decay(0.0);
        assert!(approx(a.get("a"), 0.3));
        // Out-of-range rate is clamped to [0, 1]; negative behaves like 0.
        a.decay(-5.0);
        assert!(approx(a.get("a"), 0.3));
    }

    #[test]
    fn favorite_picks_the_highest_above_threshold() {
        let mut a = TopicAffinities::new();
        a.reinforce("hiking", 1.0); // 0.3
        a.reinforce("hiking", 1.0); // 0.51
        a.reinforce("coffee", 1.0); // 0.3
        a.reinforce("mornings", -1.0); // -0.3
        assert_eq!(a.favorite(0.5), Some(("hiking", 0.51)));
        // Below any score → nothing qualifies.
        assert_eq!(a.favorite(0.9), None);
    }

    #[test]
    fn favorite_threshold_is_inclusive() {
        let mut a = TopicAffinities::new();
        a.reinforce("x", 1.0); // exactly 0.3
        assert_eq!(a.favorite(0.3), Some(("x", 0.3)));
        assert_eq!(a.favorite(0.30000001), None);
    }

    #[test]
    fn favorite_breaks_ties_by_smallest_topic_name() {
        let mut a = TopicAffinities::new();
        a.reinforce("zebra", 1.0); // 0.3
        a.reinforce("apple", 1.0); // 0.3
        assert_eq!(a.favorite(0.0), Some(("apple", 0.3)));
    }

    #[test]
    fn favorite_ignores_negative_topics() {
        let mut a = TopicAffinities::new();
        a.reinforce("mornings", -1.0);
        assert_eq!(a.favorite(0.0), None);
    }

    #[test]
    fn iter_is_ordered_by_topic() {
        let mut a = TopicAffinities::new();
        a.reinforce("beta", 1.0);
        a.reinforce("alpha", 1.0);
        let topics: Vec<&str> = a.iter().map(|(k, _)| k).collect();
        assert_eq!(topics, vec!["alpha", "beta"]);
    }

    #[test]
    fn round_trips_through_serde() {
        let mut a = TopicAffinities::new();
        a.reinforce("hiking", 1.0);
        a.reinforce("mornings", -0.5);
        let json = serde_json::to_string(&a).unwrap();
        let decoded: TopicAffinities = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, a);
    }
}
