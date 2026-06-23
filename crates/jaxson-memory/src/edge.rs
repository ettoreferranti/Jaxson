use serde::{Deserialize, Serialize};

use crate::node::MemoryId;

/// A typed relationship between two memory nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Relation {
    Likes,
    Dislikes,
    Knows,
    RelatedTo,
    HappenedOn,
    Causes,
}

impl Relation {
    /// Stable string used for persistence.
    pub fn as_str(self) -> &'static str {
        match self {
            Relation::Likes => "likes",
            Relation::Dislikes => "dislikes",
            Relation::Knows => "knows",
            Relation::RelatedTo => "related_to",
            Relation::HappenedOn => "happened_on",
            Relation::Causes => "causes",
        }
    }

    /// Parse from the persisted string.
    pub fn from_db_str(s: &str) -> Option<Self> {
        match s {
            "likes" => Some(Relation::Likes),
            "dislikes" => Some(Relation::Dislikes),
            "knows" => Some(Relation::Knows),
            "related_to" => Some(Relation::RelatedTo),
            "happened_on" => Some(Relation::HappenedOn),
            "causes" => Some(Relation::Causes),
            _ => None,
        }
    }
}

/// A directed, weighted edge `from -> to`. Weight is in `[0.0, 1.0]` and represents
/// the strength of the association; it strengthens with reinforcement and decays
/// over time.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Edge {
    pub from: MemoryId,
    pub to: MemoryId,
    pub relation: Relation,
    pub weight: f32,
}

impl Edge {
    /// Create an edge, clamping `weight` into `[0.0, 1.0]`.
    pub fn new(from: MemoryId, to: MemoryId, relation: Relation, weight: f32) -> Self {
        Edge {
            from,
            to,
            relation,
            weight: weight.clamp(0.0, 1.0),
        }
    }

    /// Reinforce the association: move the weight a fraction `amount` of the way
    /// toward `1.0` (diminishing returns). `amount` is clamped to `[0.0, 1.0]`.
    pub fn strengthened(self, amount: f32) -> Self {
        let a = amount.clamp(0.0, 1.0);
        Edge {
            weight: (self.weight + (1.0 - self.weight) * a).clamp(0.0, 1.0),
            ..self
        }
    }

    /// Decay the association by `factor` (e.g. `0.1` removes 10% of the weight).
    /// `factor` is clamped to `[0.0, 1.0]`.
    pub fn decayed(self, factor: f32) -> Self {
        let f = factor.clamp(0.0, 1.0);
        Edge {
            weight: (self.weight * (1.0 - f)).clamp(0.0, 1.0),
            ..self
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(n: u128) -> MemoryId {
        MemoryId::from_u128(n)
    }

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-6
    }

    #[test]
    fn weight_is_clamped_on_construction() {
        assert_eq!(Edge::new(id(1), id(2), Relation::Likes, 5.0).weight, 1.0);
        assert_eq!(Edge::new(id(1), id(2), Relation::Likes, -5.0).weight, 0.0);
    }

    #[test]
    fn strengthen_moves_toward_one_with_diminishing_returns() {
        let e = Edge::new(id(1), id(2), Relation::Knows, 0.4);
        // 0.4 + (1 - 0.4) * 0.5 = 0.7
        assert!(approx(e.strengthened(0.5).weight, 0.7));
        // Gain shrinks as weight grows.
        let high = Edge::new(id(1), id(2), Relation::Knows, 0.9);
        let gain_low = e.strengthened(0.5).weight - e.weight;
        let gain_high = high.strengthened(0.5).weight - high.weight;
        assert!(gain_high < gain_low);
    }

    #[test]
    fn strengthen_amount_is_clamped() {
        let e = Edge::new(id(1), id(2), Relation::Knows, 0.2);
        assert_eq!(e.strengthened(9.0).weight, 1.0);
        assert_eq!(e.strengthened(-9.0).weight, 0.2);
    }

    #[test]
    fn decay_reduces_weight_by_factor() {
        let e = Edge::new(id(1), id(2), Relation::Knows, 0.8);
        // 0.8 * (1 - 0.25) = 0.6
        assert!(approx(e.decayed(0.25).weight, 0.6));
    }

    #[test]
    fn decay_factor_is_clamped() {
        let e = Edge::new(id(1), id(2), Relation::Knows, 0.8);
        assert_eq!(e.decayed(9.0).weight, 0.0);
        assert!(approx(e.decayed(-9.0).weight, 0.8));
    }

    #[test]
    fn strengthen_and_decay_preserve_endpoints_and_relation() {
        let e = Edge::new(id(1), id(2), Relation::Likes, 0.5);
        let s = e.strengthened(0.3);
        assert_eq!((s.from, s.to, s.relation), (id(1), id(2), Relation::Likes));
    }

    #[test]
    fn relation_db_string_round_trips() {
        for r in [
            Relation::Likes,
            Relation::Dislikes,
            Relation::Knows,
            Relation::RelatedTo,
            Relation::HappenedOn,
            Relation::Causes,
        ] {
            assert_eq!(Relation::from_db_str(r.as_str()), Some(r));
        }
        assert_eq!(Relation::from_db_str("???"), None);
    }
}
