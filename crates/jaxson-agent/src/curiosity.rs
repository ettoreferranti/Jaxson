//! Proactive curiosity: deciding when Jaxson should ask a getting-to-know-you question
//! and what to ask about, gated by how well it already knows the user (F1.11 / FR-M6).
//!
//! Two pure inputs — the [`RelationshipState`] (familiarity) and the [`MemoryGraph`]
//! (what's already known) — produce an optional system-prompt fragment. The behavior is
//! graduated:
//!
//! - **Onboarding** (`familiarity` below the onboarding threshold): lead every turn with
//!   a warm question, aimed at the first gap in the onboarding curriculum.
//! - **Acquainted** (above the threshold, but gaps remain): drop the lead; only gently
//!   suggest a question about a still-missing topic.
//! - **Familiar** (above the threshold, every curriculum topic covered): say nothing and
//!   just converse.
//!
//! Targeting questions at *gaps* keeps Jaxson from re-asking what it already knows: once
//! a topic is answered and extracted into the graph, the curriculum advances to the next.

use jaxson_core::RelationshipState;
use jaxson_memory::{MemoryGraph, MemoryKind};

/// A getting-to-know-you focus: a memory category Jaxson wants to fill in, paired with
/// the phrase that steers the model toward asking about it.
struct Topic {
    kind: MemoryKind,
    about: &'static str,
}

/// What Jaxson tries to learn, in order. Earlier topics come up before later ones, so a
/// brand-new Jaxson leads with the basics (who you are) before the finer details.
/// [`MemoryKind::Episode`] (a remembered conversational moment) is intentionally absent —
/// it's a byproduct of talking, not something to ask about.
const CURRICULUM: &[Topic] = &[
    Topic {
        kind: MemoryKind::Person,
        about: "who they are and who matters most to them",
    },
    Topic {
        kind: MemoryKind::Preference,
        about: "the stuff they love and the stuff they can't stand",
    },
    Topic {
        kind: MemoryKind::Event,
        about: "what they've been up to lately",
    },
    Topic {
        kind: MemoryKind::Fact,
        about: "some fun little detail about their everyday life",
    },
];

/// Lead-in used while Jaxson barely knows the user (onboarding tier).
const LEAD: &str = "You barely know this human yet, and you're itching to find out more!";
/// Lead-in used once acquainted but still curious (gaps remain).
const CASUAL: &str = "You still want to get to know your human better.";

/// A system-prompt fragment nudging Jaxson to ask a question this turn, or `None` to just
/// converse. See the module docs for the familiarity tiers. The phrasing stays playful so
/// Jaxson sounds like an excited robot pal, not an interviewer.
pub(crate) fn proactive_hint(state: &RelationshipState, graph: &MemoryGraph) -> Option<String> {
    let gap = first_gap(graph);
    if state.should_prioritize_onboarding() {
        Some(match gap {
            Some(topic) => format!("{LEAD} Excitedly ask them about {}.", topic.about),
            None => format!("{LEAD} Ask them something fun to get to know them better."),
        })
    } else {
        gap.map(|topic| {
            format!(
                "{CASUAL} When it fits the chat, playfully ask about {}.",
                topic.about
            )
        })
    }
}

/// The first curriculum topic the graph has no memory for, if any.
fn first_gap(graph: &MemoryGraph) -> Option<&'static Topic> {
    CURRICULUM
        .iter()
        .find(|topic| !knows_kind(graph, topic.kind))
}

/// Whether the graph holds at least one memory of `kind`.
fn knows_kind(graph: &MemoryGraph, kind: MemoryKind) -> bool {
    graph.nodes().any(|node| node.kind == kind)
}

#[cfg(test)]
mod tests {
    use super::*;
    use jaxson_core::MoodVector;
    use jaxson_memory::{MemoryId, MemoryNode, Provenance};

    /// A graph holding one memory of each given kind.
    fn graph_with(kinds: &[MemoryKind]) -> MemoryGraph {
        let mut g = MemoryGraph::new();
        for (i, kind) in kinds.iter().enumerate() {
            g.insert_node(MemoryNode::new(
                MemoryId::from_u128(i as u128 + 1),
                *kind,
                "x",
                0,
                Provenance::StatedByUser,
                0.9,
            ));
        }
        g
    }

    /// Familiarity 0 — squarely in the onboarding tier.
    fn onboarding() -> RelationshipState {
        RelationshipState::INITIAL
    }

    /// Familiarity above the onboarding threshold — acquainted tier.
    fn acquainted() -> RelationshipState {
        RelationshipState::new(0.0, 0.5, MoodVector::NEUTRAL)
    }

    const ALL_TOPICS: [MemoryKind; 4] = [
        MemoryKind::Person,
        MemoryKind::Preference,
        MemoryKind::Event,
        MemoryKind::Fact,
    ];

    #[test]
    fn fresh_agent_leads_with_a_person_question() {
        let hint = proactive_hint(&onboarding(), &graph_with(&[])).unwrap();
        assert!(hint.starts_with(LEAD));
        assert!(hint.contains("matters most")); // Person — first in the curriculum
    }

    #[test]
    fn onboarding_advances_to_the_next_gap_once_a_topic_is_covered() {
        // Person is known, so the first gap is now Preference.
        let hint = proactive_hint(&onboarding(), &graph_with(&[MemoryKind::Person])).unwrap();
        assert!(hint.contains("can't stand"));
        assert!(!hint.contains("matters most"));
    }

    #[test]
    fn curriculum_is_ordered_person_preference_event_fact() {
        let expect = ["matters most", "can't stand", "been up to", "everyday life"];
        // Covering the first `i` topics reveals topic `i` as the next gap.
        for i in 0..ALL_TOPICS.len() {
            let hint = proactive_hint(&onboarding(), &graph_with(&ALL_TOPICS[..i])).unwrap();
            assert!(hint.contains(expect[i]), "gap {i}: {hint}");
        }
    }

    #[test]
    fn onboarding_with_all_topics_covered_still_asks_generically() {
        let hint = proactive_hint(&onboarding(), &graph_with(&ALL_TOPICS)).unwrap();
        assert!(hint.starts_with(LEAD));
        assert!(hint.contains("something fun"));
    }

    #[test]
    fn acquainted_drops_the_lead_but_still_nudges_a_remaining_gap() {
        let hint = proactive_hint(&acquainted(), &graph_with(&[])).unwrap();
        assert!(hint.starts_with(CASUAL));
        assert!(hint.contains("playfully ask"));
        assert!(hint.contains("matters most"));
    }

    #[test]
    fn familiar_with_no_gaps_stays_quiet() {
        assert!(proactive_hint(&acquainted(), &graph_with(&ALL_TOPICS)).is_none());
    }

    #[test]
    fn episode_memories_do_not_satisfy_any_curriculum_topic() {
        // Only an Episode is known → Person is still the first gap (we never ask about
        // episodes), so onboarding still leads with the Person question.
        let hint = proactive_hint(&onboarding(), &graph_with(&[MemoryKind::Episode])).unwrap();
        assert!(hint.contains("matters most"));
    }
}
