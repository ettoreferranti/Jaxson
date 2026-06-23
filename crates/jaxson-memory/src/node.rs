use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Stable identifier for a memory node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct MemoryId(pub Uuid);

impl MemoryId {
    /// A fresh random id (production).
    pub fn new() -> Self {
        MemoryId(Uuid::new_v4())
    }

    /// A deterministic id from an integer — handy for tests and fixtures.
    pub fn from_u128(n: u128) -> Self {
        MemoryId(Uuid::from_u128(n))
    }
}

impl Default for MemoryId {
    fn default() -> Self {
        MemoryId::new()
    }
}

impl std::fmt::Display for MemoryId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// What kind of thing a memory is about.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryKind {
    /// A standalone fact ("the user is allergic to peanuts").
    Fact,
    /// A person in the user's life.
    Person,
    /// Something that happened.
    Event,
    /// A like/dislike or taste.
    Preference,
    /// A remembered conversational moment.
    Episode,
}

/// How Jaxson came to know a memory — used for transparency and trust weighting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Provenance {
    /// The user said it directly.
    StatedByUser,
    /// Jaxson inferred it from conversation.
    InferredFromConversation,
    /// Imported from an external source.
    Imported,
}

/// A node in the memory graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryNode {
    pub id: MemoryId,
    pub kind: MemoryKind,
    pub content: String,
    /// Creation time as a Unix timestamp (seconds). Supplied by the caller so the
    /// graph stays deterministic and testable.
    pub created_at: i64,
    pub provenance: Provenance,
    /// Confidence in `[0.0, 1.0]`.
    pub confidence: f32,
    /// Optional embedding vector for similarity retrieval (added in F1.4).
    pub embedding: Option<Vec<f32>>,
}

impl MemoryNode {
    /// Create a node, clamping `confidence` into `[0.0, 1.0]`.
    pub fn new(
        id: MemoryId,
        kind: MemoryKind,
        content: impl Into<String>,
        created_at: i64,
        provenance: Provenance,
        confidence: f32,
    ) -> Self {
        MemoryNode {
            id,
            kind,
            content: content.into(),
            created_at,
            provenance,
            confidence: confidence.clamp(0.0, 1.0),
            embedding: None,
        }
    }

    /// Attach an embedding vector (builder style).
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }
}

impl MemoryKind {
    /// Stable string used for persistence.
    pub fn as_str(self) -> &'static str {
        match self {
            MemoryKind::Fact => "fact",
            MemoryKind::Person => "person",
            MemoryKind::Event => "event",
            MemoryKind::Preference => "preference",
            MemoryKind::Episode => "episode",
        }
    }

    /// Parse from the persisted string.
    pub fn from_db_str(s: &str) -> Option<Self> {
        match s {
            "fact" => Some(MemoryKind::Fact),
            "person" => Some(MemoryKind::Person),
            "event" => Some(MemoryKind::Event),
            "preference" => Some(MemoryKind::Preference),
            "episode" => Some(MemoryKind::Episode),
            _ => None,
        }
    }
}

impl Provenance {
    /// Stable string used for persistence.
    pub fn as_str(self) -> &'static str {
        match self {
            Provenance::StatedByUser => "stated_by_user",
            Provenance::InferredFromConversation => "inferred_from_conversation",
            Provenance::Imported => "imported",
        }
    }

    /// Parse from the persisted string.
    pub fn from_db_str(s: &str) -> Option<Self> {
        match s {
            "stated_by_user" => Some(Provenance::StatedByUser),
            "inferred_from_conversation" => Some(Provenance::InferredFromConversation),
            "imported" => Some(Provenance::Imported),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confidence_is_clamped_on_construction() {
        let high = MemoryNode::new(
            MemoryId::from_u128(1),
            MemoryKind::Fact,
            "x",
            0,
            Provenance::StatedByUser,
            9.0,
        );
        assert_eq!(high.confidence, 1.0);
        let low = MemoryNode::new(
            MemoryId::from_u128(2),
            MemoryKind::Fact,
            "x",
            0,
            Provenance::StatedByUser,
            -2.0,
        );
        assert_eq!(low.confidence, 0.0);
    }

    #[test]
    fn with_embedding_attaches_vector() {
        let node = MemoryNode::new(
            MemoryId::from_u128(1),
            MemoryKind::Fact,
            "x",
            0,
            Provenance::StatedByUser,
            0.5,
        )
        .with_embedding(vec![0.1, 0.2]);
        assert_eq!(node.embedding, Some(vec![0.1, 0.2]));
    }

    #[test]
    fn memory_id_displays_as_its_uuid() {
        assert_eq!(
            MemoryId::from_u128(0).to_string(),
            "00000000-0000-0000-0000-000000000000"
        );
    }

    #[test]
    fn from_u128_is_deterministic() {
        assert_eq!(MemoryId::from_u128(7), MemoryId::from_u128(7));
        assert_ne!(MemoryId::from_u128(7), MemoryId::from_u128(8));
    }

    #[test]
    fn memory_kind_db_string_round_trips() {
        for kind in [
            MemoryKind::Fact,
            MemoryKind::Person,
            MemoryKind::Event,
            MemoryKind::Preference,
            MemoryKind::Episode,
        ] {
            assert_eq!(MemoryKind::from_db_str(kind.as_str()), Some(kind));
        }
        assert_eq!(MemoryKind::from_db_str("nonsense"), None);
    }

    #[test]
    fn provenance_db_string_round_trips() {
        for p in [
            Provenance::StatedByUser,
            Provenance::InferredFromConversation,
            Provenance::Imported,
        ] {
            assert_eq!(Provenance::from_db_str(p.as_str()), Some(p));
        }
        assert_eq!(Provenance::from_db_str(""), None);
    }
}
