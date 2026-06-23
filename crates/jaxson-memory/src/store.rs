use thiserror::Error;

use crate::graph::MemoryGraph;

/// Errors from a [`MemoryStore`].
#[derive(Debug, Error)]
pub enum StoreError {
    /// The underlying storage engine failed (I/O, SQL, wrong key, …).
    #[error("storage backend error: {0}")]
    Backend(String),
    /// Persisted data could not be reconstructed into a valid graph.
    #[error("corrupt store: {0}")]
    Corrupt(String),
}

/// Persistence for the memory graph.
///
/// Snapshot semantics: [`save`](Self::save) persists the whole graph and
/// [`load`](Self::load) reconstructs it. The [`MemoryGraph`] remains the authoritative,
/// validated in-memory model; the store just makes it durable. This keeps the
/// contract identical across the in-memory and encrypted-SQLite backends.
pub trait MemoryStore {
    /// Persist the entire graph, replacing any previously stored state.
    fn save(&mut self, graph: &MemoryGraph) -> Result<(), StoreError>;

    /// Load the persisted graph (an empty graph if nothing was stored).
    fn load(&self) -> Result<MemoryGraph, StoreError>;
}

/// A non-persistent [`MemoryStore`] backed by an in-process graph. Useful for tests,
/// demos, and running Jaxson without touching disk.
#[derive(Debug, Default, Clone)]
pub struct InMemoryStore {
    graph: MemoryGraph,
}

impl InMemoryStore {
    pub fn new() -> Self {
        InMemoryStore::default()
    }
}

impl MemoryStore for InMemoryStore {
    fn save(&mut self, graph: &MemoryGraph) -> Result<(), StoreError> {
        self.graph = graph.clone();
        Ok(())
    }

    fn load(&self) -> Result<MemoryGraph, StoreError> {
        Ok(self.graph.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edge::{Edge, Relation};
    use crate::node::{MemoryId, MemoryKind, MemoryNode, Provenance};

    fn sample_graph() -> MemoryGraph {
        let mut g = MemoryGraph::new();
        g.insert_node(MemoryNode::new(
            MemoryId::from_u128(1),
            MemoryKind::Person,
            "Ettore",
            100,
            Provenance::StatedByUser,
            0.9,
        ));
        g.insert_node(MemoryNode::new(
            MemoryId::from_u128(2),
            MemoryKind::Preference,
            "likes dogs",
            101,
            Provenance::InferredFromConversation,
            0.6,
        ));
        g.insert_edge(Edge::new(
            MemoryId::from_u128(1),
            MemoryId::from_u128(2),
            Relation::Likes,
            0.7,
        ))
        .unwrap();
        g
    }

    #[test]
    fn load_on_fresh_store_is_empty() {
        let store = InMemoryStore::new();
        assert!(store.load().unwrap().is_empty());
    }

    #[test]
    fn save_then_load_round_trips_the_graph() {
        let mut store = InMemoryStore::new();
        let graph = sample_graph();
        store.save(&graph).unwrap();
        assert_eq!(store.load().unwrap(), graph);
    }

    #[test]
    fn save_replaces_previous_state() {
        let mut store = InMemoryStore::new();
        store.save(&sample_graph()).unwrap();
        store.save(&MemoryGraph::new()).unwrap();
        assert!(store.load().unwrap().is_empty());
    }
}
