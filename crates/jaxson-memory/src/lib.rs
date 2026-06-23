//! # jaxson-memory
//!
//! Jaxson's memory as a knowledge graph: [`MemoryNode`]s (facts, people, events,
//! preferences, episodes) connected by typed, weighted [`Edge`]s. The pure
//! [`MemoryGraph`] is the authoritative model; a [`MemoryStore`] makes it durable.
//!
//! The graph and the in-memory store are pure and fully mutation-tested. Encrypted,
//! on-disk persistence ([`SqliteStore`]) lives behind the `sqlite` cargo feature
//! (SQLCipher; encryption at rest per the privacy model).
//!
//! ```
//! use jaxson_memory::{Edge, MemoryGraph, MemoryId, MemoryKind, MemoryNode, Provenance, Relation};
//!
//! let mut graph = MemoryGraph::new();
//! let ettore = MemoryId::from_u128(1);
//! let dogs = MemoryId::from_u128(2);
//! graph.insert_node(MemoryNode::new(ettore, MemoryKind::Person, "Ettore", 0, Provenance::StatedByUser, 0.9));
//! graph.insert_node(MemoryNode::new(dogs, MemoryKind::Preference, "dogs", 0, Provenance::InferredFromConversation, 0.7));
//! graph.insert_edge(Edge::new(ettore, dogs, Relation::Likes, 0.8)).unwrap();
//! assert_eq!(graph.neighbors(ettore).len(), 1);
//! ```

mod edge;
mod graph;
mod node;
mod store;

pub use edge::{Edge, Relation};
pub use graph::{GraphError, MemoryGraph};
pub use node::{MemoryId, MemoryKind, MemoryNode, Provenance};
pub use store::{InMemoryStore, MemoryStore, StoreError};

#[cfg(feature = "sqlite")]
mod sqlite;
#[cfg(feature = "sqlite")]
pub use sqlite::SqliteStore;
