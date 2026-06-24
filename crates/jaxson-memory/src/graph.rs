use std::collections::HashMap;

use thiserror::Error;

use crate::edge::{Edge, Relation};
use crate::node::{MemoryId, MemoryNode};

/// Errors from graph operations.
#[derive(Debug, Error, PartialEq)]
pub enum GraphError {
    /// An edge referenced a node that isn't in the graph.
    #[error("edge references unknown node {0}")]
    MissingNode(MemoryId),
}

/// An in-memory knowledge graph of memories and their typed, weighted relations.
///
/// This is the deterministic core of `jaxson-memory`: every operation is a pure
/// function of the current state, so it is fast to unit- and mutation-test.
/// Persistence is layered on top via [`MemoryStore`](crate::MemoryStore).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MemoryGraph {
    nodes: HashMap<MemoryId, MemoryNode>,
    edges: Vec<Edge>,
}

impl MemoryGraph {
    pub fn new() -> Self {
        MemoryGraph::default()
    }

    /// Insert (or replace) a node. Returns the previous node with this id, if any.
    pub fn insert_node(&mut self, node: MemoryNode) -> Option<MemoryNode> {
        self.nodes.insert(node.id, node)
    }

    /// Look up a node by id.
    pub fn node(&self, id: MemoryId) -> Option<&MemoryNode> {
        self.nodes.get(&id)
    }

    /// Whether a node with this id exists.
    pub fn contains_node(&self, id: MemoryId) -> bool {
        self.nodes.contains_key(&id)
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Insert an edge. Both endpoints must already exist, otherwise
    /// [`GraphError::MissingNode`] is returned and the graph is unchanged.
    pub fn insert_edge(&mut self, edge: Edge) -> Result<(), GraphError> {
        if !self.contains_node(edge.from) {
            return Err(GraphError::MissingNode(edge.from));
        }
        if !self.contains_node(edge.to) {
            return Err(GraphError::MissingNode(edge.to));
        }
        self.edges.push(edge);
        Ok(())
    }

    /// All edges originating at `id`.
    pub fn edges_from(&self, id: MemoryId) -> impl Iterator<Item = &Edge> {
        self.edges.iter().filter(move |e| e.from == id)
    }

    /// The nodes directly reachable from `id` by following outgoing edges.
    pub fn neighbors(&self, id: MemoryId) -> Vec<&MemoryNode> {
        self.edges_from(id)
            .filter_map(|e| self.nodes.get(&e.to))
            .collect()
    }

    /// Remove a node and every edge incident to it (deletion propagates, per FR-M4).
    /// Returns the removed node, if it existed.
    pub fn remove_node(&mut self, id: MemoryId) -> Option<MemoryNode> {
        let removed = self.nodes.remove(&id);
        if removed.is_some() {
            self.edges.retain(|e| e.from != id && e.to != id);
        }
        removed
    }

    /// Iterate over all nodes.
    pub fn nodes(&self) -> impl Iterator<Item = &MemoryNode> {
        self.nodes.values()
    }

    /// All edges in the graph.
    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }

    /// Nodes whose content contains `query` (case-insensitive), sorted by creation time
    /// then id for stable display. An empty/whitespace query returns every node.
    pub fn search(&self, query: &str) -> Vec<&MemoryNode> {
        let needle = query.trim().to_lowercase();
        let mut matches: Vec<&MemoryNode> = self
            .nodes
            .values()
            .filter(|node| needle.is_empty() || node.content.to_lowercase().contains(&needle))
            .collect();
        matches.sort_by(|a, b| a.created_at.cmp(&b.created_at).then(a.id.cmp(&b.id)));
        matches
    }

    /// Remove every edge matching `(from, to, relation)`. Returns how many were removed.
    pub fn remove_edge(&mut self, from: MemoryId, to: MemoryId, relation: Relation) -> usize {
        let before = self.edges.len();
        self.edges
            .retain(|e| !(e.from == from && e.to == to && e.relation == relation));
        before - self.edges.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edge::Relation;
    use crate::node::{MemoryKind, Provenance};

    fn id(n: u128) -> MemoryId {
        MemoryId::from_u128(n)
    }

    fn node(n: u128) -> MemoryNode {
        MemoryNode::new(
            id(n),
            MemoryKind::Fact,
            format!("memory {n}"),
            n as i64,
            Provenance::StatedByUser,
            0.8,
        )
    }

    #[test]
    fn insert_and_get_node() {
        let mut g = MemoryGraph::new();
        assert!(g.is_empty());
        assert!(g.insert_node(node(1)).is_none());
        assert!(!g.is_empty());
        assert_eq!(g.node_count(), 1);
        assert_eq!(g.node(id(1)).unwrap().content, "memory 1");
        assert!(g.node(id(2)).is_none());
    }

    #[test]
    fn nodes_iterates_every_inserted_node() {
        let mut g = MemoryGraph::new();
        g.insert_node(node(1));
        g.insert_node(node(2));
        let mut created: Vec<i64> = g.nodes().map(|n| n.created_at).collect();
        created.sort_unstable();
        assert_eq!(created, vec![1, 2]);
    }

    #[test]
    fn insert_node_replaces_and_returns_previous() {
        let mut g = MemoryGraph::new();
        g.insert_node(node(1));
        let prev = g.insert_node(node(1));
        assert!(prev.is_some());
        assert_eq!(g.node_count(), 1);
    }

    #[test]
    fn insert_edge_requires_both_endpoints() {
        let mut g = MemoryGraph::new();
        g.insert_node(node(1));
        // `to` missing
        assert_eq!(
            g.insert_edge(Edge::new(id(1), id(2), Relation::Knows, 0.5)),
            Err(GraphError::MissingNode(id(2)))
        );
        // `from` missing
        assert_eq!(
            g.insert_edge(Edge::new(id(3), id(1), Relation::Knows, 0.5)),
            Err(GraphError::MissingNode(id(3)))
        );
        assert_eq!(g.edge_count(), 0);

        g.insert_node(node(2));
        assert!(g
            .insert_edge(Edge::new(id(1), id(2), Relation::Knows, 0.5))
            .is_ok());
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn neighbors_follows_outgoing_edges() {
        let mut g = MemoryGraph::new();
        for n in 1..=3 {
            g.insert_node(node(n));
        }
        g.insert_edge(Edge::new(id(1), id(2), Relation::Knows, 0.5))
            .unwrap();
        g.insert_edge(Edge::new(id(1), id(3), Relation::Likes, 0.5))
            .unwrap();
        g.insert_edge(Edge::new(id(2), id(3), Relation::Knows, 0.5))
            .unwrap();

        let mut neighbors: Vec<u128> = g
            .neighbors(id(1))
            .iter()
            .map(|n| n.created_at as u128)
            .collect();
        neighbors.sort_unstable();
        assert_eq!(neighbors, vec![2, 3]);
        assert_eq!(g.neighbors(id(3)).len(), 0);
    }

    #[test]
    fn remove_node_propagates_to_incident_edges() {
        let mut g = MemoryGraph::new();
        for n in 1..=3 {
            g.insert_node(node(n));
        }
        g.insert_edge(Edge::new(id(1), id(2), Relation::Knows, 0.5))
            .unwrap();
        g.insert_edge(Edge::new(id(3), id(1), Relation::Likes, 0.5))
            .unwrap();
        g.insert_edge(Edge::new(id(2), id(3), Relation::Knows, 0.5))
            .unwrap();
        assert_eq!(g.edge_count(), 3);

        let removed = g.remove_node(id(1));
        assert_eq!(removed.unwrap().id, id(1));
        assert_eq!(g.node_count(), 2);
        // Edges touching node 1 (both incoming and outgoing) are gone; the 2->3 edge stays.
        assert_eq!(g.edge_count(), 1);
        assert_eq!(g.edges()[0].from, id(2));
    }

    #[test]
    fn remove_missing_node_is_a_noop() {
        let mut g = MemoryGraph::new();
        g.insert_node(node(1));
        g.insert_edge(Edge::new(id(1), id(1), Relation::RelatedTo, 0.5))
            .unwrap();
        assert!(g.remove_node(id(99)).is_none());
        assert_eq!(g.node_count(), 1);
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn search_matches_content_case_insensitively() {
        let mut g = MemoryGraph::new();
        g.insert_node(MemoryNode::new(
            id(1),
            MemoryKind::Preference,
            "Likes Hiking",
            1,
            Provenance::StatedByUser,
            0.8,
        ));
        g.insert_node(MemoryNode::new(
            id(2),
            MemoryKind::Person,
            "Has a dog",
            2,
            Provenance::StatedByUser,
            0.8,
        ));
        let hits = g.search("HIK");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, id(1));
    }

    #[test]
    fn search_empty_returns_all_sorted_by_created_at() {
        let mut g = MemoryGraph::new();
        g.insert_node(node(3));
        g.insert_node(node(1));
        g.insert_node(node(2));
        let order: Vec<i64> = g.search("  ").iter().map(|n| n.created_at).collect();
        assert_eq!(order, vec![1, 2, 3]);
    }

    #[test]
    fn remove_edge_removes_only_matching_edges() {
        let mut g = MemoryGraph::new();
        g.insert_node(node(1));
        g.insert_node(node(2));
        g.insert_edge(Edge::new(id(1), id(2), Relation::Knows, 0.5))
            .unwrap();
        g.insert_edge(Edge::new(id(1), id(2), Relation::Likes, 0.5))
            .unwrap();
        assert_eq!(g.remove_edge(id(1), id(2), Relation::Knows), 1);
        assert_eq!(g.edge_count(), 1);
        assert_eq!(g.edges()[0].relation, Relation::Likes);
        // Removing again removes nothing.
        assert_eq!(g.remove_edge(id(1), id(2), Relation::Knows), 0);
    }

    #[test]
    fn edges_from_filters_by_source() {
        let mut g = MemoryGraph::new();
        for n in 1..=2 {
            g.insert_node(node(n));
        }
        g.insert_edge(Edge::new(id(1), id(2), Relation::Knows, 0.5))
            .unwrap();
        g.insert_edge(Edge::new(id(2), id(1), Relation::Knows, 0.5))
            .unwrap();
        assert_eq!(g.edges_from(id(1)).count(), 1);
        assert_eq!(g.edges_from(id(1)).next().unwrap().to, id(2));
    }
}
