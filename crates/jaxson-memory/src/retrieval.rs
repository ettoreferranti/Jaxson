//! Hybrid retrieval: pull the memories most relevant to a query back out of the
//! graph, combining **vector similarity** (over node embeddings) with **graph
//! traversal** (spreading relevance along weighted edges).
//!
//! The query embedding is an input here; producing embeddings from text is a separate
//! concern (it needs the model). Everything in this module is pure and deterministic.

use std::cmp::Ordering;
use std::collections::HashMap;

use crate::graph::MemoryGraph;
use crate::node::MemoryId;

/// Tuning for [`retrieve`].
#[derive(Debug, Clone, PartialEq)]
pub struct RetrievalParams {
    /// Maximum number of results to return.
    pub top_k: usize,
    /// How many edge hops to spread relevance from the vector-matched seeds.
    pub max_hops: usize,
    /// Relevance multiplier applied per hop (in `[0.0, 1.0]`); lower = tighter focus.
    pub graph_decay: f32,
    /// Minimum cosine similarity for a node to seed retrieval.
    pub min_similarity: f32,
}

impl Default for RetrievalParams {
    fn default() -> Self {
        RetrievalParams {
            top_k: 8,
            max_hops: 2,
            graph_decay: 0.5,
            min_similarity: 0.0,
        }
    }
}

/// A retrieved memory and its relevance score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Retrieved {
    pub id: MemoryId,
    pub score: f32,
}

/// Cosine similarity of two equal-length vectors, in `[-1.0, 1.0]`. Returns `0.0` for
/// empty, mismatched-length, or zero-magnitude vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}

/// Retrieve the memories most relevant to `query`, ranked by score (descending, with
/// ties broken by id for determinism), capped at `params.top_k`.
///
/// 1. **Seed** every embedded node whose cosine similarity to `query` is greater than
///    `min_similarity`, scored by that similarity.
/// 2. **Spread** relevance up to `max_hops`: each hop, a node reachable via an edge
///    `a -> b` gains a candidate score `score(a) * graph_decay * edge.weight`,
///    keeping the best score seen. This lets associated memories (even ones without
///    embeddings) surface alongside direct matches.
pub fn retrieve(graph: &MemoryGraph, query: &[f32], params: &RetrievalParams) -> Vec<Retrieved> {
    let mut scores: HashMap<MemoryId, f32> = HashMap::new();

    // 1. Vector-similarity seeds: embedded nodes scoring above the threshold.
    for node in graph.nodes() {
        if let Some(embedding) = &node.embedding {
            let sim = cosine_similarity(query, embedding);
            if sim > params.min_similarity {
                scores.insert(node.id, sim);
            }
        }
    }

    // 2. Spread relevance one hop per iteration (max-product relaxation): a node keeps
    //    the best score reachable from any seed within `max_hops`.
    for _ in 0..params.max_hops {
        let frontier = scores.clone();
        for edge in graph.edges() {
            if let Some(&from_score) = frontier.get(&edge.from) {
                let candidate = from_score * params.graph_decay * edge.weight;
                if candidate > 0.0 {
                    let best = scores
                        .get(&edge.to)
                        .map_or(candidate, |&current| current.max(candidate));
                    scores.insert(edge.to, best);
                }
            }
        }
    }

    let mut ranked: Vec<Retrieved> = scores
        .into_iter()
        .map(|(id, score)| Retrieved { id, score })
        .collect();
    ranked.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(Ordering::Equal)
            .then(a.id.cmp(&b.id))
    });
    ranked.truncate(params.top_k);
    ranked
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edge::{Edge, Relation};
    use crate::node::{MemoryKind, MemoryNode, Provenance};

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-6
    }

    fn id(n: u128) -> MemoryId {
        MemoryId::from_u128(n)
    }

    fn embedded(n: u128, embedding: Vec<f32>) -> MemoryNode {
        MemoryNode::new(
            id(n),
            MemoryKind::Fact,
            format!("memory {n}"),
            n as i64,
            Provenance::StatedByUser,
            0.8,
        )
        .with_embedding(embedding)
    }

    #[test]
    fn cosine_identical_orthogonal_opposite() {
        assert!(approx(cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]), 1.0));
        assert!(approx(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]), 0.0));
        assert!(approx(cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]), -1.0));
    }

    #[test]
    fn cosine_handles_degenerate_inputs() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 2.0]), 0.0);
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
    }

    #[test]
    fn cosine_is_magnitude_invariant() {
        assert!(approx(cosine_similarity(&[2.0, 0.0], &[9.0, 0.0]), 1.0));
    }

    #[test]
    fn ranks_nearest_embedding_first() {
        let mut g = MemoryGraph::new();
        g.insert_node(embedded(1, vec![1.0, 0.0]));
        g.insert_node(embedded(2, vec![0.0, 1.0]));
        let out = retrieve(&g, &[0.9, 0.1], &RetrievalParams::default());
        assert_eq!(out[0].id, id(1));
        assert!(out[0].score > out[1].score);
    }

    #[test]
    fn top_k_limits_results() {
        let mut g = MemoryGraph::new();
        for n in 1..=5 {
            g.insert_node(embedded(n, vec![1.0, 0.0]));
        }
        let params = RetrievalParams {
            top_k: 3,
            ..Default::default()
        };
        assert_eq!(retrieve(&g, &[1.0, 0.0], &params).len(), 3);
    }

    #[test]
    fn graph_traversal_surfaces_an_unembedded_neighbor() {
        let mut g = MemoryGraph::new();
        g.insert_node(embedded(1, vec![1.0, 0.0]));
        // Node 2 has no embedding — only reachable via the edge from node 1.
        g.insert_node(MemoryNode::new(
            id(2),
            MemoryKind::Person,
            "no embedding",
            2,
            Provenance::StatedByUser,
            0.5,
        ));
        g.insert_edge(Edge::new(id(1), id(2), Relation::Knows, 0.5))
            .unwrap();

        let params = RetrievalParams {
            graph_decay: 0.5,
            ..Default::default()
        };
        let out = retrieve(&g, &[1.0, 0.0], &params);
        let two = out
            .iter()
            .find(|r| r.id == id(2))
            .expect("neighbor retrieved");
        // score(2) = sim(1)=1.0 * decay 0.5 * weight 0.5 = 0.25
        assert!(approx(two.score, 0.25));
    }

    #[test]
    fn max_hops_bounds_the_spread() {
        // 1 (embedded) -> 2 -> 3, no embeddings on 2/3.
        let mut g = MemoryGraph::new();
        g.insert_node(embedded(1, vec![1.0, 0.0]));
        for n in 2..=3 {
            g.insert_node(MemoryNode::new(
                id(n),
                MemoryKind::Fact,
                "x",
                n as i64,
                Provenance::StatedByUser,
                0.5,
            ));
        }
        g.insert_edge(Edge::new(id(1), id(2), Relation::Knows, 1.0))
            .unwrap();
        g.insert_edge(Edge::new(id(2), id(3), Relation::Knows, 1.0))
            .unwrap();

        let one_hop = RetrievalParams {
            max_hops: 1,
            top_k: 10,
            ..Default::default()
        };
        let ids: Vec<MemoryId> = retrieve(&g, &[1.0, 0.0], &one_hop)
            .iter()
            .map(|r| r.id)
            .collect();
        assert!(ids.contains(&id(2)));
        assert!(!ids.contains(&id(3))); // two hops away, out of reach

        let two_hop = RetrievalParams {
            max_hops: 2,
            ..one_hop
        };
        let ids: Vec<MemoryId> = retrieve(&g, &[1.0, 0.0], &two_hop)
            .iter()
            .map(|r| r.id)
            .collect();
        assert!(ids.contains(&id(3)));
    }

    #[test]
    fn min_similarity_filters_weak_seeds() {
        let mut g = MemoryGraph::new();
        g.insert_node(embedded(1, vec![1.0, 0.0])); // sim 1.0 to query
        g.insert_node(embedded(2, vec![0.3, 1.0])); // weak sim to query
        let params = RetrievalParams {
            min_similarity: 0.9,
            max_hops: 0,
            ..Default::default()
        };
        let out = retrieve(&g, &[1.0, 0.0], &params);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].id, id(1));
    }

    #[test]
    fn empty_query_retrieves_nothing() {
        let mut g = MemoryGraph::new();
        g.insert_node(embedded(1, vec![1.0, 0.0]));
        assert!(retrieve(&g, &[], &RetrievalParams::default()).is_empty());
    }

    #[test]
    fn ties_break_by_id_for_determinism() {
        let mut g = MemoryGraph::new();
        // Same embedding => identical similarity => deterministic id order.
        g.insert_node(embedded(2, vec![1.0, 0.0]));
        g.insert_node(embedded(1, vec![1.0, 0.0]));
        let out = retrieve(&g, &[1.0, 0.0], &RetrievalParams::default());
        assert_eq!(out[0].id, id(1));
        assert_eq!(out[1].id, id(2));
    }

    #[test]
    fn similarity_exactly_at_threshold_is_excluded() {
        // Orthogonal node has similarity 0.0; the threshold is strict (`>`).
        let mut g = MemoryGraph::new();
        g.insert_node(embedded(1, vec![0.0, 1.0]));
        assert!(retrieve(&g, &[1.0, 0.0], &RetrievalParams::default()).is_empty());
    }

    #[test]
    fn zero_weight_edge_does_not_spread() {
        let mut g = MemoryGraph::new();
        g.insert_node(embedded(1, vec![1.0, 0.0]));
        g.insert_node(MemoryNode::new(
            id(2),
            MemoryKind::Fact,
            "x",
            2,
            Provenance::StatedByUser,
            0.5,
        ));
        g.insert_edge(Edge::new(id(1), id(2), Relation::Knows, 0.0))
            .unwrap();
        let out = retrieve(&g, &[1.0, 0.0], &RetrievalParams::default());
        assert!(out.iter().all(|r| r.id != id(2)));
    }

    /// Build: seed A (sim 1.0) and seed B (sim 0.5), both edge->C (no embedding),
    /// weight 1.0, decay 0.5. C's best score is max(0.5, 0.25) = 0.5 regardless of the
    /// order edges are visited — this pins the "keep the best path" relaxation.
    fn best_path_graph(edge_a_first: bool) -> MemoryGraph {
        let mut g = MemoryGraph::new();
        g.insert_node(embedded(1, vec![1.0, 0.0])); // A, sim 1.0
        g.insert_node(embedded(2, vec![0.5, 0.866_025_4])); // B, sim 0.5
        g.insert_node(MemoryNode::new(
            id(3),
            MemoryKind::Fact,
            "C",
            3,
            Provenance::StatedByUser,
            0.5,
        ));
        let a_to_c = Edge::new(id(1), id(3), Relation::Knows, 1.0);
        let b_to_c = Edge::new(id(2), id(3), Relation::Knows, 1.0);
        if edge_a_first {
            g.insert_edge(a_to_c).unwrap();
            g.insert_edge(b_to_c).unwrap();
        } else {
            g.insert_edge(b_to_c).unwrap();
            g.insert_edge(a_to_c).unwrap();
        }
        g
    }

    #[test]
    fn keeps_best_path_when_stronger_edge_is_first() {
        let g = best_path_graph(true);
        let out = retrieve(&g, &[1.0, 0.0], &RetrievalParams::default());
        let c = out.iter().find(|r| r.id == id(3)).unwrap();
        assert!(approx(c.score, 0.5));
    }

    #[test]
    fn keeps_best_path_when_weaker_edge_is_first() {
        let g = best_path_graph(false);
        let out = retrieve(&g, &[1.0, 0.0], &RetrievalParams::default());
        let c = out.iter().find(|r| r.id == id(3)).unwrap();
        assert!(approx(c.score, 0.5));
    }
}
