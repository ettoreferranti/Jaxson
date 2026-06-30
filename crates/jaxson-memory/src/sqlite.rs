//! Encrypted-at-rest persistence using SQLCipher (via `rusqlite`).
//!
//! The database file is encrypted with a key the caller holds (in production, from
//! the macOS Keychain — see the privacy model). Opening with the wrong key fails.
//! Reserved SQL words are avoided by naming the edge columns `src`/`dst`.

use std::path::Path;

use rusqlite::{params, Connection};
use uuid::Uuid;

use crate::edge::{Edge, Relation};
use crate::graph::MemoryGraph;
use crate::node::{MemoryId, MemoryKind, MemoryNode, Provenance};
use crate::store::{MemoryStore, StoreError};

const SCHEMA: &str = "
    CREATE TABLE IF NOT EXISTS nodes (
        id          TEXT PRIMARY KEY,
        kind        TEXT NOT NULL,
        content     TEXT NOT NULL,
        created_at  INTEGER NOT NULL,
        provenance  TEXT NOT NULL,
        confidence  REAL NOT NULL,
        embedding   BLOB
    );
    CREATE TABLE IF NOT EXISTS edges (
        src       TEXT NOT NULL,
        dst       TEXT NOT NULL,
        relation  TEXT NOT NULL,
        weight    REAL NOT NULL
    );
";

fn backend<E: std::fmt::Display>(e: E) -> StoreError {
    StoreError::Backend(e.to_string())
}

/// A [`MemoryStore`] backed by an encrypted SQLCipher database on disk.
pub struct SqliteStore {
    conn: Connection,
}

impl SqliteStore {
    /// Open (creating if needed) an encrypted database at `path`, unlocked with `key`.
    ///
    /// Opening an existing database with the wrong key fails (the schema step is the
    /// first real access and surfaces the error).
    pub fn open(path: impl AsRef<Path>, key: &str) -> Result<Self, StoreError> {
        let conn = Connection::open(path).map_err(backend)?;
        conn.pragma_update(None, "key", key).map_err(backend)?;
        conn.execute_batch(SCHEMA).map_err(backend)?;
        Ok(SqliteStore { conn })
    }
}

impl MemoryStore for SqliteStore {
    fn save(&mut self, graph: &MemoryGraph) -> Result<(), StoreError> {
        let tx = self.conn.transaction().map_err(backend)?;
        tx.execute("DELETE FROM edges", []).map_err(backend)?;
        tx.execute("DELETE FROM nodes", []).map_err(backend)?;
        {
            let mut insert_node = tx
                .prepare(
                    "INSERT INTO nodes (id, kind, content, created_at, provenance, confidence, embedding) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                )
                .map_err(backend)?;
            for node in graph.nodes() {
                let embedding = node.embedding.as_ref().map(|v| encode_embedding(v));
                insert_node
                    .execute(params![
                        node.id.to_string(),
                        node.kind.as_str(),
                        node.content,
                        node.created_at,
                        node.provenance.as_str(),
                        node.confidence,
                        embedding,
                    ])
                    .map_err(backend)?;
            }

            let mut insert_edge = tx
                .prepare("INSERT INTO edges (src, dst, relation, weight) VALUES (?1, ?2, ?3, ?4)")
                .map_err(backend)?;
            for edge in graph.edges() {
                insert_edge
                    .execute(params![
                        edge.from.to_string(),
                        edge.to.to_string(),
                        edge.relation.as_str(),
                        edge.weight,
                    ])
                    .map_err(backend)?;
            }
        }
        tx.commit().map_err(backend)?;
        Ok(())
    }

    fn load(&self) -> Result<MemoryGraph, StoreError> {
        let mut graph = MemoryGraph::new();

        let mut node_stmt = self
            .conn
            .prepare("SELECT id, kind, content, created_at, provenance, confidence, embedding FROM nodes")
            .map_err(backend)?;
        let node_rows = node_stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, f64>(5)?,
                    row.get::<_, Option<Vec<u8>>>(6)?,
                ))
            })
            .map_err(backend)?;

        for row in node_rows {
            let (id, kind, content, created_at, provenance, confidence, embedding) =
                row.map_err(backend)?;
            let kind = MemoryKind::from_db_str(&kind)
                .ok_or_else(|| StoreError::Corrupt(format!("unknown memory kind {kind:?}")))?;
            let provenance = Provenance::from_db_str(&provenance)
                .ok_or_else(|| StoreError::Corrupt(format!("unknown provenance {provenance:?}")))?;
            let mut node = MemoryNode::new(
                parse_id(&id)?,
                kind,
                content,
                created_at,
                provenance,
                confidence as f32,
            );
            if let Some(bytes) = embedding {
                node = node.with_embedding(decode_embedding(&bytes)?);
            }
            graph.insert_node(node);
        }

        let mut edge_stmt = self
            .conn
            .prepare("SELECT src, dst, relation, weight FROM edges")
            .map_err(backend)?;
        let edge_rows = edge_stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, f64>(3)?,
                ))
            })
            .map_err(backend)?;

        for row in edge_rows {
            let (src, dst, relation, weight) = row.map_err(backend)?;
            let relation = Relation::from_db_str(&relation)
                .ok_or_else(|| StoreError::Corrupt(format!("unknown relation {relation:?}")))?;
            let edge = Edge::new(parse_id(&src)?, parse_id(&dst)?, relation, weight as f32);
            graph
                .insert_edge(edge)
                .map_err(|e| StoreError::Corrupt(e.to_string()))?;
        }

        Ok(graph)
    }
}

fn parse_id(s: &str) -> Result<MemoryId, StoreError> {
    Uuid::parse_str(s)
        .map(MemoryId)
        .map_err(|e| StoreError::Corrupt(format!("invalid id {s:?}: {e}")))
}

/// Pack an embedding into little-endian f32 bytes.
fn encode_embedding(v: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for f in v {
        bytes.extend_from_slice(&f.to_le_bytes());
    }
    bytes
}

/// Unpack little-endian f32 bytes back into an embedding.
fn decode_embedding(bytes: &[u8]) -> Result<Vec<f32>, StoreError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(StoreError::Corrupt(format!(
            "embedding blob length {} is not a multiple of 4",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_graph() -> MemoryGraph {
        let mut g = MemoryGraph::new();
        g.insert_node(
            MemoryNode::new(
                MemoryId::from_u128(1),
                MemoryKind::Person,
                "Ettore",
                100,
                Provenance::StatedByUser,
                0.9,
            )
            .with_embedding(vec![0.25, -0.5, 0.75]),
        );
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

    fn db_path(dir: &tempfile::TempDir) -> std::path::PathBuf {
        dir.path().join("memory.jaxsondb")
    }

    #[test]
    fn round_trips_through_an_encrypted_db_across_reopens() {
        let dir = tempfile::tempdir().unwrap();
        let path = db_path(&dir);
        let graph = sample_graph();

        {
            let mut store = SqliteStore::open(&path, "correct horse battery").unwrap();
            store.save(&graph).unwrap();
        } // dropped: forces a real reopen below

        let store = SqliteStore::open(&path, "correct horse battery").unwrap();
        assert_eq!(store.load().unwrap(), graph);
    }

    #[test]
    fn the_database_file_is_encrypted_at_rest() {
        // Verify the on-disk file is genuinely encrypted (FR-S4 / privacy): a recognizable
        // secret we store must not appear in the clear, and the file must not be a plaintext
        // SQLite database (SQLCipher encrypts the header too).
        const SECRET: &str = "PEANUT-ALLERGY-SECRET-9f3a2b";
        let dir = tempfile::tempdir().unwrap();
        let path = db_path(&dir);
        {
            let mut store = SqliteStore::open(&path, "a-strong-key").unwrap();
            let mut g = MemoryGraph::new();
            g.insert_node(MemoryNode::new(
                MemoryId::from_u128(1),
                MemoryKind::Fact,
                SECRET,
                0,
                Provenance::StatedByUser,
                1.0,
            ));
            store.save(&g).unwrap();
        }

        let bytes = std::fs::read(&path).unwrap();
        assert!(!bytes.is_empty(), "database file is empty");
        // A plaintext SQLite DB begins with this magic string; an encrypted one does not.
        assert!(
            !bytes.starts_with(b"SQLite format 3\0"),
            "file is an unencrypted SQLite database"
        );
        // The stored content must not be readable on disk.
        let needle = SECRET.as_bytes();
        assert!(
            !bytes.windows(needle.len()).any(|w| w == needle),
            "plaintext memory content leaked to disk"
        );
    }

    #[test]
    fn embeddings_survive_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = SqliteStore::open(db_path(&dir), "k").unwrap();
        store.save(&sample_graph()).unwrap();
        let loaded = store.load().unwrap();
        assert_eq!(
            loaded.node(MemoryId::from_u128(1)).unwrap().embedding,
            Some(vec![0.25, -0.5, 0.75])
        );
    }

    #[test]
    fn fresh_db_loads_empty() {
        let dir = tempfile::tempdir().unwrap();
        let store = SqliteStore::open(db_path(&dir), "k").unwrap();
        assert!(store.load().unwrap().is_empty());
    }

    #[test]
    fn wrong_key_cannot_open_the_database() {
        let dir = tempfile::tempdir().unwrap();
        let path = db_path(&dir);
        {
            let mut store = SqliteStore::open(&path, "the-right-key").unwrap();
            store.save(&sample_graph()).unwrap();
        }
        // Encryption at rest: a different key must not unlock the data.
        assert!(SqliteStore::open(&path, "the-wrong-key").is_err());
    }

    #[test]
    fn save_replaces_previous_state() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = SqliteStore::open(db_path(&dir), "k").unwrap();
        store.save(&sample_graph()).unwrap();
        store.save(&MemoryGraph::new()).unwrap();
        assert!(store.load().unwrap().is_empty());
    }

    #[test]
    fn decode_embedding_rejects_misaligned_blobs() {
        assert!(decode_embedding(&[0, 1, 2]).is_err());
        assert_eq!(decode_embedding(&[]).unwrap(), Vec::<f32>::new());
    }
}
