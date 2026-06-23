//! Turning text into an embedding vector.
//!
//! The agent needs embeddings to store and retrieve memories. The real implementation
//! (F1.4b) will run the local model; [`HashEmbedder`] is a deterministic stand-in
//! based on hashed word buckets — not semantic, but it makes the whole loop runnable
//! and testable without a model, and identical/overlapping text retrieves itself.

/// Produces an embedding vector for a piece of text.
pub trait Embedder {
    fn embed(&self, text: &str) -> Vec<f32>;
}

/// A deterministic bag-of-hashed-words embedder. Each lowercased whitespace token is
/// hashed into one of `dims` buckets; overlapping vocabulary yields positive cosine
/// similarity. Deterministic across runs (fixed FNV-1a hash).
#[derive(Debug, Clone)]
pub struct HashEmbedder {
    dims: usize,
}

impl HashEmbedder {
    pub fn new(dims: usize) -> Self {
        HashEmbedder { dims: dims.max(1) }
    }

    pub fn dims(&self) -> usize {
        self.dims
    }
}

impl Default for HashEmbedder {
    fn default() -> Self {
        HashEmbedder::new(64)
    }
}

impl Embedder for HashEmbedder {
    fn embed(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0.0f32; self.dims];
        for token in text.split_whitespace() {
            let bucket = (fnv1a(token.to_lowercase().as_bytes()) % self.dims as u64) as usize;
            v[bucket] += 1.0;
        }
        v
    }
}

/// FNV-1a, 64-bit — a small, fast, deterministic hash (no RNG seeding like
/// `std`'s default hasher).
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_has_requested_dimensions() {
        assert_eq!(HashEmbedder::new(32).embed("hello world").len(), 32);
        // dims floored at 1.
        assert_eq!(HashEmbedder::new(0).embed("x").len(), 1);
    }

    #[test]
    fn dims_getter_reports_the_configured_size() {
        assert_eq!(HashEmbedder::new(32).dims(), 32);
        assert_eq!(HashEmbedder::new(0).dims(), 1);
    }

    #[test]
    fn fnv1a_matches_known_test_vectors() {
        // Standard FNV-1a 64-bit vectors — pins the hash (offset basis, xor, multiply).
        assert_eq!(fnv1a(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a(b"a"), 0xaf63_dc4c_8601_ec8c);
    }

    #[test]
    fn distinct_words_land_in_distinct_buckets() {
        // Guards against a degenerate hash that sends everything to one bucket.
        let e = HashEmbedder::new(64);
        assert_ne!(e.embed("alpha"), e.embed("beta"));
    }

    #[test]
    fn is_deterministic() {
        let e = HashEmbedder::default();
        assert_eq!(e.embed("I like hiking"), e.embed("I like hiking"));
    }

    #[test]
    fn shared_words_produce_overlapping_vectors() {
        let e = HashEmbedder::new(64);
        let a = e.embed("hiking with the dog");
        let b = e.embed("hiking is great");
        // Dot product > 0 because "hiking" lands in the same bucket in both.
        let dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        assert!(dot > 0.0);
    }

    #[test]
    fn empty_text_is_all_zeros() {
        assert_eq!(HashEmbedder::new(8).embed("   "), vec![0.0; 8]);
    }

    #[test]
    fn counts_repeated_tokens() {
        let e = HashEmbedder::new(64);
        let total: f32 = e.embed("ho ho ho").iter().sum();
        assert_eq!(total, 3.0);
    }
}
