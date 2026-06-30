//! The parental-control gate (FR-S3): a passcode the parent sets, stored only as a
//! salted, iterated hash, used to unlock reviewing memories and tuning guardrail
//! [`Strictness`](crate::Strictness).
//!
//! **Threat model.** This guards against the *child* using the device — not a determined
//! attacker. The real perimeter is the OS account and the encrypted memory DB (see
//! `docs/PRIVACY-SECURITY.md`); a short kid-set passcode can't withstand offline brute
//! force regardless of hashing, so the salt + iterations only raise the bar against casual
//! inspection of the stored file. No plaintext passcode is ever stored.
//!
//! Pure and mutation-graded; the UI and where the hash is persisted are the app's job.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Rounds of SHA-256 applied over the salted passcode. Slows casual brute force without
/// making set/verify noticeable.
const ITERATIONS: u32 = 50_000;

/// A parent passcode stored as a salted, iterated SHA-256 hash — never the plaintext.
/// Serializable so the app can persist it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PasscodeHash {
    /// Random per-passcode salt (hex), so equal passcodes hash differently.
    salt: String,
    /// `ITERATIONS`-fold SHA-256 of `salt || passcode` (hex).
    hash: String,
}

impl PasscodeHash {
    /// Hash a freshly chosen `passcode` with a new random salt.
    pub fn new(passcode: &str) -> Self {
        let salt = uuid::Uuid::new_v4().simple().to_string();
        let hash = derive(&salt, passcode);
        PasscodeHash { salt, hash }
    }

    /// Whether `attempt` matches the stored passcode.
    pub fn verify(&self, attempt: &str) -> bool {
        // Re-derive with the stored salt and compare. Equality on the hex digests is a
        // fixed-length compare; the secret here is local, so timing isn't in scope.
        derive(&self.salt, attempt) == self.hash
    }
}

/// Derive the hex digest of `salt || passcode`, folded [`ITERATIONS`] times.
fn derive(salt: &str, passcode: &str) -> String {
    let mut digest = Sha256::digest(format!("{salt}{passcode}").as_bytes());
    for _ in 1..ITERATIONS {
        digest = Sha256::digest(digest);
    }
    hex(&digest)
}

/// Lowercase hex of a byte slice.
fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verifies_the_correct_passcode() {
        let pc = PasscodeHash::new("1234");
        assert!(pc.verify("1234"));
    }

    #[test]
    fn rejects_a_wrong_passcode() {
        let pc = PasscodeHash::new("1234");
        assert!(!pc.verify("4321"));
        assert!(!pc.verify("1234 "));
        assert!(!pc.verify(""));
    }

    #[test]
    fn stores_no_plaintext() {
        let pc = PasscodeHash::new("secret-passcode");
        assert!(!pc.hash.contains("secret-passcode"));
        assert!(!pc.salt.contains("secret-passcode"));
    }

    #[test]
    fn same_passcode_hashes_differently_each_time() {
        // Random salt → distinct hashes (and salts), yet both verify.
        let a = PasscodeHash::new("1234");
        let b = PasscodeHash::new("1234");
        assert_ne!(a.hash, b.hash);
        assert_ne!(a.salt, b.salt);
        assert!(a.verify("1234") && b.verify("1234"));
    }

    #[test]
    fn salt_actually_feeds_the_hash() {
        // Same passcode + same salt is deterministic; a different salt changes the digest.
        assert_eq!(derive("aaaa", "pin"), derive("aaaa", "pin"));
        assert_ne!(derive("aaaa", "pin"), derive("bbbb", "pin"));
    }

    #[test]
    fn round_trips_through_serde() {
        let pc = PasscodeHash::new("0000");
        let json = serde_json::to_string(&pc).unwrap();
        let back: PasscodeHash = serde_json::from_str(&json).unwrap();
        assert_eq!(pc, back);
        assert!(back.verify("0000"));
    }
}
