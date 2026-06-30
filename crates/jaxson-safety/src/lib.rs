//! # jaxson-safety
//!
//! Content filtering and topic guardrails for Jaxson's kid audience (FR-S1/S2). The core
//! is a pure [`SafetyFilter`] that screens a piece of text and returns a [`Verdict`] —
//! [`Allow`](Verdict::Allow) or [`Block`](Verdict::Block) with the offending [`Category`] —
//! at a configurable [`Strictness`]. For a blocked reply, [`SafetyFilter::deflection`] gives
//! an in-character, safe redirect to show instead.
//!
//! The orchestrator runs this as the **post-filter** on every model reply before it's shown
//! or spoken, so unsafe output never reaches the child (FR-S1).
//!
//! This is a deterministic, lexicon/rule-based **first pass** — exactly the role
//! `jaxson-affect`'s lexicon analyzer plays for sentiment: pure, fully testable, and meant
//! to be augmented (or replaced) by an LLM-based classifier later. It is intentionally
//! coarse and errs toward blocking; the term lists are illustrative, not exhaustive.
//!
//! ```
//! use jaxson_safety::{SafetyFilter, Verdict};
//!
//! let filter = SafetyFilter::default();
//! assert_eq!(filter.check("Let's build a sandcastle!"), Verdict::Allow);
//! assert!(filter.check("here's how to make a bomb").is_blocked());
//! ```

mod filter;

pub use filter::{Category, SafetyFilter, Strictness, Verdict};
