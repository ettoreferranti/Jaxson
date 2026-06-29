//! The opening line Jaxson shows when a session starts. A brand-new Jaxson introduces
//! itself and asks the user's name; once it has met them it greets them warmly instead —
//! by name when it can find one in the graph. Re-asking a friend their name every time
//! breaks the illusion of a companion that remembers you.
//!
//! Pure: a [`MemoryGraph`] in, a greeting string out, so it's unit- and mutation-tested
//! without a model or a UI.

use jaxson_memory::{MemoryGraph, MemoryKind};

/// First-meeting opener, when Jaxson knows nothing about the user yet.
const FIRST_MEETING: &str = "Hi! I'm Jaxson. What's your name?";

/// Phrases that mark the user's *own* name in a `Person` memory. Deliberately narrow so we
/// don't mistake someone else the user knows ("the user has a friend named Alex") for the
/// user themselves — extraction writes the user's name as "The user's name is Ettore".
const NAME_MARKERS: &[&str] = &["user's name is ", "user is named ", "user is called "];

/// The line Jaxson opens a session with, given what it remembers.
pub(crate) fn opening_greeting(graph: &MemoryGraph) -> String {
    match known_user_name(graph) {
        Some(name) => format!("Hey, {name}! So good to see you again! What's new?"),
        None if knows_user(graph) => {
            "Hey, you're back! I've missed you. What's new?".to_string()
        }
        None => FIRST_MEETING.to_string(),
    }
}

/// Whether Jaxson has met this user before — i.e. it remembers anything at all.
fn knows_user(graph: &MemoryGraph) -> bool {
    graph.node_count() > 0
}

/// The user's first name, if a `Person` memory records it.
fn known_user_name(graph: &MemoryGraph) -> Option<String> {
    graph
        .nodes()
        .filter(|n| n.kind == MemoryKind::Person)
        .find_map(|n| name_from_content(&n.content))
}

/// Pull the user's first name out of a memory sentence like "The user's name is Ettore",
/// or `None` if it isn't a statement of the user's name.
fn name_from_content(content: &str) -> Option<String> {
    let lower = content.to_lowercase();
    for marker in NAME_MARKERS {
        if let Some(pos) = lower.find(marker) {
            // Index into the original (not the lowercased) text to keep the name's casing.
            let rest = &content[pos + marker.len()..];
            if let Some(name) = first_name_token(rest) {
                return Some(name);
            }
        }
    }
    None
}

/// The first whitespace-delimited token of `rest`, stripped of surrounding punctuation —
/// the user's first name. `None` if there's nothing alphanumeric there.
fn first_name_token(rest: &str) -> Option<String> {
    let token = rest.split_whitespace().next()?;
    let cleaned = token.trim_matches(|c: char| !c.is_alphanumeric());
    (!cleaned.is_empty()).then(|| cleaned.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use jaxson_memory::{MemoryId, MemoryNode, Provenance};

    fn person(content: &str) -> MemoryNode {
        MemoryNode::new(
            MemoryId::from_u128(1),
            MemoryKind::Person,
            content,
            0,
            Provenance::StatedByUser,
            0.9,
        )
    }

    fn graph_with(node: MemoryNode) -> MemoryGraph {
        let mut g = MemoryGraph::new();
        g.insert_node(node);
        g
    }

    #[test]
    fn empty_graph_introduces_and_asks_the_name() {
        assert_eq!(opening_greeting(&MemoryGraph::new()), FIRST_MEETING);
    }

    #[test]
    fn greets_by_name_when_the_user_name_is_known() {
        let greeting = opening_greeting(&graph_with(person("The user's name is Ettore")));
        assert!(greeting.contains("Ettore"), "{greeting}");
        assert!(!greeting.contains("What's your name"));
    }

    #[test]
    fn uses_only_the_first_name_and_strips_trailing_punctuation() {
        assert_eq!(
            name_from_content("The user's name is Ettore Ferranti."),
            Some("Ettore".to_string())
        );
        assert_eq!(
            name_from_content("The user is called Sam."),
            Some("Sam".to_string())
        );
    }

    #[test]
    fn does_not_mistake_another_person_for_the_user() {
        // A friend's name must not be read as the user's own name.
        assert_eq!(
            name_from_content("The user has a friend named Alex"),
            None
        );
    }

    #[test]
    fn returning_user_without_a_known_name_gets_a_warm_welcome_back() {
        // Knows *something* (a non-name Person memory) but not the name → no name, no
        // "what's your name", and not the first-meeting line.
        let greeting = opening_greeting(&graph_with(person("The user has a sister")));
        assert_ne!(greeting, FIRST_MEETING);
        assert!(!greeting.contains("What's your name"));
    }
}
