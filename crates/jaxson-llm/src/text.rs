//! Small pure helpers for cleaning up model output.

/// Remove `<think>…</think>` reasoning blocks that "thinking" models (e.g. Qwen3) emit
/// before their actual answer, returning the trimmed remainder.
///
/// Handles multiple blocks and surrounding text. An unclosed `<think>` (e.g. generation
/// hit the token limit mid-thought) drops everything from it onward.
pub fn strip_reasoning(text: &str) -> String {
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";

    let mut out = String::new();
    let mut rest = text;
    loop {
        match rest.find(OPEN) {
            None => {
                out.push_str(rest);
                break;
            }
            Some(start) => {
                out.push_str(&rest[..start]);
                let after_open = &rest[start + OPEN.len()..];
                match after_open.find(CLOSE) {
                    Some(end) => rest = &after_open[end + CLOSE.len()..],
                    None => break, // unclosed: drop the rest
                }
            }
        }
    }
    out.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leaves_plain_text_alone_but_trims() {
        assert_eq!(strip_reasoning("  hello there  "), "hello there");
    }

    #[test]
    fn removes_a_leading_reasoning_block() {
        assert_eq!(
            strip_reasoning("<think>hmm, let me think</think>Hello!"),
            "Hello!"
        );
    }

    #[test]
    fn keeps_text_on_both_sides() {
        assert_eq!(strip_reasoning("a<think>x</think>b"), "ab");
    }

    #[test]
    fn removes_multiple_and_multiline_blocks() {
        let input = "one<think>r1</think>two<think>line1\nline2</think>three";
        assert_eq!(strip_reasoning(input), "onetwothree");
    }

    #[test]
    fn unclosed_think_drops_the_remainder() {
        assert_eq!(strip_reasoning("answer<think>still thinking..."), "answer");
        assert_eq!(strip_reasoning("<think>only thinking, cut off"), "");
    }

    #[test]
    fn text_without_tags_is_unchanged() {
        assert_eq!(
            strip_reasoning("just a normal reply"),
            "just a normal reply"
        );
    }
}
