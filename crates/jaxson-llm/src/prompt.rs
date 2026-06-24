//! Prompt assembly: turn a persona, the memories Jaxson has retrieved, and the
//! conversation history into the exact string an instruct model expects.
//!
//! This is deliberately pure and string-only so it is fast to unit-test and
//! mutation-test — getting the chat template wrong is a common, silent source of
//! degraded model output, so we pin the format with tests.

use crate::message::Message;

/// A chat prompt format. Pick the one matching the loaded model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatTemplate {
    /// ChatML (`<|im_start|>` … `<|im_end|>`), used by Qwen and many others.
    ChatMl,
    /// Llama-3 instruct header format.
    Llama3,
    /// A plain `Role: content` transcript — a readable fallback / for base models.
    Plain,
}

impl ChatTemplate {
    /// Render `messages` into a single prompt string, ending with an open assistant
    /// turn so the model continues as Jaxson.
    pub fn render(self, messages: &[Message]) -> String {
        match self {
            ChatTemplate::ChatMl => render_chatml(messages),
            ChatTemplate::Llama3 => render_llama3(messages),
            ChatTemplate::Plain => render_plain(messages),
        }
    }

    /// The end-of-turn token(s) generation should stop at for this template, so a model
    /// doesn't run past its turn (and the token doesn't leak into the reply).
    pub fn stop_tokens(self) -> &'static [&'static str] {
        match self {
            ChatTemplate::ChatMl => &["<|im_end|>"],
            ChatTemplate::Llama3 => &["<|eot_id|>"],
            ChatTemplate::Plain => &[],
        }
    }

    /// Best-guess chat template for a model by name (e.g. an Ollama model id). Llama
    /// models use the Llama-3 format; others default to ChatML (Qwen, etc.). Picking the
    /// wrong template makes models emit garbled control tokens and degrades output.
    pub fn for_model_name(name: &str) -> ChatTemplate {
        if name.to_lowercase().contains("llama") {
            ChatTemplate::Llama3
        } else {
            ChatTemplate::ChatMl
        }
    }
}

fn render_chatml(messages: &[Message]) -> String {
    let mut out = String::new();
    for m in messages {
        out.push_str("<|im_start|>");
        out.push_str(m.role.tag());
        out.push('\n');
        out.push_str(&m.content);
        out.push_str("<|im_end|>\n");
    }
    out.push_str("<|im_start|>assistant\n");
    out
}

fn render_llama3(messages: &[Message]) -> String {
    let mut out = String::from("<|begin_of_text|>");
    for m in messages {
        out.push_str("<|start_header_id|>");
        out.push_str(m.role.tag());
        out.push_str("<|end_header_id|>\n\n");
        out.push_str(&m.content);
        out.push_str("<|eot_id|>");
    }
    out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    out
}

fn render_plain(messages: &[Message]) -> String {
    let mut out = String::new();
    for m in messages {
        out.push_str(m.role.label());
        out.push_str(": ");
        out.push_str(&m.content);
        out.push('\n');
    }
    out.push_str("Assistant:");
    out
}

/// Build the system message that frames Jaxson: its persona, plus the memories it has
/// retrieved for this turn (rendered as a bulleted list). With no memories, the
/// content is just the persona.
pub fn build_system_message(persona: &str, memories: &[String]) -> Message {
    if memories.is_empty() {
        return Message::system(persona);
    }
    let mut content = String::from(persona);
    content.push_str("\n\nThings you remember about the user:");
    for memory in memories {
        content.push_str("\n- ");
        content.push_str(memory);
    }
    Message::system(content)
}

/// Assemble the full message list for a turn: the system message first, then the
/// conversation history in order.
pub fn assemble(persona: &str, memories: &[String], history: &[Message]) -> Vec<Message> {
    let mut messages = Vec::with_capacity(history.len() + 1);
    messages.push(build_system_message(persona, memories));
    messages.extend_from_slice(history);
    messages
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::Role;

    fn convo() -> Vec<Message> {
        vec![Message::system("You are Jaxson."), Message::user("Hi!")]
    }

    #[test]
    fn chatml_wraps_each_turn_and_opens_an_assistant_turn() {
        let out = ChatTemplate::ChatMl.render(&convo());
        assert_eq!(
            out,
            "<|im_start|>system\nYou are Jaxson.<|im_end|>\n\
             <|im_start|>user\nHi!<|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }

    #[test]
    fn llama3_starts_with_bos_and_opens_an_assistant_turn() {
        let out = ChatTemplate::Llama3.render(&convo());
        assert!(out.starts_with("<|begin_of_text|>"));
        assert!(out.contains("<|start_header_id|>user<|end_header_id|>\n\nHi!<|eot_id|>"));
        assert!(out.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn stop_tokens_match_the_template() {
        assert_eq!(ChatTemplate::ChatMl.stop_tokens(), &["<|im_end|>"]);
        assert_eq!(ChatTemplate::Llama3.stop_tokens(), &["<|eot_id|>"]);
        assert!(ChatTemplate::Plain.stop_tokens().is_empty());
    }

    #[test]
    fn template_inferred_from_model_name() {
        assert_eq!(
            ChatTemplate::for_model_name("llama3.1:8b"),
            ChatTemplate::Llama3
        );
        assert_eq!(
            ChatTemplate::for_model_name("Llama-3-Instruct"),
            ChatTemplate::Llama3
        );
        assert_eq!(
            ChatTemplate::for_model_name("qwen3:latest"),
            ChatTemplate::ChatMl
        );
        assert_eq!(
            ChatTemplate::for_model_name("mistral"),
            ChatTemplate::ChatMl
        );
    }

    #[test]
    fn plain_is_a_role_labeled_transcript_cueing_the_assistant() {
        let out = ChatTemplate::Plain.render(&convo());
        assert_eq!(out, "System: You are Jaxson.\nUser: Hi!\nAssistant:");
    }

    #[test]
    fn empty_messages_still_open_an_assistant_turn() {
        assert_eq!(ChatTemplate::ChatMl.render(&[]), "<|im_start|>assistant\n");
        assert!(ChatTemplate::Llama3
            .render(&[])
            .ends_with("assistant<|end_header_id|>\n\n"));
        assert_eq!(ChatTemplate::Plain.render(&[]), "Assistant:");
    }

    #[test]
    fn system_message_without_memories_is_just_the_persona() {
        let m = build_system_message("You are Jaxson.", &[]);
        assert_eq!(m.role, Role::System);
        assert_eq!(m.content, "You are Jaxson.");
    }

    #[test]
    fn system_message_lists_each_memory_as_a_bullet() {
        let m = build_system_message(
            "You are Jaxson.",
            &["likes dogs".to_string(), "name is Ettore".to_string()],
        );
        assert_eq!(
            m.content,
            "You are Jaxson.\n\nThings you remember about the user:\n- likes dogs\n- name is Ettore"
        );
    }

    #[test]
    fn assemble_puts_system_first_then_history_in_order() {
        let history = [
            Message::user("a"),
            Message::assistant("b"),
            Message::user("c"),
        ];
        let out = assemble("persona", &[], &history);
        assert_eq!(out.len(), 4);
        assert_eq!(out[0].role, Role::System);
        assert_eq!(out[1], history[0]);
        assert_eq!(out[2], history[1]);
        assert_eq!(out[3], history[2]);
    }

    #[test]
    fn assemble_injects_memories_into_the_system_message() {
        let out = assemble("persona", &["fact".to_string()], &[]);
        assert_eq!(out.len(), 1);
        assert!(out[0].content.contains("- fact"));
    }
}
