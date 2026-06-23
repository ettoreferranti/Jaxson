//! Build the instruction that asks the model to extract memories as JSON.

use jaxson_llm::Message;

/// System instruction describing the extraction task and the exact JSON schema.
pub const EXTRACTION_SYSTEM: &str = "\
You extract durable facts about the user from a conversation, to remember later. \
Respond with ONLY a JSON object of the form:\n\
{\"memories\":[{\"kind\":\"fact|person|event|preference|episode\",\"content\":\"...\",\"confidence\":0.0-1.0}],\
\"relations\":[{\"from\":<memory index>,\"to\":<memory index>,\"relation\":\"likes|dislikes|knows|related_to|happened_on|causes\",\"weight\":0.0-1.0}]}\n\
Only include things clearly stated or strongly implied. Do not invent details. \
If there is nothing worth remembering, return {\"memories\":[],\"relations\":[]}.";

/// Render the recent turns as a plain `Role: content` transcript.
pub fn transcript(recent: &[Message]) -> String {
    recent
        .iter()
        .map(|m| format!("{}: {}", m.role.label(), m.content))
        .collect::<Vec<_>>()
        .join("\n")
}

/// The messages to send to the model for an extraction pass.
pub fn extraction_messages(recent: &[Message]) -> Vec<Message> {
    vec![
        Message::system(EXTRACTION_SYSTEM),
        Message::user(format!(
            "Conversation:\n{}\n\nExtract memories as JSON.",
            transcript(recent)
        )),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use jaxson_llm::Message;

    #[test]
    fn transcript_labels_each_turn() {
        let turns = [Message::user("hi"), Message::assistant("hello")];
        assert_eq!(transcript(&turns), "User: hi\nAssistant: hello");
    }

    #[test]
    fn extraction_messages_lead_with_the_schema_then_the_transcript() {
        let msgs = extraction_messages(&[Message::user("I love dogs")]);
        assert_eq!(msgs.len(), 2);
        assert!(msgs[0].content.contains("\"memories\""));
        assert!(msgs[0].content.contains("\"relations\""));
        assert!(msgs[1].content.contains("User: I love dogs"));
    }
}
