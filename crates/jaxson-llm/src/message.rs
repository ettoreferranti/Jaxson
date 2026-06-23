use serde::{Deserialize, Serialize};

/// Who authored a chat message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// The system / persona instruction that frames Jaxson's behavior.
    System,
    /// The human talking to Jaxson.
    User,
    /// Jaxson itself.
    Assistant,
}

impl Role {
    /// The lowercase tag used by chat templates (`system`, `user`, `assistant`).
    pub fn tag(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }

    /// A capitalized label used by the plain-text template (`System`, `User`, …).
    pub fn label(self) -> &'static str {
        match self {
            Role::System => "System",
            Role::User => "User",
            Role::Assistant => "Assistant",
        }
    }
}

/// A single turn in a conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Message {
            role,
            content: content.into(),
        }
    }

    /// A `system` message.
    pub fn system(content: impl Into<String>) -> Self {
        Message::new(Role::System, content)
    }

    /// A `user` message.
    pub fn user(content: impl Into<String>) -> Self {
        Message::new(Role::User, content)
    }

    /// An `assistant` message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Message::new(Role::Assistant, content)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tags_are_lowercase() {
        assert_eq!(Role::System.tag(), "system");
        assert_eq!(Role::User.tag(), "user");
        assert_eq!(Role::Assistant.tag(), "assistant");
    }

    #[test]
    fn labels_are_capitalized() {
        assert_eq!(Role::System.label(), "System");
        assert_eq!(Role::User.label(), "User");
        assert_eq!(Role::Assistant.label(), "Assistant");
    }

    #[test]
    fn constructors_set_role_and_content() {
        assert_eq!(
            Message::user("hi"),
            Message {
                role: Role::User,
                content: "hi".to_string()
            }
        );
        assert_eq!(Message::system("s").role, Role::System);
        assert_eq!(Message::assistant("a").role, Role::Assistant);
    }

    #[test]
    fn round_trips_through_serde() {
        let m = Message::user("hello");
        let json = serde_json::to_string(&m).unwrap();
        assert_eq!(json, r#"{"role":"user","content":"hello"}"#);
        let back: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(back, m);
    }
}
