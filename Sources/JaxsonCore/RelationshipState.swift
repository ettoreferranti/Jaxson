import Foundation

/// The relationship state machine at the core of Jaxson.
///
/// Memories and interactions emit ``RelationshipEvent``s; applying an event is a
/// pure, clamped transition that mutates the scalars below. Because state is a pure
/// function of the event history, behavior is deterministic and testable — which is
/// exactly why the agent's "algorithms" live here rather than in the LLM.
///
/// - `trust`: how safe/open the user is with Jaxson, `[0, 1]`. Gates sensitive
///   topics.
/// - `familiarity`: how much Jaxson knows the user, `[0, 1]`. Low familiarity biases
///   the agent toward asking onboarding questions.
/// - `mood`: the current affective state that drives the face.
public struct RelationshipState: Equatable, Sendable, Codable {
    public private(set) var trust: Double
    public private(set) var familiarity: Double
    public private(set) var mood: MoodVector

    /// A fresh relationship: no trust earned, nothing known, neutral mood — Jaxson
    /// starts knowing nothing about the user (the "Ron's defect" premise).
    public static let initial = RelationshipState(
        trust: 0,
        familiarity: 0,
        mood: .neutral
    )

    public init(trust: Double, familiarity: Double, mood: MoodVector) {
        self.trust = RelationshipState.clampUnit(trust)
        self.familiarity = RelationshipState.clampUnit(familiarity)
        self.mood = mood
    }

    private static func clampUnit(_ value: Double) -> Double {
        min(1, max(0, value))
    }

    /// Familiarity below this threshold means Jaxson should prioritize learning about
    /// the user by asking questions (FR-M1 / FR-M6).
    public static let onboardingFamiliarityThreshold = 0.3

    /// Trust below this threshold keeps sensitive topics locked (FR-M6).
    public static let sensitiveTopicTrustThreshold = 0.5

    /// Whether the agent should currently lead with getting-to-know-you questions.
    public var shouldPrioritizeOnboarding: Bool {
        familiarity < RelationshipState.onboardingFamiliarityThreshold
    }

    /// Whether sensitive topics are currently unlocked.
    public var allowsSensitiveTopics: Bool {
        trust >= RelationshipState.sensitiveTopicTrustThreshold
    }

    /// Apply an event, returning the resulting state (value semantics; deterministic).
    public func applying(_ event: RelationshipEvent) -> RelationshipState {
        switch event {
        case .learnedFact:
            // Each new fact nudges familiarity up with diminishing returns.
            return RelationshipState(
                trust: trust,
                familiarity: familiarity + (1 - familiarity) * 0.05,
                mood: mood
            )
        case .positiveInteraction(let strength):
            let s = clampStrength(strength)
            return RelationshipState(
                trust: trust + (1 - trust) * 0.1 * s,
                familiarity: familiarity,
                mood: mood.blended(toward: MoodVector(valence: 1, arousal: 0.5), fraction: 0.3 * s)
            )
        case .negativeInteraction(let strength):
            let s = clampStrength(strength)
            return RelationshipState(
                trust: trust * (1 - 0.15 * s),
                familiarity: familiarity,
                mood: mood.blended(toward: MoodVector(valence: -1, arousal: 0.3), fraction: 0.3 * s)
            )
        case .moodObserved(let target, let fraction):
            return RelationshipState(
                trust: trust,
                familiarity: familiarity,
                mood: mood.blended(toward: target, fraction: fraction)
            )
        case .memoryForgotten:
            // Deleting a memory should pull familiarity back (FR-M4: deletion is real).
            return RelationshipState(
                trust: trust,
                familiarity: familiarity * 0.95,
                mood: mood
            )
        }
    }

    /// Apply a sequence of events in order.
    public func applying<S: Sequence>(_ events: S) -> RelationshipState
    where S.Element == RelationshipEvent {
        events.reduce(self) { $0.applying($1) }
    }

    private func clampStrength(_ value: Double) -> Double {
        min(1, max(0, value))
    }
}

/// Events that mutate ``RelationshipState``. Memories and interactions are translated
/// into these so all state transitions flow through one well-tested path.
public enum RelationshipEvent: Equatable, Sendable {
    /// Jaxson learned a new fact about the user.
    case learnedFact
    /// A positive exchange of the given strength `[0, 1]`.
    case positiveInteraction(strength: Double)
    /// A negative exchange of the given strength `[0, 1]`.
    case negativeInteraction(strength: Double)
    /// Conversation sentiment pulls mood toward `target` by `fraction`.
    case moodObserved(target: MoodVector, fraction: Double)
    /// A memory was deleted by the user.
    case memoryForgotten
}
