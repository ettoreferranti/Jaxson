import Foundation

/// A point in a circumplex model of affect.
///
/// Jaxson's affect engine produces a `MoodVector`; the face renders from it. Using a
/// continuous valence/arousal space (rather than a fixed list of emotions) lets the
/// face blend and transition smoothly, and lets us *derive* a discrete dominant
/// `Emotion` for cases that want one.
///
/// - `valence`: unpleasant (-1) … pleasant (+1)
/// - `arousal`: calm/sleepy (-1) … excited/alert (+1)
///
/// Both components are always clamped to `[-1, 1]`.
public struct MoodVector: Equatable, Sendable, Codable {
    public private(set) var valence: Double
    public private(set) var arousal: Double

    /// Neutral, calm, alert resting mood.
    public static let neutral = MoodVector(valence: 0, arousal: 0)

    public init(valence: Double, arousal: Double) {
        self.valence = MoodVector.clamp(valence)
        self.arousal = MoodVector.clamp(arousal)
    }

    private static func clamp(_ value: Double) -> Double {
        min(1, max(-1, value))
    }

    /// Linear interpolation toward `target` by `fraction` (clamped to `[0, 1]`).
    ///
    /// The affect engine uses this to smooth mood transitions so the face never
    /// snaps between expressions.
    public func blended(toward target: MoodVector, fraction: Double) -> MoodVector {
        let f = min(1, max(0, fraction))
        return MoodVector(
            valence: valence + (target.valence - valence) * f,
            arousal: arousal + (target.arousal - arousal) * f
        )
    }

    /// The discrete emotion this mood most closely expresses.
    public var dominantEmotion: Emotion {
        // Near the origin we read as neutral rather than forcing a quadrant.
        let magnitude = (valence * valence + arousal * arousal).squareRoot()
        if magnitude < 0.2 { return .neutral }

        switch (valence >= 0, arousal >= 0) {
        case (true, true): return .happy      // pleasant + activated
        case (true, false): return .content   // pleasant + calm
        case (false, true): return .upset      // unpleasant + activated
        case (false, false): return .sad       // unpleasant + low energy
        }
    }
}

/// A discrete emotion the face can render, derived from a `MoodVector`.
public enum Emotion: String, CaseIterable, Sendable, Codable {
    case neutral
    case happy
    case content
    case sad
    case upset
}
