import XCTest
@testable import JaxsonCore

final class RelationshipStateTests: XCTestCase {
    func testInitialStateKnowsNothing() {
        let s = RelationshipState.initial
        XCTAssertEqual(s.trust, 0)
        XCTAssertEqual(s.familiarity, 0)
        XCTAssertEqual(s.mood, .neutral)
    }

    func testScalarsAreClamped() {
        let s = RelationshipState(trust: 2, familiarity: -1, mood: .neutral)
        XCTAssertEqual(s.trust, 1)
        XCTAssertEqual(s.familiarity, 0)
    }

    func testFreshAgentPrioritizesOnboarding() {
        XCTAssertTrue(RelationshipState.initial.shouldPrioritizeOnboarding)
        let known = RelationshipState(trust: 0, familiarity: 0.5, mood: .neutral)
        XCTAssertFalse(known.shouldPrioritizeOnboarding)
    }

    func testSensitiveTopicsLockedUntilTrustThreshold() {
        XCTAssertFalse(RelationshipState.initial.allowsSensitiveTopics)
        let trusted = RelationshipState(trust: 0.5, familiarity: 0, mood: .neutral)
        XCTAssertTrue(trusted.allowsSensitiveTopics)
        let almost = RelationshipState(trust: 0.49, familiarity: 0, mood: .neutral)
        XCTAssertFalse(almost.allowsSensitiveTopics)
    }

    func testLearnedFactRaisesFamiliarityWithDiminishingReturns() {
        let once = RelationshipState.initial.applying(.learnedFact)
        XCTAssertGreaterThan(once.familiarity, 0)
        let high = RelationshipState(trust: 0, familiarity: 0.9, mood: .neutral)
        let gainLow = once.familiarity - RelationshipState.initial.familiarity
        let gainHigh = high.applying(.learnedFact).familiarity - high.familiarity
        XCTAssertLessThan(gainHigh, gainLow)
    }

    func testFamiliarityConvergesBelowOne() {
        let events = Array(repeating: RelationshipEvent.learnedFact, count: 200)
        let s = RelationshipState.initial.applying(events)
        XCTAssertLessThanOrEqual(s.familiarity, 1)
        XCTAssertFalse(s.shouldPrioritizeOnboarding)
    }

    func testPositiveInteractionRaisesTrustAndBrightensMood() {
        let s = RelationshipState.initial.applying(.positiveInteraction(strength: 1))
        XCTAssertGreaterThan(s.trust, 0)
        XCTAssertGreaterThan(s.mood.valence, 0)
    }

    func testNegativeInteractionLowersTrustAndDarkensMood() {
        let trusting = RelationshipState(trust: 0.8, familiarity: 0.5, mood: .neutral)
        let s = trusting.applying(.negativeInteraction(strength: 1))
        XCTAssertLessThan(s.trust, 0.8)
        XCTAssertLessThan(s.mood.valence, 0)
        XCTAssertEqual(s.familiarity, 0.5)
    }

    func testInteractionStrengthIsClamped() {
        let huge = RelationshipState.initial.applying(.positiveInteraction(strength: 99))
        let one = RelationshipState.initial.applying(.positiveInteraction(strength: 1))
        XCTAssertEqual(huge, one)
        let negStrength = RelationshipState.initial.applying(.positiveInteraction(strength: -5))
        XCTAssertEqual(negStrength.trust, 0)
    }

    func testForgettingLowersFamiliarity() {
        let known = RelationshipState(trust: 0.5, familiarity: 0.8, mood: .neutral)
        let after = known.applying(.memoryForgotten)
        XCTAssertLessThan(after.familiarity, 0.8)
        XCTAssertEqual(after.trust, 0.5)
    }

    func testMoodObservedBlendsTowardTarget() {
        let happyTarget = MoodVector(valence: 1, arousal: 1)
        let s = RelationshipState.initial.applying(.moodObserved(target: happyTarget, fraction: 0.5))
        XCTAssertEqual(s.mood.valence, 0.5, accuracy: 1e-12)
        XCTAssertEqual(s.mood.arousal, 0.5, accuracy: 1e-12)
    }

    func testApplyingSequenceEqualsFold() {
        let events: [RelationshipEvent] = [
            .learnedFact,
            .positiveInteraction(strength: 0.5),
            .learnedFact,
            .negativeInteraction(strength: 0.2),
        ]
        let folded = events.reduce(RelationshipState.initial) { $0.applying($1) }
        XCTAssertEqual(RelationshipState.initial.applying(events), folded)
    }

    func testCodableRoundTrip() throws {
        let original = RelationshipState(
            trust: 0.3,
            familiarity: 0.6,
            mood: MoodVector(valence: 0.2, arousal: -0.4)
        )
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(RelationshipState.self, from: data)
        XCTAssertEqual(decoded, original)
    }
}
