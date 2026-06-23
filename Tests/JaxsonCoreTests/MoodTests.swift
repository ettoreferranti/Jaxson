import XCTest
@testable import JaxsonCore

final class MoodTests: XCTestCase {
    func testComponentsAreClamped() {
        let m = MoodVector(valence: 5, arousal: -9)
        XCTAssertEqual(m.valence, 1)
        XCTAssertEqual(m.arousal, -1)
    }

    func testNeutralIsOrigin() {
        XCTAssertEqual(MoodVector.neutral.valence, 0)
        XCTAssertEqual(MoodVector.neutral.arousal, 0)
    }

    func testBlendMovesFractionTowardTarget() {
        let start = MoodVector(valence: 0, arousal: 0)
        let target = MoodVector(valence: 1, arousal: -1)
        let mid = start.blended(toward: target, fraction: 0.5)
        XCTAssertEqual(mid.valence, 0.5, accuracy: 1e-12)
        XCTAssertEqual(mid.arousal, -0.5, accuracy: 1e-12)
    }

    func testBlendFractionIsClamped() {
        let start = MoodVector(valence: 0, arousal: 0)
        let target = MoodVector(valence: 1, arousal: 1)
        XCTAssertEqual(start.blended(toward: target, fraction: 5), target)
        XCTAssertEqual(start.blended(toward: target, fraction: -3), start)
    }

    func testDominantEmotionNeutralNearOrigin() {
        XCTAssertEqual(MoodVector(valence: 0.1, arousal: 0.1).dominantEmotion, .neutral)
    }

    func testDominantEmotionMapsQuadrants() {
        let cases: [(Double, Double, Emotion)] = [
            (0.8, 0.8, .happy),
            (0.8, -0.8, .content),
            (-0.8, 0.8, .upset),
            (-0.8, -0.8, .sad),
        ]
        for (valence, arousal, expected) in cases {
            XCTAssertEqual(
                MoodVector(valence: valence, arousal: arousal).dominantEmotion,
                expected,
                "v=\(valence) a=\(arousal)"
            )
        }
    }

    func testCodableRoundTrip() throws {
        let original = MoodVector(valence: 0.42, arousal: -0.17)
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(MoodVector.self, from: data)
        XCTAssertEqual(decoded, original)
    }
}
