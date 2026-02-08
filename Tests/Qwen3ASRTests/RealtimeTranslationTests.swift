import XCTest
@testable import Qwen3ASR

/// Tests for realtime translation functionality
@available(macOS 13.0, *)
final class RealtimeTranslationTests: XCTestCase {

    // MARK: - Transcript Stabilizer Tests

    func testTranscriptStabilizerBasic() async {
        let stabilizer = TranscriptStabilizer(config: .init(
            stabilityThreshold: 2,
            minPrefixLength: 3
        ))

        // First update
        var state = await stabilizer.update(transcript: "Hello")
        XCTAssertEqual(state.committed, "")
        XCTAssertEqual(state.pending, "Hello")
        XCTAssertFalse(state.isStable)

        // Same update (first match)
        state = await stabilizer.update(transcript: "Hello")
        XCTAssertEqual(state.committed, "")
        XCTAssertEqual(state.pending, "Hello")
        XCTAssertFalse(state.isStable)

        // Same update again (should stabilize)
        state = await stabilizer.update(transcript: "Hello")
        XCTAssertEqual(state.committed, "Hello")
        XCTAssertEqual(state.pending, "")
        XCTAssertTrue(state.isStable)
    }

    func testTranscriptStabilizerPrefixMatching() async {
        let stabilizer = TranscriptStabilizer(config: .init(
            stabilityThreshold: 2,
            minPrefixLength: 3
        ))

        // Build up text
        _ = await stabilizer.update(transcript: "Hello")
        _ = await stabilizer.update(transcript: "Hello")
        _ = await stabilizer.update(transcript: "Hello")

        // Add more text
        var state = await stabilizer.update(transcript: "Hello world")
        XCTAssertEqual(state.committed, "Hello")
        XCTAssertEqual(state.pending, " world")

        state = await stabilizer.update(transcript: "Hello world")
        state = await stabilizer.update(transcript: "Hello world")

        XCTAssertEqual(state.committed, "Hello world")
        XCTAssertTrue(state.isStable)
    }

    func testTranscriptStabilizerForceCommit() async {
        let stabilizer = TranscriptStabilizer(config: .init(
            stabilityThreshold: 3,
            minPrefixLength: 5
        ))

        _ = await stabilizer.update(transcript: "Hi")

        let state = await stabilizer.forceCommit()
        XCTAssertEqual(state.committed, "Hi")
        XCTAssertEqual(state.pending, "")
    }

    func testTranscriptStabilizerReset() async {
        let stabilizer = TranscriptStabilizer(config: .init(
            stabilityThreshold: 2,
            minPrefixLength: 3
        ))

        _ = await stabilizer.update(transcript: "Hello world")
        _ = await stabilizer.update(transcript: "Hello world")
        _ = await stabilizer.update(transcript: "Hello world")

        await stabilizer.reset()

        let state = await stabilizer.currentState()
        XCTAssertEqual(state.committed, "")
        XCTAssertEqual(state.pending, "")
    }

    // MARK: - Simple VAD Tests

    func testVADSilenceToSpeech() async {
        let vad = SimpleVAD(config: .init(
            energyThreshold: 0.1,
            silenceDurationMs: 500,
            minSpeechDurationMs: 100
        ))

        // Silence
        var state = await vad.process(frame: [Float](repeating: 0.01, count: 320))
        XCTAssertEqual(state, .silence)

        // Speech (high energy)
        state = await vad.process(frame: [Float](repeating: 0.5, count: 320))
        if case .speech = state {
            // Good
        } else {
            XCTFail("Expected speech state")
        }
    }

    // MARK: - Tokenizer Encoding Tests

    func testTokenizerEncodeWords() throws {
        let tokenizer = Qwen3Tokenizer()

        // Create a simple vocab for testing
        // Note: "Hello" (without space) becomes "Hello" in the processed text
        // after stripping leading space and replacing spaces with "Ġ"
        tokenizer.idToToken = [
            0: "Hello",
            1: "Ġworld",
            2: "!",
            3: "Ġtest",
            4: "ing",
            5: "Ġ",
            6: "H",
            7: "ello",
            8: "w",
            9: "o",
            10: "r",
            11: "l",
            12: "d"
        ]

        for (id, token) in tokenizer.idToToken {
            tokenizer.tokenToId[token] = id
        }

        // Test encoding
        let tokens = tokenizer.encode("Hello world!")

        // Check that we got some tokens back
        XCTAssertGreaterThan(tokens.count, 0)
        
        // Should find "Hello" (token 0), "Ġworld" (token 1), and "!" (token 2)
        // The tokenizer processes "Hello world!" as "HelloĠworld!" (leading space stripped, internal spaces -> Ġ)
        XCTAssertTrue(tokens.contains(0), "Should contain 'Hello' token")
        XCTAssertTrue(tokens.contains(1), "Should contain 'Ġworld' token") 
        XCTAssertTrue(tokens.contains(2), "Should contain '!' token")
    }

    // MARK: - Audio Source Tests

    func testFileAudioSource() async throws {
        // Create synthetic audio data
        let sampleRate = 16000
        let duration = 1.0 // 1 second
        let sampleCount = Int(Double(sampleRate) * duration)
        let audioData = [Float](repeating: 0.0, count: sampleCount)

        let source = FileAudioSource(
            audioData: audioData,
            sampleRate: sampleRate,
            frameSizeMs: 20
        )

        try await source.start()

        var frameCount = 0
        for await frame in await source.frames() {
            // Should get 20ms frames at 16kHz = 320 samples
            XCTAssertEqual(frame.count, 320)
            frameCount += 1

            if frameCount >= 10 {
                break
            }
        }

        XCTAssertGreaterThan(frameCount, 0)

        await source.stop()
    }

    func testRealtimeTranslationOptions() {
        let options = RealtimeTranslationOptions(
            targetLanguage: "ja",
            sourceLanguage: "en",
            windowSeconds: 15.0,
            stepMs: 250,
            enableVAD: false,
            enableTranslation: false
        )

        XCTAssertEqual(options.targetLanguage, "ja")
        XCTAssertEqual(options.sourceLanguage, "en")
        XCTAssertEqual(options.windowSeconds, 15.0)
        XCTAssertEqual(options.stepMs, 250)
        XCTAssertFalse(options.enableVAD)
        XCTAssertFalse(options.enableTranslation)
    }

    func testRealtimeTranslationEvent() {
        let event = RealtimeTranslationEvent(
            kind: .partial,
            transcript: "Hello world",
            translation: nil,
            isStable: false
        )

        XCTAssertEqual(event.kind, .partial)
        XCTAssertEqual(event.transcript, "Hello world")
        XCTAssertNil(event.translation)
        XCTAssertFalse(event.isStable)
    }
}
