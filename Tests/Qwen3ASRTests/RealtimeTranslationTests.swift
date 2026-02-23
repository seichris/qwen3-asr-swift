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

        // Build a minimal byte-level tokenizer fixture on disk.
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen3_tokenizer_test_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)

        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let vocab: [String: Int] = [
            "H": 0, "e": 1, "l": 2, "o": 3,
            "Ġ": 4, "w": 5, "r": 6, "d": 7, "!": 8
        ]
        let vocabData = try JSONSerialization.data(withJSONObject: vocab, options: [])
        try vocabData.write(to: tmpDir.appendingPathComponent("vocab.json"))

        // Empty-but-valid merges file (character-level fallback).
        let merges = "#version: 0.2\n"
        try merges.write(to: tmpDir.appendingPathComponent("merges.txt"), atomically: true, encoding: .utf8)

        try tokenizer.load(from: tmpDir.appendingPathComponent("vocab.json"))

        // Test encoding
        let tokens = tokenizer.encode("Hello world!")

        XCTAssertGreaterThan(tokens.count, 0)
        XCTAssertTrue(tokens.contains(8), "Should contain '!' token")

        let decoded = tokenizer.decode(tokens: tokens)
        XCTAssertEqual(decoded, "Hello world!")
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
