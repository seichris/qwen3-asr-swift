import XCTest
@testable import Qwen3ASR

final class DashScopeRealtimeTests: XCTestCase {
    func testPCM16LEDataClampsSamples() {
        let samples: [Float] = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
        let data = DashScopeRealtimeProtocol.pcm16LEData(from: samples)

        XCTAssertEqual(data.count, samples.count * 2)

        var decoded: [Int16] = []
        decoded.reserveCapacity(samples.count)
        for idx in stride(from: 0, to: data.count, by: 2) {
            let lo = UInt16(data[idx])
            let hi = UInt16(data[idx + 1]) << 8
            decoded.append(Int16(bitPattern: lo | hi))
        }

        XCTAssertEqual(decoded.first, Int16.min)
        XCTAssertEqual(decoded.last, Int16.max)
        XCTAssertEqual(decoded[3], 0)
    }

    func testParsePartialServerEvent() throws {
        let raw = """
        {"type":"response.audio_transcript.delta","delta":"hello"}
        """
        let data = try XCTUnwrap(raw.data(using: .utf8))
        let parsed = DashScopeRealtimeProtocol.parseServerEvent(from: data)
        XCTAssertEqual(parsed, .partial("hello"))
    }

    func testParseFinalServerEventFromNestedTranscript() throws {
        let raw = """
        {"type":"conversation.item.input_audio_transcription.completed","item":{"transcript":"done"}}
        """
        let data = try XCTUnwrap(raw.data(using: .utf8))
        let parsed = DashScopeRealtimeProtocol.parseServerEvent(from: data)
        XCTAssertEqual(parsed, .final("done"))
    }

    func testParseErrorServerEvent() throws {
        let raw = """
        {"type":"error","error":{"message":"bad key"}}
        """
        let data = try XCTUnwrap(raw.data(using: .utf8))
        let parsed = DashScopeRealtimeProtocol.parseServerEvent(from: data)
        XCTAssertEqual(parsed, .error("bad key"))
    }
}
