import XCTest
@testable import Qwen3ASR

final class SecurityRegressionTests: XCTestCase {

    private func writeTempFile(_ data: Data, name: String) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen3asr-tests", isDirectory: true)
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent(name, isDirectory: false)
        try data.write(to: url, options: .atomic)
        return url
    }

    private func makeWAVStub(
        chunkId: String,
        chunkSize: UInt32,
        totalSize: Int
    ) -> Data {
        var data = Data(repeating: 0, count: totalSize)

        func putASCII(_ s: String, at offset: Int) {
            let bytes = Array(s.utf8)
            data.replaceSubrange(offset..<(offset + bytes.count), with: bytes)
        }

        func putU16LE(_ v: UInt16, at offset: Int) {
            var x = v.littleEndian
            withUnsafeBytes(of: &x) { b in
                data.replaceSubrange(offset..<(offset + 2), with: b)
            }
        }

        func putU32LE(_ v: UInt32, at offset: Int) {
            var x = v.littleEndian
            withUnsafeBytes(of: &x) { b in
                data.replaceSubrange(offset..<(offset + 4), with: b)
            }
        }

        // RIFF/WAVE signatures
        putASCII("RIFF", at: 0)
        putASCII("WAVE", at: 8)

        // Minimal fields used by AudioFileLoader.loadWAV() fixed offsets
        putU16LE(1, at: 20)         // audioFormat = PCM
        putU16LE(1, at: 22)         // numChannels = 1
        putU32LE(24_000, at: 24)    // sampleRate
        putU16LE(16, at: 34)        // bitsPerSample = 16

        // First chunk scanned from offset 36
        putASCII(chunkId, at: 36)
        putU32LE(chunkSize, at: 40)

        return data
    }

    func testLoadWAVRejectsChunkSizeBeyondFileBounds() throws {
        // Make a file where the first chunk claims a huge size that would
        // advance beyond file bounds. This should be rejected (no crash).
        let wav = makeWAVStub(
            chunkId: "JUNK",
            chunkSize: 0x7FFF_FFF0,
            totalSize: 64
        )
        let url = try writeTempFile(wav, name: "bad_chunk_bounds.wav")

        XCTAssertThrowsError(try AudioFileLoader.loadWAV(url: url))
    }

    func testLoadWAVRejectsMissingDataChunk() throws {
        // Valid header, but no "data" chunk present.
        let wav = makeWAVStub(
            chunkId: "JUNK",
            chunkSize: 0,
            totalSize: 64
        )
        let url = try writeTempFile(wav, name: "missing_data_chunk.wav")

        XCTAssertThrowsError(try AudioFileLoader.loadWAV(url: url))
    }
}

