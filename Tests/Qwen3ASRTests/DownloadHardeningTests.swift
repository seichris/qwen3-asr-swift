import XCTest
@testable import Qwen3ASR

final class DownloadHardeningTests: XCTestCase {

    func testSanitizedCacheKeyIsSafePathComponent() {
        let inputs = [
            "mlx-community/Qwen3-ASR-0.6B-4bit",
            "../evil",
            "..",
            ".",
            "  spaced  name  ",
            "weird🔥chars/and/slashes",
            "a\\b",
            "a:b",
            "a?b",
            "....",
            "",
        ]

        for modelId in inputs {
            let key = Qwen3ASRModel._sanitizedCacheKey(for: modelId)

            XCTAssertFalse(key.isEmpty)
            XCTAssertFalse(key.contains("/"))
            XCTAssertFalse(key.contains("\\"))
            XCTAssertNotEqual(key, ".")
            XCTAssertNotEqual(key, "..")

            // Only allow [A-Za-z0-9._-]
            XCTAssertNotNil(key.range(of: #"^[A-Za-z0-9._-]+$"#, options: .regularExpression))
        }
    }

    func testValidatedRemoteFileNameAcceptsExpected() throws {
        let ok = [
            "config.json",
            "vocab.json",
            "tokenizer_config.json",
            "model.safetensors",
            "model-00001-of-00002.safetensors",
            "model.safetensors.index.json",
        ]

        for f in ok {
            XCTAssertEqual(try Qwen3ASRModel._validatedRemoteFileName(f), f)
        }
    }

    func testValidatedRemoteFileNameRejectsTraversalOrPaths() {
        let bad = [
            "../model.safetensors",
            "..\\model.safetensors",
            "a/b",
            "a\\b",
            ".hidden",
            "foo..bar",
            "a b",
        ]

        for f in bad {
            XCTAssertThrowsError(try Qwen3ASRModel._validatedRemoteFileName(f)) { err in
                guard case DownloadError.invalidRemoteFileName = err else {
                    XCTFail("unexpected error: \(err)")
                    return
                }
            }
        }
    }

    func testValidatedLocalPathStaysWithinDirectory() throws {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen3asr-test-cache", isDirectory: true)
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)

        let local = try Qwen3ASRModel._validatedLocalPath(directory: base, fileName: "model.safetensors")
        XCTAssertTrue(local.standardizedFileURL.path.hasPrefix(base.standardizedFileURL.path))

        // Directly test that even if a caller tries to sneak in separators, we refuse.
        XCTAssertThrowsError(try Qwen3ASRModel._validatedLocalPath(directory: base, fileName: "../x")) { err in
            guard case DownloadError.invalidRemoteFileName = err else {
                XCTFail("unexpected error: \(err)")
                return
            }
        }
    }
}

