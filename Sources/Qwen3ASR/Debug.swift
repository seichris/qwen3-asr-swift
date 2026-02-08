import Foundation

enum Qwen3ASRDebug {
    // Evaluate once to avoid repeatedly reading ProcessInfo.
    static let enabled: Bool = {
        let raw = ProcessInfo.processInfo.environment["QWEN3_ASR_DEBUG"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()

        switch raw {
        case "1", "true", "yes", "y", "on":
            return true
        default:
            return false
        }
    }()

    static func log(_ message: @autoclosure () -> String) {
        guard enabled else { return }
        print(message())
    }
}

