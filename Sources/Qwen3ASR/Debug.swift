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

    // Heavy tensor statistics (mean/std/etc.) can significantly slow realtime and may stress iOS devices.
    // Keep it opt-in even when QWEN3_ASR_DEBUG is enabled.
    static let tensorStatsEnabled: Bool = {
        guard enabled else { return false }
        let raw = ProcessInfo.processInfo.environment["QWEN3_ASR_DEBUG_TENSOR_STATS"]?
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

    static func logTensorStats(_ message: @autoclosure () -> String) {
        guard tensorStatsEnabled else { return }
        print(message())
    }
}
