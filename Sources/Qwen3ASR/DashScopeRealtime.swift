import Foundation

public struct DashScopeRealtimeOptions: Sendable {
    public static let defaultModel = "qwen3-asr-flash-realtime-2026-02-10"

    public var model: String
    public var language: String?
    public var sampleRate: Int
    public var enableServerVAD: Bool
    public var vadThreshold: Double
    public var vadSilenceDurationMs: Int
    public var websocketBaseURL: URL
    public var betaHeader: String

    public init(
        model: String = DashScopeRealtimeOptions.defaultModel,
        language: String? = nil,
        sampleRate: Int = 16000,
        enableServerVAD: Bool = true,
        vadThreshold: Double = 0.2,
        vadSilenceDurationMs: Int = 800,
        websocketBaseURL: URL = URL(string: "wss://dashscope.aliyuncs.com/api-ws/v1/realtime")!,
        betaHeader: String = "realtime=v1"
    ) {
        self.model = model
        self.language = language
        self.sampleRate = sampleRate
        self.enableServerVAD = enableServerVAD
        self.vadThreshold = vadThreshold
        self.vadSilenceDurationMs = vadSilenceDurationMs
        self.websocketBaseURL = websocketBaseURL
        self.betaHeader = betaHeader
    }
}

public enum DashScopeRealtimeError: Error, LocalizedError {
    case missingAPIKey
    case invalidURL
    case unsupportedSampleRate(expected: Int, actual: Int)
    case websocketClosed(code: URLSessionWebSocketTask.CloseCode, reason: String)
    case serializationFailed

    public var errorDescription: String? {
        switch self {
        case .missingAPIKey:
            return "Missing DashScope API key. Set DASHSCOPE_API_KEY."
        case .invalidURL:
            return "Failed to build DashScope websocket URL."
        case .unsupportedSampleRate(let expected, let actual):
            return "Unsupported sample rate. Expected \(expected) Hz but got \(actual) Hz."
        case .websocketClosed(let code, let reason):
            return "DashScope websocket closed (\(code.rawValue)): \(reason)"
        case .serializationFailed:
            return "Failed to serialize websocket event."
        }
    }
}

enum DashScopeParsedServerEvent: Equatable {
    case partial(String)
    case final(String)
    case error(String)
    case ignored
}

struct DashScopeRealtimeProtocol {
    static func sessionUpdateEvent(options: DashScopeRealtimeOptions) -> [String: Any] {
        var transcription: [String: Any] = [:]
        if let language = normalizedLanguage(options.language) {
            transcription["language"] = language
        }

        let turnDetection: Any = options.enableServerVAD
            ? [
                "type": "server_vad",
                "threshold": options.vadThreshold,
                "silence_duration_ms": options.vadSilenceDurationMs
            ]
            : NSNull()

        return [
            "event_id": eventId(),
            "type": "session.update",
            "session": [
                "modalities": ["text"],
                "input_audio_format": "pcm",
                "sample_rate": options.sampleRate,
                "input_audio_transcription": transcription,
                "turn_detection": turnDetection
            ]
        ]
    }

    static func appendAudioEvent(base64Audio: String) -> [String: Any] {
        [
            "event_id": eventId(),
            "type": "input_audio_buffer.append",
            "audio": base64Audio
        ]
    }

    static func commitEvent() -> [String: Any] {
        [
            "event_id": eventId(),
            "type": "input_audio_buffer.commit"
        ]
    }

    static func pcm16LEData(from samples: [Float]) -> Data {
        var data = Data(capacity: samples.count * MemoryLayout<Int16>.size)
        for sample in samples {
            let clamped = max(-1.0, min(1.0, sample))
            let scaled: Int16
            if clamped <= -1.0 {
                scaled = Int16.min
            } else {
                scaled = Int16((clamped * Float(Int16.max)).rounded())
            }
            var le = scaled.littleEndian
            withUnsafeBytes(of: &le) { ptr in
                data.append(ptr.bindMemory(to: UInt8.self))
            }
        }
        return data
    }

    static func parseServerEvent(from data: Data) -> DashScopeParsedServerEvent {
        guard
            let object = try? JSONSerialization.jsonObject(with: data),
            let root = object as? [String: Any]
        else {
            return .ignored
        }

        let type = (root["type"] as? String)?.lowercased() ?? ""
        if type == "error" || type.contains(".error") {
            let message = findText(in: root, preferredKeys: ["message", "error", "detail"])
                ?? "Unknown server error"
            return .error(message)
        }

        guard let text = findText(in: root, preferredKeys: ["delta", "transcript", "text"]) else {
            return .ignored
        }
        let cleaned = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleaned.isEmpty else { return .ignored }

        if type.contains("delta") {
            return .partial(cleaned)
        }
        if type.contains("complete") || type.contains("completed") || type.contains("done") || type.contains("final") {
            return .final(cleaned)
        }

        return .ignored
    }

    static func jsonString(from object: [String: Any]) throws -> String {
        guard JSONSerialization.isValidJSONObject(object) else {
            throw DashScopeRealtimeError.serializationFailed
        }
        let data = try JSONSerialization.data(withJSONObject: object, options: [])
        guard let text = String(data: data, encoding: .utf8) else {
            throw DashScopeRealtimeError.serializationFailed
        }
        return text
    }

    private static func eventId() -> String {
        "event_\(UUID().uuidString.replacingOccurrences(of: "-", with: ""))"
    }

    private static func normalizedLanguage(_ raw: String?) -> String? {
        guard let raw else { return nil }
        let cleaned = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        return cleaned.isEmpty || cleaned.lowercased() == "auto" ? nil : cleaned
    }

    private static func findText(in value: Any, preferredKeys: [String], depth: Int = 0) -> String? {
        guard depth <= 8 else { return nil }

        if let dict = value as? [String: Any] {
            for key in preferredKeys {
                if let direct = dict[key] as? String, !direct.isEmpty {
                    return direct
                }
            }
            for (_, nestedValue) in dict {
                if let found = findText(in: nestedValue, preferredKeys: preferredKeys, depth: depth + 1) {
                    return found
                }
            }
            return nil
        }
        if let array = value as? [Any] {
            for item in array {
                if let found = findText(in: item, preferredKeys: preferredKeys, depth: depth + 1) {
                    return found
                }
            }
            return nil
        }
        return nil
    }
}

@available(macOS 13.0, iOS 17.0, *)
public actor DashScopeRealtimeClient {
    public static let apiKeyEnvironmentVariable = "DASHSCOPE_API_KEY"

    private let apiKey: String
    private let options: DashScopeRealtimeOptions
    private let urlSession: URLSession

    public init(
        apiKey: String,
        options: DashScopeRealtimeOptions = .init(),
        urlSession: URLSession = .shared
    ) {
        self.apiKey = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        self.options = options
        self.urlSession = urlSession
    }

    public static func fromEnvironment(options: DashScopeRealtimeOptions = .init()) throws -> DashScopeRealtimeClient {
        let env = ProcessInfo.processInfo.environment
        guard let raw = env[apiKeyEnvironmentVariable], !raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw DashScopeRealtimeError.missingAPIKey
        }
        return DashScopeRealtimeClient(apiKey: raw, options: options)
    }

    public func transcribe(audioSource: any AudioFrameSource) -> AsyncStream<RealtimeTranslationEvent> {
        AsyncStream { continuation in
            let task = Task { [weak self] in
                await self?.run(audioSource: audioSource, continuation: continuation)
            }

            continuation.onTermination = { _ in
                task.cancel()
                Task {
                    await audioSource.stop()
                }
            }
        }
    }

    private func run(
        audioSource: any AudioFrameSource,
        continuation: AsyncStream<RealtimeTranslationEvent>.Continuation
    ) async {
        do {
            guard !apiKey.isEmpty else {
                throw DashScopeRealtimeError.missingAPIKey
            }
            guard audioSource.sampleRate == options.sampleRate else {
                throw DashScopeRealtimeError.unsupportedSampleRate(
                    expected: options.sampleRate,
                    actual: audioSource.sampleRate
                )
            }

            guard var components = URLComponents(url: options.websocketBaseURL, resolvingAgainstBaseURL: false) else {
                throw DashScopeRealtimeError.invalidURL
            }
            components.queryItems = [URLQueryItem(name: "model", value: options.model)]
            guard let url = components.url else {
                throw DashScopeRealtimeError.invalidURL
            }

            var request = URLRequest(url: url)
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
            request.setValue(options.betaHeader, forHTTPHeaderField: "OpenAI-Beta")

            let socket = urlSession.webSocketTask(with: request)
            socket.resume()

            try await sendJSONEvent(DashScopeRealtimeProtocol.sessionUpdateEvent(options: options), over: socket)

            let receiveTask = Task { [weak self] in
                await self?.receiveLoop(from: socket, continuation: continuation)
            }

            let frames = await audioSource.frames()
            try await audioSource.start()
            defer {
                Task { await audioSource.stop() }
            }

            for await frame in frames {
                if Task.isCancelled { break }
                let pcmData = DashScopeRealtimeProtocol.pcm16LEData(from: frame)
                let event = DashScopeRealtimeProtocol.appendAudioEvent(base64Audio: pcmData.base64EncodedString())
                try await sendJSONEvent(event, over: socket)
            }

            if !Task.isCancelled {
                try? await sendJSONEvent(DashScopeRealtimeProtocol.commitEvent(), over: socket)
                try? await Task.sleep(nanoseconds: 750_000_000)
            }

            socket.cancel(with: .normalClosure, reason: nil)
            receiveTask.cancel()
            _ = await receiveTask.result
        } catch {
            continuation.yield(.init(
                kind: .metrics,
                transcript: "",
                translation: nil,
                isStable: true,
                metadata: ["error": String(describing: error)]
            ))
        }

        continuation.finish()
    }

    private func sendJSONEvent(
        _ object: [String: Any],
        over socket: URLSessionWebSocketTask
    ) async throws {
        let payload = try DashScopeRealtimeProtocol.jsonString(from: object)
        try await socket.send(.string(payload))
    }

    private func receiveLoop(
        from socket: URLSessionWebSocketTask,
        continuation: AsyncStream<RealtimeTranslationEvent>.Continuation
    ) async {
        while !Task.isCancelled {
            do {
                let message = try await socket.receive()
                let data: Data
                switch message {
                case .string(let text):
                    data = Data(text.utf8)
                case .data(let binary):
                    data = binary
                @unknown default:
                    continue
                }

                switch DashScopeRealtimeProtocol.parseServerEvent(from: data) {
                case .partial(let text):
                    continuation.yield(.init(kind: .partial, transcript: text, isStable: false))
                case .final(let text):
                    continuation.yield(.init(kind: .final, transcript: text, isStable: true))
                case .error(let message):
                    continuation.yield(.init(
                        kind: .metrics,
                        transcript: "",
                        translation: nil,
                        isStable: true,
                        metadata: ["error": message]
                    ))
                case .ignored:
                    continue
                }
            } catch {
                if Task.isCancelled { return }

                if let urlError = error as? URLError, urlError.code == .networkConnectionLost {
                    continuation.yield(.init(
                        kind: .metrics,
                        transcript: "",
                        translation: nil,
                        isStable: true,
                        metadata: ["error": "DashScope connection lost."]
                    ))
                    return
                }

                let closeCode = socket.closeCode
                if closeCode != .invalid {
                    let reasonData = socket.closeReason ?? Data()
                    let reason = String(data: reasonData, encoding: .utf8) ?? ""
                    continuation.yield(.init(
                        kind: .metrics,
                        transcript: "",
                        translation: nil,
                        isStable: true,
                        metadata: ["error": DashScopeRealtimeError.websocketClosed(code: closeCode, reason: reason).localizedDescription]
                    ))
                    return
                }

                continuation.yield(.init(
                    kind: .metrics,
                    transcript: "",
                    translation: nil,
                    isStable: true,
                    metadata: ["error": String(describing: error)]
                ))
                return
            }
        }
    }
}
