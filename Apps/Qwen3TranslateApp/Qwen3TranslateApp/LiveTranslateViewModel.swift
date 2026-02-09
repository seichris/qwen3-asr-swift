import Foundation
import Qwen3ASR
import os

#if canImport(Translation)
import Translation
#endif

@MainActor
final class LiveTranslateViewModel: ObservableObject {
    private let log = Logger(subsystem: "Qwen3TranslateApp", category: "LiveTranslate")

    enum Status: Equatable {
        case idle
        case loadingModel(progress: Double, status: String)
        case ready
        case running
        case error(String)
    }

    struct Segment: Identifiable, Equatable {
        let id: UUID
        var transcript: String
        var translation: String?
        var date: Date
    }

    @Published var status: Status = .idle
    @Published var partialTranscript: String = ""
    @Published var segments: [Segment] = []
    @Published var isRunning: Bool = false
    @Published var isStopping: Bool = false
    @Published var debugFileTranscript: String = ""
    @Published var debugFileInfo: String = ""
    @Published var debugIsTranscribingFile: Bool = false

    private var model: Qwen3ASRModel?
    private var activeAudioSource: MicrophoneAudioSource?
    private var stopRequested: Bool = false

    func clear() {
        partialTranscript = ""
        segments.removeAll()
    }

    func unloadModel() {
        model = nil
        if !isRunning {
            status = .idle
        }
    }

    func transcribeAudioFile(modelId: String, from: SupportedLanguage, url: URL) async {
        // This is meant as a "first principles" debugging path: single-shot file transcription,
        // closest to the CLI `transcribe` command, bypassing microphone capture, VAD, and stabilizer.
        debugIsTranscribingFile = true
        debugFileTranscript = ""
        debugFileInfo = url.lastPathComponent
        defer { debugIsTranscribingFile = false }

        do {
            let model = try await loadModelIfNeeded(modelId: modelId)
            let raw = await Task(priority: .userInitiated) {
                autoreleasepool {
                    do {
                        let samples = try AudioFileLoader.load(url: url, targetSampleRate: 24000)
                        let text = model.transcribe(
                            audio: samples,
                            sampleRate: 24000,
                            language: from.modelName,
                            maxTokens: 448
                        )
                        return (samples.count, text)
                    } catch {
                        return (0, "Error: \(String(describing: error))")
                    }
                }
            }.value

            let (nSamples, rawText) = raw
            debugFileInfo = "\(url.lastPathComponent) (\(nSamples) samples @ 24kHz)"
            debugFileTranscript = extractASRText(from: rawText)
        } catch {
            debugFileTranscript = "Error: \(String(describing: error))"
        }
    }

    func start() {
        log.info("start() requested")
        stopRequested = false
        isStopping = false
        clear()
        isRunning = true
    }

    func requestStop() {
        guard isRunning else { return }
        log.info("requestStop() requested")
        stopRequested = true
        isStopping = true

        // Best-effort: stop capture immediately. The streaming task will drain and then exit.
        if let src = activeAudioSource {
            Task { await src.stop() }
        }
    }

    @available(iOS 18.0, macOS 15.0, *)
    func runNoTranslation(modelId: String, from: SupportedLanguage) async {
        status = .running

        do {
            if stopRequested {
                isStopping = false
                isRunning = false
                status = (model == nil) ? .idle : .ready
                return
            }

            log.info("runNoTranslation() start. from=\(from.displayName, privacy: .public)")
            let model = try await loadModelIfNeeded(modelId: modelId)

            // iOS devices (especially in Debug) can fall behind realtime; use enough buffering to avoid dropping frames.
            let audioSource = MicrophoneAudioSource(frameSizeMs: 20, bufferedFrames: 500)
            activeAudioSource = audioSource
            defer {
                activeAudioSource = nil
                isStopping = false
                isRunning = false
                if case .running = status {
                    status = (self.model == nil) ? .idle : .ready
                }
            }

            let windowSeconds = 10.0
            let stepMs = 500

            let options = RealtimeTranslationOptions(
                targetLanguage: "English",
                sourceLanguage: from.modelName,
                windowSeconds: windowSeconds,
                stepMs: stepMs,
                enableVAD: true,
                enableTranslation: false
            )

            let stream = await model.realtimeTranslate(
                audioSource: audioSource,
                options: options
            )

            for await event in stream {
                if Task.isCancelled { break }
                if stopRequested { break }
                await handleNoTranslation(event: event)
            }

            if !Task.isCancelled {
                activeAudioSource = nil
                isStopping = false
                isRunning = false
                status = .ready
                log.info("runNoTranslation() finished")
            }
        } catch {
            if !Task.isCancelled {
                log.error("runNoTranslation() error. error=\(String(describing: error), privacy: .public)")
                activeAudioSource = nil
                isStopping = false
                isRunning = false
                status = .error(String(describing: error))
            }
        }
    }

    @available(iOS 18.0, macOS 15.0, *)
    func runGoogleTranslation(modelId: String, from: SupportedLanguage, to: SupportedLanguage) async {
        status = .running

        do {
            if stopRequested {
                isStopping = false
                isRunning = false
                status = (model == nil) ? .idle : .ready
                return
            }

            guard let apiKey = ProcessInfo.processInfo.environment["QWEN3_ASR_GOOGLE_TRANSLATE_API_KEY"],
                  !apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            else {
                throw MissingGoogleAPIKeyError()
            }

            log.info("runGoogleTranslation() start. from=\(from.displayName, privacy: .public) to=\(to.displayName, privacy: .public)")

            let model = try await loadModelIfNeeded(modelId: modelId)

            let audioSource = MicrophoneAudioSource(frameSizeMs: 20, bufferedFrames: 500)
            activeAudioSource = audioSource
            defer {
                activeAudioSource = nil
                isStopping = false
                isRunning = false
                if case .running = status {
                    status = (self.model == nil) ? .idle : .ready
                }
            }

            let windowSeconds = 10.0
            let stepMs = 500

            let options = RealtimeTranslationOptions(
                targetLanguage: "English",
                sourceLanguage: from.modelName,
                windowSeconds: windowSeconds,
                stepMs: stepMs,
                enableVAD: true,
                enableTranslation: false
            )

            let stream = await model.realtimeTranslate(
                audioSource: audioSource,
                options: options
            )

            for await event in stream {
                if Task.isCancelled { break }
                if stopRequested { break }
                await handleGoogle(event: event, apiKey: apiKey, from: from, to: to)
            }

            if !Task.isCancelled {
                activeAudioSource = nil
                isStopping = false
                isRunning = false
                status = .ready
                log.info("runGoogleTranslation() finished")
            }
        } catch {
            if !Task.isCancelled {
                log.error("runGoogleTranslation() error. error=\(String(describing: error), privacy: .public)")
                activeAudioSource = nil
                isStopping = false
                isRunning = false
                status = .error(String(describing: error))
            }
        }
    }

    #if canImport(Translation)
    @available(iOS 18.0, macOS 15.0, *)
    func run(
        translationSession: TranslationSession,
        modelId: String,
        from: SupportedLanguage,
        to: SupportedLanguage
    ) async {
        status = .running

        do {
            if stopRequested {
                // Start immediately followed by stop can happen if UI toggles quickly.
                isStopping = false
                isRunning = false
                status = (model == nil) ? .idle : .ready
                return
            }

            log.info("run() start. from=\(from.displayName, privacy: .public) to=\(to.displayName, privacy: .public)")

            let model = try await loadModelIfNeeded(modelId: modelId)

            let audioSource = MicrophoneAudioSource(frameSizeMs: 20, bufferedFrames: 500)
            activeAudioSource = audioSource
            defer {
                activeAudioSource = nil
                isStopping = false
                isRunning = false
                if case .running = status {
                    status = (self.model == nil) ? .idle : .ready
                }
            }
            let windowSeconds = 10.0
            let stepMs = 500

            let options = RealtimeTranslationOptions(
                targetLanguage: "English",
                sourceLanguage: from.modelName,
                windowSeconds: windowSeconds,
                stepMs: stepMs,
                enableVAD: true,
                enableTranslation: false
            )

            let stream = await model.realtimeTranslate(
                audioSource: audioSource,
                options: options
            )

            for await event in stream {
                if Task.isCancelled { break }
                if stopRequested { break }
                await handle(event: event, translationSession: translationSession)
            }

            if !Task.isCancelled {
                activeAudioSource = nil
                isStopping = false
                isRunning = false
                status = .ready
                log.info("run() finished")
            }
        } catch {
            if !Task.isCancelled {
                log.error("run() error. error=\(String(describing: error), privacy: .public)")
                activeAudioSource = nil
                isStopping = false
                isRunning = false
                status = .error(String(describing: error))
            }
        }
    }
    #endif

    private func handleNoTranslation(event: RealtimeTranslationEvent) async {
        switch event.kind {
        case .partial:
            partialTranscript = event.transcript

        case .final:
            partialTranscript = ""
            let cleaned = event.transcript.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !cleaned.isEmpty else { return }
            segments.append(.init(id: UUID(), transcript: cleaned, translation: nil, date: event.timestamp))

        case .metrics:
            if let err = event.metadata?["error"], !err.isEmpty {
                log.error("metrics error: \(err, privacy: .public)")
                status = .error(err)
            }

        case .translation:
            break
        }
    }

    @available(iOS 18.0, macOS 15.0, *)
    private func handleGoogle(
        event: RealtimeTranslationEvent,
        apiKey: String,
        from: SupportedLanguage,
        to: SupportedLanguage
    ) async {
        switch event.kind {
        case .partial:
            partialTranscript = event.transcript

        case .final:
            partialTranscript = ""
            let cleaned = event.transcript.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !cleaned.isEmpty else { return }

            let id = UUID()
            segments.append(.init(id: id, transcript: cleaned, translation: nil, date: event.timestamp))

            if stopRequested { return }
            if Task.isCancelled { return }

            do {
                let translated = try await translateGoogleWithTimeout(
                    cleaned,
                    apiKey: apiKey,
                    sourceLanguage: from.id,
                    targetLanguage: to.id,
                    timeoutSeconds: 1.2
                )
                if Task.isCancelled { return }
                if stopRequested { return }
                let t = translated.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !t.isEmpty else { return }
                guard let idx = segments.lastIndex(where: { $0.id == id }) else { return }
                segments[idx].translation = t
            } catch {
                // Best-effort: keep transcription running if translation fails/cancels.
                log.debug("google translation failed (ignored): \(String(describing: error), privacy: .public)")
                return
            }

        case .metrics:
            if let err = event.metadata?["error"], !err.isEmpty {
                log.error("metrics error: \(err, privacy: .public)")
                status = .error(err)
            }

        case .translation:
            break
        }
    }

    #if canImport(Translation)
    @available(iOS 18.0, macOS 15.0, *)
    private func handle(event: RealtimeTranslationEvent, translationSession: TranslationSession) async {
        switch event.kind {
        case .partial:
            partialTranscript = event.transcript

        case .final:
            partialTranscript = ""
            let cleaned = event.transcript.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !cleaned.isEmpty else { return }

            let id = UUID()
            segments.append(.init(id: id, transcript: cleaned, translation: nil, date: event.timestamp))

            // If we're stopping, don't start new translation work.
            if stopRequested { return }
            if Task.isCancelled { return }
            do {
                // Translation can be slow and should not block Stop. Bound it so we don't hang the run loop.
                let translated = try await translateWithTimeout(
                    cleaned,
                    using: translationSession,
                    timeoutSeconds: 1.2
                )
                if Task.isCancelled { return }
                if stopRequested { return }
                let t = translated.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !t.isEmpty else { return }
                guard let idx = segments.lastIndex(where: { $0.id == id }) else { return }
                segments[idx].translation = t
            } catch {
                // Best-effort: keep transcription running if translation fails/cancels.
                log.debug("translation failed (ignored): \(String(describing: error), privacy: .public)")
                return
            }

        case .metrics:
            if let err = event.metadata?["error"], !err.isEmpty {
                log.error("metrics error: \(err, privacy: .public)")
                status = .error(err)
            }

        case .translation:
            // Not used in the app path; translation is computed here to keep TranslationSession scoped
            // to the `.translationTask` lifecycle.
            break
        }
    }

    @available(iOS 18.0, macOS 15.0, *)
    private func translateWithTimeout(
        _ text: String,
        using session: TranslationSession,
        timeoutSeconds: Double
    ) async throws -> String {
        try await withThrowingTaskGroup(of: String.self) { group in
            group.addTask {
                try await AppleTranslation.translate(text, using: session)
            }
            group.addTask {
                try await Task.sleep(nanoseconds: UInt64(timeoutSeconds * 1_000_000_000.0))
                return ""
            }

            // Take whichever finishes first; cancel the other.
            let first = try await group.next() ?? ""
            group.cancelAll()
            return first
        }
    }
    #endif

    @available(iOS 18.0, macOS 15.0, *)
    private func translateGoogleWithTimeout(
        _ text: String,
        apiKey: String,
        sourceLanguage: String?,
        targetLanguage: String,
        timeoutSeconds: Double
    ) async throws -> String {
        try await withThrowingTaskGroup(of: String.self) { group in
            group.addTask {
                try await GoogleCloudTranslation.translate(
                    text,
                    apiKey: apiKey,
                    sourceLanguage: sourceLanguage,
                    targetLanguage: targetLanguage
                )
            }
            group.addTask {
                try await Task.sleep(nanoseconds: UInt64(timeoutSeconds * 1_000_000_000.0))
                return ""
            }

            let first = try await group.next() ?? ""
            group.cancelAll()
            return first
        }
    }

    private struct MissingGoogleAPIKeyError: LocalizedError {
        var errorDescription: String? {
            "Missing Google API key. Set QWEN3_ASR_GOOGLE_TRANSLATE_API_KEY in the app environment."
        }
    }

    private func extractASRText(from raw: String) -> String {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "" }
        guard let range = trimmed.range(of: "<asr_text>") else { return trimmed }
        return String(trimmed[range.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func loadModelIfNeeded(modelId: String) async throws -> Qwen3ASRModel {
        if let model { return model }

        status = .loadingModel(progress: 0, status: "Starting")
        let loaded = try await Qwen3ASRModel.fromPretrained(modelId: modelId) { [weak self] progress, status in
            Task { @MainActor in
                self?.status = .loadingModel(progress: progress, status: status)
            }
        }

        model = loaded
        status = .ready
        return loaded
    }
}
