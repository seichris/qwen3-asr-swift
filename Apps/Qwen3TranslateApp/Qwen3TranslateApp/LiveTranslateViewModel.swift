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

    private var model: Qwen3ASRModel?
    private var activeAudioSource: MicrophoneAudioSource?
    private var activeRunID: UUID?

    func clear() {
        partialTranscript = ""
        segments.removeAll()
    }

    func stop() {
        let runID = activeRunID
        log.info("stop() requested. activeRunID=\(String(describing: runID))")
        activeRunID = nil
        if let src = activeAudioSource {
            Task {
                await src.stop()
            }
        }
        activeAudioSource = nil
        status = (model == nil) ? .idle : .ready
    }

    #if canImport(Translation)
    @available(iOS 18.0, macOS 15.0, *)
    func run(
        translationSession: TranslationSession,
        modelId: String,
        from: SupportedLanguage,
        to: SupportedLanguage
    ) async {
        let runID = UUID()
        activeRunID = runID
        status = .running

        do {
            log.info("run() start. runID=\(runID.uuidString) from=\(from.displayName, privacy: .public) to=\(to.displayName, privacy: .public)")

            let model = try await loadModelIfNeeded(modelId: modelId)

            let audioSource = MicrophoneAudioSource(frameSizeMs: 20)
            activeAudioSource = audioSource
            let options = RealtimeTranslationOptions(
                targetLanguage: "English",
                sourceLanguage: from.modelName,
                windowSeconds: 10.0,
                stepMs: 500,
                enableVAD: true,
                enableTranslation: false
            )

            let stream = await model.realtimeTranslate(
                audioSource: audioSource,
                options: options
            )

            for await event in stream {
                if Task.isCancelled { break }
                if activeRunID != runID { break }
                await handle(event: event, translationSession: translationSession)
            }

            if !Task.isCancelled {
                if activeRunID == runID {
                    activeRunID = nil
                    activeAudioSource = nil
                }
                log.info("run() finished. runID=\(runID.uuidString)")
                status = .ready
            }
        } catch {
            if !Task.isCancelled {
                log.error("run() error. runID=\(runID.uuidString) error=\(String(describing: error), privacy: .public)")
                status = .error(String(describing: error))
            }
        }
    }
    #endif

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

            if Task.isCancelled { return }
            do {
                let translated = try await AppleTranslation.translate(cleaned, using: translationSession)
                if Task.isCancelled { return }
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
    #endif

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
