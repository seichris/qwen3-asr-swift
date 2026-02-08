import Foundation
import Qwen3ASR

#if canImport(Translation)
import Translation
#endif

@MainActor
final class LiveTranslateViewModel: ObservableObject {
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

    func clear() {
        partialTranscript = ""
        segments.removeAll()
    }

    func stop() {
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
        status = .running

        do {
            _ = to // currently used only for TranslationSession config in the view

            let model = try await loadModelIfNeeded(modelId: modelId)

            let audioSource = MicrophoneAudioSource(frameSizeMs: 20)
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
                await handle(event: event, translationSession: translationSession)
            }

            if !Task.isCancelled {
                status = .ready
            }
        } catch {
            if !Task.isCancelled {
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
                return
            }

        case .metrics:
            break

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
