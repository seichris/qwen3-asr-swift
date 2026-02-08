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
    private var streamTask: Task<Void, Never>?
    private var lastFinalSegmentId: UUID?

    func clear() {
        partialTranscript = ""
        segments.removeAll()
        lastFinalSegmentId = nil
    }

    func stop() {
        streamTask?.cancel()
        streamTask = nil
        status = (model == nil) ? .idle : .ready
    }

    #if canImport(Translation)
    @available(iOS 18.0, macOS 15.0, *)
    func start(
        translationSession: TranslationSession,
        modelId: String,
        from: SupportedLanguage,
        to: SupportedLanguage
    ) {
        stop()
        status = .running

        streamTask = Task { [weak self] in
            guard let self else { return }
            do {
                let model = try await loadModelIfNeeded(modelId: modelId)

                let audioSource = MicrophoneAudioSource(frameSizeMs: 20)
                let options = RealtimeTranslationOptions(
                    targetLanguage: to.modelName,
                    sourceLanguage: from.modelName,
                    windowSeconds: 10.0,
                    stepMs: 500,
                    enableVAD: true,
                    enableTranslation: false
                )

                let stream = await model.realtimeTranslate(
                    audioSource: audioSource,
                    options: options,
                    translationSession: translationSession
                )

                for await event in stream {
                    if Task.isCancelled { break }
                    handle(event: event)
                }

                if !Task.isCancelled {
                    status = .ready
                }
            } catch {
                status = .error(String(describing: error))
            }
        }
    }
    #endif

    private func handle(event: RealtimeTranslationEvent) {
        switch event.kind {
        case .partial:
            partialTranscript = event.transcript

        case .final:
            partialTranscript = ""
            let cleaned = event.transcript.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !cleaned.isEmpty else { return }

            let id = UUID()
            lastFinalSegmentId = id
            segments.append(.init(id: id, transcript: cleaned, translation: nil, date: event.timestamp))

        case .translation:
            guard let t = event.translation?.trimmingCharacters(in: .whitespacesAndNewlines), !t.isEmpty else { return }
            guard let id = lastFinalSegmentId else { return }
            guard let idx = segments.lastIndex(where: { $0.id == id }) else { return }
            segments[idx].translation = t

        case .metrics:
            break
        }
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

