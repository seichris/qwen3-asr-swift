import Foundation

#if canImport(Translation)
@preconcurrency import Translation

public extension Qwen3ASRModel {
    /// Create a realtime translator that uses Apple's Translation framework for `[TRANS]`.
    ///
    /// This wraps the core realtime stream and translates *final* segments using the provided
    /// `TranslationSession`, emitting `.translation` events.
    ///
    /// Notes:
    /// - `TranslationSession` availability is currently `macOS 15+` / `iOS 18+` in the SDK.
    /// - A `TranslationSession` is typically provided by a SwiftUI host via `.translationTask(...)`.
    @available(macOS 15.0, iOS 18.0, *)
    func realtimeTranslate(
        audioSource: any AudioFrameSource,
        options: RealtimeTranslationOptions,
        translationSession: TranslationSession
    ) async -> AsyncStream<RealtimeTranslationEvent> {
        var baseOptions = options
        // Disable the legacy model-based translation pass; we will emit translations via Apple Translation.
        baseOptions.enableTranslation = false

        let baseStream = await realtimeTranslate(audioSource: audioSource, options: baseOptions)

        return AsyncStream { continuation in
            let task = Task {
                for await event in baseStream {
                    if Task.isCancelled { break }
                    continuation.yield(event)

                    guard event.kind == .final else { continue }
                    let src = event.transcript.trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !src.isEmpty else { continue }

                    if Task.isCancelled { break }
                    do {
                        let translated = try await AppleTranslation.translate(src, using: translationSession)
                        if Task.isCancelled { break }
                        let cleaned = translated.trimmingCharacters(in: .whitespacesAndNewlines)
                        guard !cleaned.isEmpty else { continue }

                        continuation.yield(.init(
                            kind: .translation,
                            transcript: src,
                            translation: cleaned,
                            isStable: true
                        ))
                    } catch {
                        // Best-effort: skip translation failures and keep streaming ASR events.
                        continue
                    }
                }
                continuation.finish()
            }

            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }
}
#endif
