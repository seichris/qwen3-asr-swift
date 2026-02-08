import Foundation

/// Options for realtime translation
public struct RealtimeTranslationOptions: Sendable {
    public var targetLanguage: String
    public var sourceLanguage: String?  // nil = auto-detect
    public var windowSeconds: Double
    public var stepMs: Int
    public var enableVAD: Bool
    public var enableTranslation: Bool
    
    public init(
        targetLanguage: String,
        sourceLanguage: String? = nil,
        windowSeconds: Double = 10.0,
        stepMs: Int = 500,
        enableVAD: Bool = true,
        enableTranslation: Bool = true
    ) {
        self.targetLanguage = targetLanguage
        self.sourceLanguage = sourceLanguage
        self.windowSeconds = windowSeconds
        self.stepMs = stepMs
        self.enableVAD = enableVAD
        self.enableTranslation = enableTranslation
    }
}

/// Event types for realtime translation
public struct RealtimeTranslationEvent: Sendable {
    public enum Kind: Sendable {
        case partial       // Partial/in-progress transcript
        case final         // Finalized transcript segment
        case translation   // Translated text
        case metrics       // Performance metrics
    }
    
    public var kind: Kind
    public var transcript: String
    public var translation: String?
    public var isStable: Bool
    public var timestamp: Date
    public var metadata: [String: String]?
    
    public init(
        kind: Kind,
        transcript: String,
        translation: String? = nil,
        isStable: Bool = false,
        timestamp: Date = Date(),
        metadata: [String: String]? = nil
    ) {
        self.kind = kind
        self.transcript = transcript
        self.translation = translation
        self.isStable = isStable
        self.timestamp = timestamp
        self.metadata = metadata
    }
}

/// Realtime translation engine
@available(macOS 13.0, *)
public actor RealtimeTranslator {
    private let model: Qwen3ASRModel
    private let options: RealtimeTranslationOptions
    private let stabilizer: TranscriptStabilizer
    private let vad: SimpleVAD?
    private var sampleRate: Int = 16000
    
    // Ring buffer for sliding window
    private var audioBuffer = FloatRingBuffer(capacity: 1)
    private var maxBufferSize: Int = 1
    private var stepSamples: Int = 1
    private var samplesSinceLastInference: Int = 0
    private var isInferring: Bool = false
    
    // Translation cache for finalized segments
    private var translationCache: [String: String] = [:]
    
    public init(
        model: Qwen3ASRModel,
        options: RealtimeTranslationOptions
    ) {
        self.model = model
        self.options = options
        self.stabilizer = TranscriptStabilizer(config: .init(
            stabilityThreshold: 2,
            minPrefixLength: 3
        ))
        self.vad = options.enableVAD ? SimpleVAD() : nil
    }
    
    /// Start realtime translation from an audio source
    public func translate(audioSource: any AudioFrameSource) -> AsyncStream<RealtimeTranslationEvent> {
        AsyncStream { continuation in
            continuation.onTermination = { _ in
                Task {
                    await audioSource.stop()
                }
            }
            Task {
                do {
                    let sourceSampleRate = audioSource.sampleRate
                    await self.resetStreamingState(sampleRate: sourceSampleRate)

                    // Create the stream before starting capture to avoid dropping early audio.
                    let frames = await audioSource.frames()
                    try await audioSource.start()
                    
                    // Process audio frames
                    for await frame in frames {
                        if Task.isCancelled { break }

                        // Add to ring buffer (keeps newest samples).
                        audioBuffer.append(contentsOf: frame)
                        samplesSinceLastInference += frame.count
                        
                        // Process VAD
                        if let vad = self.vad {
                            let vadState = await vad.process(frame: frame)
                            if case .speechEnd = vadState {
                                // Force commit on speech end
                                let forced = await self.stabilizer.forceCommit()
                                if !forced.newlyCommitted.isEmpty {
                                    continuation.yield(.init(
                                        kind: .final,
                                        transcript: forced.newlyCommitted,
                                        isStable: true
                                    ))

                                    if options.enableTranslation {
                                        await translateAndEmit(text: forced.newlyCommitted, continuation: continuation)
                                    }
                                }
                            }
                        }
                        
                        // Run transcription periodically
                        if self.shouldProcess() {
                            await self.processAudioBuffer(continuation: continuation)
                        }
                    }
                    
                    // Final processing
                    await self.finalize(continuation: continuation)
                    await audioSource.stop()
                    
                    continuation.finish()
                } catch {
                    continuation.yield(.init(
                        kind: .metrics,
                        transcript: "",
                        translation: nil,
                        isStable: true,
                        metadata: ["error": String(describing: error)]
                    ))
                    await audioSource.stop()
                    continuation.finish()
                }
            }
        }
    }

    private func resetStreamingState(sampleRate: Int) async {
        self.sampleRate = sampleRate
        self.maxBufferSize = max(1, Int(options.windowSeconds * Double(sampleRate)))
        self.stepSamples = max(1, Int(Double(sampleRate) * Double(options.stepMs) / 1000.0))
        self.samplesSinceLastInference = 0
        self.isInferring = false
        self.audioBuffer = FloatRingBuffer(capacity: maxBufferSize)
        await stabilizer.reset()
    }
    
    /// Check if we should run transcription now
    private func shouldProcess() -> Bool {
        if isInferring { return false }
        if audioBuffer.size < Int(0.5 * Double(sampleRate)) { return false } // need enough audio
        return samplesSinceLastInference >= stepSamples
    }
    
    /// Process current audio buffer and emit events
    private func processAudioBuffer(continuation: AsyncStream<RealtimeTranslationEvent>.Continuation) async {
        guard audioBuffer.size >= Int(0.5 * Double(sampleRate)) else { return }
        guard !isInferring else { return }
        isInferring = true
        defer { isInferring = false }
        samplesSinceLastInference = 0

        let audioSnapshot = audioBuffer.toArray()
        let t0 = DispatchTime.now().uptimeNanoseconds
        
        // Run transcription
        let rawTranscript = model.transcribe(
            audio: audioSnapshot,
            sampleRate: sampleRate,
            language: options.sourceLanguage,
            maxTokens: 200
        )
        let t1 = DispatchTime.now().uptimeNanoseconds

        let parsed = Self.parseASROutput(rawTranscript)
        let transcript = parsed.text
        
        // Update stabilizer
        let state = await stabilizer.update(transcript: transcript)
        
        // Emit partial event
        let event = RealtimeTranslationEvent(
            kind: .partial,
            transcript: state.committed + state.pending,
            isStable: state.isStable
        )
        continuation.yield(event)

        if Qwen3ASRDebug.enabled {
            continuation.yield(.init(
                kind: .metrics,
                transcript: "",
                translation: nil,
                isStable: true,
                metadata: [
                    "audio_samples": "\(audioSnapshot.count)",
                    "sample_rate": "\(sampleRate)",
                    "inference_ms": String(format: "%.2f", Double(t1 - t0) / 1_000_000.0),
                    "detected_language": parsed.language ?? "",
                ]
            ))
        }
        
        // Handle finalized text
        if state.isStable && !state.newlyCommitted.isEmpty {
            let finalEvent = RealtimeTranslationEvent(
                kind: .final,
                transcript: state.newlyCommitted,
                isStable: true
            )
            continuation.yield(finalEvent)
            
            // Translate if enabled
            if options.enableTranslation {
                await translateAndEmit(
                    text: state.newlyCommitted,
                    continuation: continuation
                )
            }
        }

    }
    
    /// Translate text and emit event
    private func translateAndEmit(
        text: String,
        continuation: AsyncStream<RealtimeTranslationEvent>.Continuation
    ) async {
        // Check cache
        if let cached = translationCache[text] {
            let event = RealtimeTranslationEvent(
                kind: .translation,
                transcript: text,
                translation: cached,
                isStable: true
            )
            continuation.yield(event)
            return
        }
        
        // Perform translation using text-only generation.
        // Keep the prompt single-line to avoid relying on explicit newline tokens.
        let prompt = "Translate to \(options.targetLanguage). Output only the translation. Text: \(text)"
        
        if let translation = model.generateTextOnly(
            prompt: prompt,
            maxTokens: 200
        ) {
            let cleaned = Self.parseASROutput(translation).text.trimmingCharacters(in: .whitespacesAndNewlines)
            // Cache the translation
            translationCache[text] = cleaned
            
            let event = RealtimeTranslationEvent(
                kind: .translation,
                transcript: text,
                translation: cleaned,
                isStable: true
            )
            continuation.yield(event)
        }
    }
    
    /// Finalize translation (commit any pending text)
    private func finalize(continuation: AsyncStream<RealtimeTranslationEvent>.Continuation) async {
        let finalState = await stabilizer.forceCommit()
        
        if !finalState.newlyCommitted.isEmpty {
            let event = RealtimeTranslationEvent(
                kind: .final,
                transcript: finalState.newlyCommitted,
                isStable: true
            )
            continuation.yield(event)
            
            if options.enableTranslation {
                await translateAndEmit(
                    text: finalState.newlyCommitted,
                    continuation: continuation
                )
            }
        }
    }
    
    /// Stop translation and cleanup
    public func stop() {
        // Cleanup is handled by task cancellation
    }

    private static func parseASROutput(_ raw: String) -> (language: String?, text: String) {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return (nil, "") }

        // Typical model output: "language Chinese<asr_text>你好。"
        if let range = trimmed.range(of: "<asr_text>") {
            let after = String(trimmed[range.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
            let before = String(trimmed[..<range.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)

            let lang: String?
            if before.hasPrefix("language ") {
                lang = String(before.dropFirst("language ".count)).trimmingCharacters(in: .whitespacesAndNewlines)
            } else {
                lang = nil
            }
            return (lang, after)
        }

        return (nil, trimmed)
    }
}

/// Convenience extension on Qwen3ASRModel
public extension Qwen3ASRModel {
    /// Create a realtime translator
    @available(macOS 13.0, *)
    func realtimeTranslate(
        audioSource: any AudioFrameSource,
        options: RealtimeTranslationOptions
    ) async -> AsyncStream<RealtimeTranslationEvent> {
        let translator = RealtimeTranslator(
            model: self,
            options: options
        )
        return await translator.translate(audioSource: audioSource)
    }
}
