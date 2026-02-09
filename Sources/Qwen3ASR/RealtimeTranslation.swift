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
@available(macOS 13.0, iOS 17.0, *)
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

    // VAD-driven gating: avoid running ASR on silence (saves a lot of compute on iOS).
    private var inSpeech: Bool = false
    private var forceInference: Bool = false
    private var commitAfterInference: Bool = false
    
    // Track last emitted translation text so we can emit only the suffix.
    private var lastTranslatedText: String = ""

    // Count inferences for debugging.
    private var inferenceCount: Int = 0

    private static let realtimeMaxTokens: Int = {
        let env = ProcessInfo.processInfo.environment
        if let raw = env["QWEN3_ASR_REALTIME_MAX_TOKENS"]?.trimmingCharacters(in: .whitespacesAndNewlines),
           let n = Int(raw), n > 0 {
            return n
        }
        #if os(iOS)
        return 96
        #else
        return 200
        #endif
    }()

    private static let realtimeMaxAudioSeconds: Double? = {
        let env = ProcessInfo.processInfo.environment
        if let raw = env["QWEN3_ASR_REALTIME_MAX_AUDIO_SECONDS"]?.trimmingCharacters(in: .whitespacesAndNewlines),
           let s = Double(raw), s > 0 {
            return s
        }
        #if os(iOS)
        // Keep the decode window small enough for iPhone-class devices.
        return 4.0
        #else
        // Desktop can typically handle the full sliding window.
        return nil
        #endif
    }()
    
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

                    // IMPORTANT: Do not block frame ingestion while running inference.
                    // On slower devices (notably iPhone in Debug) inference can take > stepMs,
                    // and pausing the frame loop causes AsyncStream buffering to drop audio.
                    // That produces discontinuous windows and "gibberish" transcripts.
                    let ingestTask = Task {
                        for await frame in frames {
                            if Task.isCancelled { break }
                            await self.ingest(frame: frame, continuation: continuation)
                        }
                    }
                    let inferenceTask = Task {
                        while !Task.isCancelled {
                            // Small polling interval; actual cadence is governed by `shouldProcess()`.
                            try? await Task.sleep(nanoseconds: 50_000_000) // 50ms
                            if await self.shouldProcess() {
                                await self.processAudioBuffer(continuation: continuation)
                            }
                        }
                    }

                    // When audio ends (or stop() finishes the frames stream), stop scheduling inference.
                    _ = await ingestTask.result
                    inferenceTask.cancel()
                    _ = await inferenceTask.result
                    
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

    private func ingest(
        frame: [Float],
        continuation: AsyncStream<RealtimeTranslationEvent>.Continuation
    ) async {
        // Add to ring buffer (keeps newest samples).
        audioBuffer.append(contentsOf: frame)
        samplesSinceLastInference += frame.count

        // Process VAD
        if let vad = self.vad {
            let vadState = await vad.process(frame: frame)
            switch vadState {
            case .silence:
                inSpeech = false
            case .speech(_):
                inSpeech = true
            case .speechEnd(_):
                inSpeech = false
                // Run one more inference on the freshest window, then force-commit pending text.
                forceInference = true
                commitAfterInference = true
            }
        }
    }

    private func resetStreamingState(sampleRate: Int) async {
        self.sampleRate = sampleRate
        self.maxBufferSize = max(1, Int(options.windowSeconds * Double(sampleRate)))
        self.stepSamples = max(1, Int(Double(sampleRate) * Double(options.stepMs) / 1000.0))
        self.samplesSinceLastInference = 0
        self.isInferring = false
        self.inSpeech = false
        self.forceInference = false
        self.commitAfterInference = false
        self.audioBuffer = FloatRingBuffer(capacity: maxBufferSize)
        await stabilizer.reset()
    }
    
    /// Check if we should run transcription now
    private func shouldProcess() async -> Bool {
        if isInferring { return false }
        if audioBuffer.size < Int(0.5 * Double(sampleRate)) { return false } // need enough audio
        if forceInference { return true }
        if vad != nil && !inSpeech { return false }
        return samplesSinceLastInference >= stepSamples
    }
    
    /// Process current audio buffer and emit events
    private func processAudioBuffer(continuation: AsyncStream<RealtimeTranslationEvent>.Continuation) async {
        guard audioBuffer.size >= Int(0.5 * Double(sampleRate)) else { return }
        guard !isInferring else { return }
        isInferring = true
        defer { isInferring = false }
        let shouldCommitAfter = commitAfterInference
        commitAfterInference = false
        forceInference = false
        samplesSinceLastInference = 0
        inferenceCount += 1

        var audioSnapshot = audioBuffer.toArray()
        if let maxSeconds = Self.realtimeMaxAudioSeconds {
            let maxSamples = max(1, Int(Double(sampleRate) * maxSeconds))
            if audioSnapshot.count > maxSamples {
                audioSnapshot = Array(audioSnapshot.suffix(maxSamples))
            }
        }
        let t0 = DispatchTime.now().uptimeNanoseconds
        
        // Run transcription
        let rawTranscript = await transcribeAsync(
            audio: audioSnapshot,
            sampleRate: sampleRate,
            language: options.sourceLanguage,
            maxTokens: Self.realtimeMaxTokens
        )
        let t1 = DispatchTime.now().uptimeNanoseconds

        // Hard throttle: if we fell behind while inferring (iPhone Debug is common),
        // drop backlog instead of trying to "catch up" with back-to-back inference calls.
        // We still keep the ring buffer up-to-date, so the next inference uses fresh audio.
        if samplesSinceLastInference >= stepSamples {
            samplesSinceLastInference = 0
        }

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
            let ms = Double(t1 - t0) / 1_000_000.0
            let msStr = String(format: "%.2f", ms)
            let mem = Qwen3ASRRuntimeMetrics.residentMemoryMB().map(String.init) ?? ""
            Qwen3ASRDebug.log("RealtimeTranslator: inference=\(inferenceCount) inference_ms=\(msStr) audio_samples=\(audioSnapshot.count) ring=\(audioBuffer.size) mem_mb=\(mem)")
            continuation.yield(.init(
                kind: .metrics,
                transcript: "",
                translation: nil,
                isStable: true,
                metadata: [
                    "audio_samples": "\(audioSnapshot.count)",
                    "sample_rate": "\(sampleRate)",
                    "inference_ms": msStr,
                    "detected_language": parsed.language ?? "",
                    "resident_mb": mem,
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
                    audio: audioSnapshot,
                    continuation: continuation
                )
            }
        }

        // On speech end, force-commit any remaining pending text after one last inference.
        if shouldCommitAfter {
            let forced = await stabilizer.forceCommit()
            if !forced.newlyCommitted.isEmpty {
                continuation.yield(.init(
                    kind: .final,
                    transcript: forced.newlyCommitted,
                    isStable: true
                ))

                if options.enableTranslation {
                    await translateAndEmit(
                        text: forced.newlyCommitted,
                        audio: audioSnapshot,
                        continuation: continuation
                    )
                }
            }
        }

    }
    
    /// Translate text and emit event
    private func translateAndEmit(
        text: String,
        audio: [Float],
        continuation: AsyncStream<RealtimeTranslationEvent>.Continuation
    ) async {
        // Prefer "built-in" translation behavior by forcing output language during ASR decoding.
        // This avoids relying on text-only prompting/tokenization, which is not robust yet.
        let translatedRaw = await transcribeAsync(
            audio: audio,
            sampleRate: sampleRate,
            language: options.targetLanguage,
            maxTokens: Self.realtimeMaxTokens
        )

        let translatedFull = Self.parseASROutput(translatedRaw).text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !translatedFull.isEmpty else { return }

        let newlyTranslated = Self.computeNewSuffix(previous: lastTranslatedText, current: translatedFull)
        lastTranslatedText = translatedFull

        let cleanedNew = newlyTranslated.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleanedNew.isEmpty else { return }

        continuation.yield(.init(
            kind: .translation,
            transcript: text,
            translation: cleanedNew,
            isStable: true
        ))
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
                    audio: audioBuffer.toArray(),
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

    private static func computeNewSuffix(previous: String, current: String) -> String {
        if previous.isEmpty { return current }
        if current.hasPrefix(previous) {
            return String(current.dropFirst(previous.count))
        }

        // Find max overlap where suffix(previous) == prefix(current).
        let p = Array(previous)
        let c = Array(current)
        let maxLen = min(p.count, c.count)
        var overlap = 0
        for len in stride(from: maxLen, to: 0, by: -1) {
            let start = p.count - len
            var ok = true
            for i in 0..<len {
                if p[start + i] != c[i] {
                    ok = false
                    break
                }
            }
            if ok {
                overlap = len
                break
            }
        }
        return String(current.dropFirst(overlap))
    }

    private func transcribeAsync(
        audio: [Float],
        sampleRate: Int,
        language: String?,
        maxTokens: Int
    ) async -> String {
        let model = self.model
        return await withCheckedContinuation { cont in
            DispatchQueue.global(qos: .userInitiated).async {
                // On iOS, repeated inference can build up temporary allocations on background threads.
                // Keep them bounded so we don't spiral into Metal/driver issues.
                let out = autoreleasepool {
                    model.transcribe(
                        audio: audio,
                        sampleRate: sampleRate,
                        language: language,
                        maxTokens: maxTokens
                    )
                }
                cont.resume(returning: out)
            }
        }
    }
}

/// Convenience extension on Qwen3ASRModel
public extension Qwen3ASRModel {
    /// Create a realtime translator
    @available(macOS 13.0, iOS 17.0, *)
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
