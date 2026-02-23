import Foundation
import Qwen3ASR
import os
#if os(iOS)
@preconcurrency import AVFoundation
#if canImport(ReplayKit)
import ReplayKit
#endif
#endif

#if canImport(Translation)
import Translation
#endif

enum DashScopeWorkspace: String, CaseIterable, Identifiable, Equatable {
    case mainland
    case singapore

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .mainland: return "DashScope Mainland"
        case .singapore: return "DashScope Singapore"
        }
    }

    var apiKeyEnvironmentVariable: String {
        switch self {
        case .mainland: return DashScopeRealtimeClient.apiKeyEnvironmentVariable
        case .singapore: return "DASHSCOPE_API_KEY_SG"
        }
    }

    var websocketBaseURL: URL {
        switch self {
        case .mainland:
            return URL(string: "wss://dashscope.aliyuncs.com/api-ws/v1/realtime")!
        case .singapore:
            return URL(string: "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime")!
        }
    }
}

enum RealtimeInputAudioSource: String, CaseIterable, Identifiable, Equatable {
    case microphone
    case deviceAudio

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .microphone:
            return "Microphone"
        case .deviceAudio:
            return "Device Audio"
        }
    }
}

#if os(iOS) && canImport(ReplayKit)
@available(iOS 17.0, *)
actor ReplayKitDeviceAudioSource: AudioFrameSource {
    nonisolated let sampleRate: Int = 16000
    let frameSize: Int
    let bufferedFrames: Int

    private let recorder = RPScreenRecorder.shared()
    private var continuation: AsyncStream<[Float]>.Continuation?
    private var isRunning = false
    private var pendingSamples: [Float] = []
    private var pendingStartIndex: Int = 0
    private var converter: AVAudioConverter?
    private var converterInputFormat: AVAudioFormat?
    private let outputFormat: AVAudioFormat

    init(frameSizeMs: Double = 20.0, bufferedFrames: Int = 500) {
        self.frameSize = Int(Double(sampleRate) * frameSizeMs / 1000.0)
        self.bufferedFrames = max(1, bufferedFrames)
        self.outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(sampleRate),
            channels: 1,
            interleaved: false
        )!
    }

    func frames() -> AsyncStream<[Float]> {
        AsyncStream(bufferingPolicy: .bufferingNewest(bufferedFrames)) { continuation in
            Task { [weak self] in
                await self?.setContinuation(continuation)
            }
        }
    }

    func start() async throws {
        guard !isRunning else { return }

        if !recorder.isAvailable {
            throw RealtimeError.microphoneSetupFailed("Device audio capture is unavailable on this device.")
        }

        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            recorder.startCapture(
                handler: { [weak self] sampleBuffer, sampleType, error in
                    if let error {
                        NSLog("ReplayKitDeviceAudioSource capture callback error: %@", error.localizedDescription)
                        return
                    }
                    guard sampleType == .audioApp else { return }
                    Task { [weak self] in
                        await self?.process(sampleBuffer: sampleBuffer)
                    }
                },
                completionHandler: { error in
                    if let error {
                        cont.resume(throwing: RealtimeError.microphoneSetupFailed("ReplayKit startCapture failed: \(error.localizedDescription)"))
                        return
                    }
                    cont.resume(returning: ())
                }
            )
        }

        isRunning = true
    }

    func stop() async {
        guard isRunning else { return }
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            recorder.stopCapture { _ in
                cont.resume(returning: ())
            }
        }
        isRunning = false
        converter = nil
        converterInputFormat = nil
        continuation?.finish()
    }

    private func setContinuation(_ continuation: AsyncStream<[Float]>.Continuation) {
        self.continuation = continuation
    }

        private func process(sampleBuffer: CMSampleBuffer) {
            guard let continuation else { return }
            guard let formatDesc = CMSampleBufferGetFormatDescription(sampleBuffer) else { return }
            let inputFormat = AVAudioFormat(cmAudioFormatDescription: formatDesc)

        let sampleCount = CMSampleBufferGetNumSamples(sampleBuffer)
        guard sampleCount > 0 else { return }
        guard let inBuffer = AVAudioPCMBuffer(
            pcmFormat: inputFormat,
            frameCapacity: AVAudioFrameCount(sampleCount)
        ) else { return }
        inBuffer.frameLength = AVAudioFrameCount(sampleCount)

        let copyStatus = CMSampleBufferCopyPCMDataIntoAudioBufferList(
            sampleBuffer,
            at: 0,
            frameCount: Int32(sampleCount),
            into: inBuffer.mutableAudioBufferList
        )
        guard copyStatus == noErr else { return }

        if converter == nil || needsConverterReset(for: inputFormat) {
            converter = AVAudioConverter(from: inputFormat, to: outputFormat)
            converterInputFormat = inputFormat
        }
        guard let converter else { return }

        let ratio = outputFormat.sampleRate / inputFormat.sampleRate
        let outCapacity = max(1, Int(Double(sampleCount) * ratio + 64))
        guard let outBuffer = AVAudioPCMBuffer(
            pcmFormat: outputFormat,
            frameCapacity: AVAudioFrameCount(outCapacity)
        ) else { return }

        var error: NSError?
        var didProvideInput = false
        let status = converter.convert(to: outBuffer, error: &error) { _, outStatus in
            if didProvideInput {
                outStatus.pointee = .noDataNow
                return nil
            }
            didProvideInput = true
            outStatus.pointee = .haveData
            return inBuffer
        }
        if status == .error {
            if let error {
                NSLog("ReplayKitDeviceAudioSource conversion failed: %@", error.localizedDescription)
            }
            return
        }

        guard let channelData = outBuffer.floatChannelData?[0] else { return }
        let n = Int(outBuffer.frameLength)
        pendingSamples.append(contentsOf: UnsafeBufferPointer(start: channelData, count: n))

        while (pendingSamples.count - pendingStartIndex) >= frameSize {
            let start = pendingStartIndex
            let end = start + frameSize
            let frame = Array(pendingSamples[start..<end])
            pendingStartIndex = end
            continuation.yield(frame)
        }

        if pendingStartIndex > 8192 {
            pendingSamples.removeFirst(pendingStartIndex)
            pendingStartIndex = 0
        }
    }

    private func needsConverterReset(for inputFormat: AVAudioFormat) -> Bool {
        guard let current = converterInputFormat else { return true }
        if current.sampleRate != inputFormat.sampleRate { return true }
        if current.channelCount != inputFormat.channelCount { return true }
        if current.commonFormat != inputFormat.commonFormat { return true }
        if current.isInterleaved != inputFormat.isInterleaved { return true }
        return false
    }
}
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
    private var activeAudioSource: (any AudioFrameSource)?
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
                        let env = ProcessInfo.processInfo.environment
                        let maxTokens: Int = {
                            if let raw = env["QWEN3_ASR_FILE_MAX_TOKENS"]?.trimmingCharacters(in: .whitespacesAndNewlines),
                               let n = Int(raw), n > 0 {
                                return min(max(n, 32), 2048)
                            }
                            #if os(iOS)
                            // Keep debug file transcription modest on iPhone to reduce jetsam risk.
                            return 256
                            #else
                            return 448
                            #endif
                        }()
                        let text = model.transcribe(
                            audio: samples,
                            sampleRate: 24000,
                            language: from.modelNameOptional,
                            maxTokens: maxTokens
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
    func runNoTranslation(
        modelId: String,
        from: SupportedLanguage,
        inputSource: RealtimeInputAudioSource
    ) async {
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

            let audioSource = try makeRealtimeAudioSource(inputSource: inputSource)
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
                sourceLanguage: from.modelNameOptional,
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
    func runGoogleTranslation(
        modelId: String,
        from: SupportedLanguage,
        to: SupportedLanguage,
        inputSource: RealtimeInputAudioSource
    ) async {
        status = .running

        do {
            if stopRequested {
                isStopping = false
                isRunning = false
                status = (model == nil) ? .idle : .ready
                return
            }

            guard let apiKey = credentialValue(for: "QWEN3_ASR_GOOGLE_TRANSLATE_API_KEY"),
                  !apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            else {
                throw MissingGoogleAPIKeyError()
            }

            log.info("runGoogleTranslation() start. from=\(from.displayName, privacy: .public) to=\(to.displayName, privacy: .public)")

            let model = try await loadModelIfNeeded(modelId: modelId)

            let audioSource = try makeRealtimeAudioSource(inputSource: inputSource)
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
                sourceLanguage: from.modelNameOptional,
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

    @available(iOS 18.0, macOS 15.0, *)
    func runDashScopeHosted(
        from: SupportedLanguage,
        workspace: DashScopeWorkspace,
        inputSource: RealtimeInputAudioSource
    ) async {
        status = .running

        do {
            if stopRequested {
                isStopping = false
                isRunning = false
                status = (model == nil) ? .idle : .ready
                return
            }

            let (client, modelName) = try makeDashScopeClient(from: from, workspace: workspace)

            log.info("runDashScopeHosted() start. workspace=\(workspace.rawValue, privacy: .public), from=\(from.displayName, privacy: .public), model=\(modelName, privacy: .public)")

            let audioSource = try makeRealtimeAudioSource(inputSource: inputSource)
            activeAudioSource = audioSource
            defer {
                activeAudioSource = nil
                isStopping = false
                isRunning = false
                if case .running = status {
                    status = (self.model == nil) ? .idle : .ready
                }
            }

            let stream = await client.transcribe(audioSource: audioSource)
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
                log.info("runDashScopeHosted() finished")
            }
        } catch {
            if !Task.isCancelled {
                log.error("runDashScopeHosted() error. error=\(String(describing: error), privacy: .public)")
                activeAudioSource = nil
                isStopping = false
                isRunning = false
                status = .error(String(describing: error))
            }
        }
    }

    @available(iOS 18.0, macOS 15.0, *)
    func runDashScopeHostedGoogle(
        from: SupportedLanguage,
        to: SupportedLanguage,
        workspace: DashScopeWorkspace,
        inputSource: RealtimeInputAudioSource
    ) async {
        status = .running

        do {
            if stopRequested {
                isStopping = false
                isRunning = false
                status = (model == nil) ? .idle : .ready
                return
            }

            guard let googleAPIKey = credentialValue(for: "QWEN3_ASR_GOOGLE_TRANSLATE_API_KEY"),
                  !googleAPIKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            else {
                throw MissingGoogleAPIKeyError()
            }

            let (client, modelName) = try makeDashScopeClient(from: from, workspace: workspace)
            log.info("runDashScopeHostedGoogle() start. workspace=\(workspace.rawValue, privacy: .public), from=\(from.displayName, privacy: .public) to=\(to.displayName, privacy: .public), model=\(modelName, privacy: .public)")
            log.info("runDashScopeHostedGoogle() google key configured: yes")

            let audioSource = try makeRealtimeAudioSource(inputSource: inputSource)
            activeAudioSource = audioSource
            defer {
                activeAudioSource = nil
                isStopping = false
                isRunning = false
                if case .running = status {
                    status = (self.model == nil) ? .idle : .ready
                }
            }

            let stream = await client.transcribe(audioSource: audioSource)
            for await event in stream {
                if Task.isCancelled { break }
                if stopRequested { break }
                await handleGoogle(event: event, apiKey: googleAPIKey, from: from, to: to)
            }

            if !Task.isCancelled {
                activeAudioSource = nil
                isStopping = false
                isRunning = false
                status = .ready
                log.info("runDashScopeHostedGoogle() finished")
            }
        } catch {
            if !Task.isCancelled {
                log.error("runDashScopeHostedGoogle() error. error=\(String(describing: error), privacy: .public)")
                activeAudioSource = nil
                isStopping = false
                isRunning = false
                status = .error(String(describing: error))
            }
        }
    }

    #if canImport(Translation)
    @available(iOS 18.0, macOS 15.0, *)
    func runDashScopeHosted(
        translationSession: TranslationSession,
        from: SupportedLanguage,
        to: SupportedLanguage,
        workspace: DashScopeWorkspace,
        inputSource: RealtimeInputAudioSource
    ) async {
        status = .running

        do {
            if stopRequested {
                isStopping = false
                isRunning = false
                status = (model == nil) ? .idle : .ready
                return
            }

            let (client, modelName) = try makeDashScopeClient(from: from, workspace: workspace)
            log.info("runDashScopeHosted(apple) start. workspace=\(workspace.rawValue, privacy: .public), from=\(from.displayName, privacy: .public) to=\(to.displayName, privacy: .public), model=\(modelName, privacy: .public)")

            let audioSource = try makeRealtimeAudioSource(inputSource: inputSource)
            activeAudioSource = audioSource
            defer {
                activeAudioSource = nil
                isStopping = false
                isRunning = false
                if case .running = status {
                    status = (self.model == nil) ? .idle : .ready
                }
            }

            let stream = await client.transcribe(audioSource: audioSource)
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
                log.info("runDashScopeHosted(apple) finished")
            }
        } catch {
            if !Task.isCancelled {
                log.error("runDashScopeHosted(apple) error. error=\(String(describing: error), privacy: .public)")
                activeAudioSource = nil
                isStopping = false
                isRunning = false
                status = .error(String(describing: error))
            }
        }
    }

    @available(iOS 18.0, macOS 15.0, *)
    func run(
        translationSession: TranslationSession,
        modelId: String,
        from: SupportedLanguage,
        to: SupportedLanguage,
        inputSource: RealtimeInputAudioSource
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

            let audioSource = try makeRealtimeAudioSource(inputSource: inputSource)
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
                sourceLanguage: from.modelNameOptional,
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
            guard !cleaned.isEmpty else {
                log.debug("handleGoogle() skipped empty final transcript")
                return
            }

            let id = UUID()
            segments.append(.init(id: id, transcript: cleaned, translation: nil, date: event.timestamp))

            if stopRequested {
                log.debug("handleGoogle() stop requested before translation")
                return
            }
            if Task.isCancelled {
                log.debug("handleGoogle() task cancelled before translation")
                return
            }

            let googleSource = Qwen3ASRLanguage.googleCodeOptional((from.id == "auto") ? nil : from.id)
            let googleTarget = Qwen3ASRLanguage.googleCode(to.id)
            let googleTimeoutSeconds = googleTranslationTimeoutSeconds()
            log.info("handleGoogle() translate start. source=\(googleSource ?? "auto", privacy: .public), target=\(googleTarget, privacy: .public), chars=\(cleaned.count), timeout=\(googleTimeoutSeconds)")

            do {
                let translated = try await translateGoogleWithTimeout(
                    cleaned,
                    apiKey: apiKey,
                    sourceLanguage: googleSource,
                    targetLanguage: googleTarget,
                    timeoutSeconds: googleTimeoutSeconds
                )
                if Task.isCancelled { return }
                if stopRequested { return }
                let t = translated.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !t.isEmpty else {
                    log.debug("handleGoogle() translation empty after trim")
                    return
                }
                guard let idx = segments.lastIndex(where: { $0.id == id }) else { return }
                segments[idx].translation = t
                log.info("handleGoogle() translate success. chars=\(t.count)")
            } catch let firstError {
                var finalError: Error = firstError
                log.error("handleGoogle() translate failed first attempt: \(String(describing: firstError), privacy: .public)")
                if googleSource != nil {
                    log.info("handleGoogle() retrying with source=auto")
                    do {
                        let translated = try await translateGoogleWithTimeout(
                            cleaned,
                            apiKey: apiKey,
                            sourceLanguage: nil,
                            targetLanguage: googleTarget,
                            timeoutSeconds: googleTimeoutSeconds
                        )
                        if Task.isCancelled { return }
                        if stopRequested { return }
                        let t = translated.trimmingCharacters(in: .whitespacesAndNewlines)
                        guard !t.isEmpty else {
                            log.debug("handleGoogle() retry translation empty after trim")
                            return
                        }
                        guard let idx = segments.lastIndex(where: { $0.id == id }) else { return }
                        segments[idx].translation = t
                        log.info("handleGoogle() translate success on retry. chars=\(t.count)")
                        return
                    } catch let retryError {
                        finalError = retryError
                        log.error("google translation retry failed: \(String(describing: retryError), privacy: .public)")
                    }
                }

                // Best-effort: keep transcription running if translation fails/cancels.
                let message = googleFailureMessage(from: finalError)
                status = .error(message)
                log.error("google translation failed: \(message, privacy: .public)")
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
            if first.isEmpty {
                throw GoogleTranslationTimeoutError(timeoutSeconds: timeoutSeconds)
            }
            return first
        }
    }

    private func googleTranslationTimeoutSeconds() -> Double {
        let key = "QWEN3_ASR_GOOGLE_TIMEOUT_SECONDS"
        if let raw = ProcessInfo.processInfo.environment[key]?.trimmingCharacters(in: .whitespacesAndNewlines),
           let value = Double(raw),
           value > 0 {
            return value
        }
        if let raw = UserDefaults.standard.string(forKey: key)?.trimmingCharacters(in: .whitespacesAndNewlines),
           let value = Double(raw),
           value > 0 {
            return value
        }
        return 12.0
    }

    private func googleFailureMessage(from error: Error) -> String {
        if let timeout = error as? GoogleTranslationTimeoutError {
            return "Google Translate timed out after \(timeout.timeoutSeconds)s. Network may block Google APIs; VPN often helps."
        }

        if let urlError = error as? URLError {
            return "Google Translate network error (\(urlError.code.rawValue)): \(urlError.localizedDescription)"
        }

        if let translationError = error as? GoogleCloudTranslation.TranslationError {
            switch translationError {
            case .missingAPIKey:
                return "Google Translate API key is missing."
            case .invalidHTTPResponse:
                return "Google Translate returned an invalid HTTP response."
            case .httpStatus(let code, _):
                return "Google Translate HTTP \(code). Check API key, billing, and network."
            case .emptyTranslation:
                return "Google Translate returned empty output."
            }
        }

        return "Google Translate failed: \(String(describing: error))"
    }

    private func makeRealtimeAudioSource(inputSource: RealtimeInputAudioSource) throws -> any AudioFrameSource {
        switch inputSource {
        case .microphone:
            return MicrophoneAudioSource(frameSizeMs: 20, bufferedFrames: 500)
        case .deviceAudio:
            #if os(iOS) && canImport(ReplayKit)
            return ReplayKitDeviceAudioSource(frameSizeMs: 20, bufferedFrames: 500)
            #else
            throw UnsupportedInputSourceError(inputSource: inputSource)
            #endif
        }
    }

    private func makeDashScopeClient(
        from: SupportedLanguage,
        workspace: DashScopeWorkspace
    ) throws -> (DashScopeRealtimeClient, String) {
        let keyName = workspace.apiKeyEnvironmentVariable
        guard let apiKey = credentialValue(for: keyName),
              !apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw MissingDashScopeAPIKeyError(keyName: keyName)
        }

        let hostedModel = ProcessInfo.processInfo.environment["DASHSCOPE_REALTIME_MODEL"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let modelName = (hostedModel?.isEmpty == false)
            ? hostedModel!
            : DashScopeRealtimeOptions.defaultModel

        let options = DashScopeRealtimeOptions(
            model: modelName,
            language: (from.id == "auto") ? nil : from.id,
            sampleRate: 16000,
            enableServerVAD: true,
            websocketBaseURL: workspace.websocketBaseURL
        )
        return (DashScopeRealtimeClient(apiKey: apiKey, options: options), modelName)
    }

    private func credentialValue(for key: String) -> String? {
        let envValue = ProcessInfo.processInfo.environment[key]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if let envValue, !envValue.isEmpty {
            return envValue
        }

        let storedValue = UserDefaults.standard.string(forKey: key)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if let storedValue, !storedValue.isEmpty {
            return storedValue
        }

        return nil
    }

    private struct MissingGoogleAPIKeyError: LocalizedError {
        var errorDescription: String? {
            "Missing Google API key. Set QWEN3_ASR_GOOGLE_TRANSLATE_API_KEY in Xcode Run environment or in the app Settings."
        }
    }

    private struct MissingDashScopeAPIKeyError: LocalizedError {
        let keyName: String

        var errorDescription: String? {
            "Missing DashScope API key. Set \(keyName) in Xcode Run environment or in the app Settings."
        }
    }

    private struct GoogleTranslationTimeoutError: LocalizedError {
        let timeoutSeconds: Double

        var errorDescription: String? {
            "Google translation timed out after \(timeoutSeconds)s."
        }
    }

    private struct UnsupportedInputSourceError: LocalizedError {
        let inputSource: RealtimeInputAudioSource

        var errorDescription: String? {
            "Input source \(inputSource.displayName) is unavailable on this platform."
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
