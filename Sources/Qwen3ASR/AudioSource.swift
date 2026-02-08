import Foundation
@preconcurrency import AVFoundation

/// Audio frame source protocol for realtime translation
public protocol AudioFrameSource: Actor {
    /// Stream of audio frames (mono float samples)
    func frames() -> AsyncStream<[Float]>
    /// Sample rate of the audio
    nonisolated var sampleRate: Int { get }
    /// Start capturing audio
    func start() async throws
    /// Stop capturing audio
    func stop() async
}

/// Microphone audio capture using AVAudioEngine
@available(macOS 13.0, iOS 17.0, *)
public actor MicrophoneAudioSource: AudioFrameSource {
    public let sampleRate: Int = 16000  // WhisperFeatureExtractor reference rate
    public let frameSize: Int  // Number of samples per frame (e.g., 320 for 20ms at 16kHz)
    public let bufferedFrames: Int
    
    private var audioEngine: AVAudioEngine?
    private var converter: AVAudioConverter?
    private var outputFormat: AVAudioFormat?
    private var continuation: AsyncStream<[Float]>.Continuation?
    private var isRunning = false
    private var pendingSamples: [Float] = []
    private var pendingStartIndex: Int = 0
    
    public init(frameSizeMs: Double = 20.0, bufferedFrames: Int = 50) {
        // Calculate frame size: sampleRate * frameSizeMs / 1000
        self.frameSize = Int(Double(sampleRate) * frameSizeMs / 1000.0)
        self.bufferedFrames = max(1, bufferedFrames)
    }
    
    public func frames() -> AsyncStream<[Float]> {
        AsyncStream(bufferingPolicy: .bufferingNewest(bufferedFrames)) { continuation in
            Task { [weak self] in
                await self?.setContinuation(continuation)
            }
        }
    }
    
    public func start() async throws {
        guard !isRunning else { return }

        #if os(iOS)
        try await configureAudioSession()
        #endif
        
        let audioEngine = AVAudioEngine()
        self.audioEngine = audioEngine
        
        let inputNode = audioEngine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)

        // Convert whatever the hardware provides into mono float32 @ 16kHz.
        // We must tap in the node's native format (or nil) to avoid format-mismatch exceptions.
        let outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(sampleRate),
            channels: 1,
            interleaved: false
        )!
        self.outputFormat = outputFormat
        self.converter = AVAudioConverter(from: inputFormat, to: outputFormat)

        // Capture conversion objects for the tap callback without touching actor state from that thread.
        let converter = self.converter
        let capturedOutputFormat = self.outputFormat
        
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: inputFormat) { [weak self] buffer, _ in
            guard let self else { return }

            // The PCM buffer is only valid for the duration of this callback.
            // Convert + copy samples synchronously, then hop onto the actor.
            let samples: [Float]

            if let converter, let outputFormat = capturedOutputFormat {
                let ratio = outputFormat.sampleRate / buffer.format.sampleRate
                let outCapacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio + 64)
                guard let out = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: outCapacity) else {
                    return
                }

                var error: NSError?
                var didProvideInput = false
                let status = converter.convert(to: out, error: &error) { _, outStatus in
                    if didProvideInput {
                        outStatus.pointee = .noDataNow
                        return nil
                    }
                    didProvideInput = true
                    outStatus.pointee = .haveData
                    return buffer
                }

                if status == .error {
                    if let error {
                        Qwen3ASRDebug.log("MicrophoneAudioSource: conversion failed: \(error)")
                    }
                    return
                }

                guard let channelData = out.floatChannelData?[0] else { return }
                let n = Int(out.frameLength)
                samples = Array(UnsafeBufferPointer(start: channelData, count: n))
            } else {
                // Fallback: best-effort float32 extraction without resampling/downmix.
                guard let channelData = buffer.floatChannelData?[0] else { return }
                let n = Int(buffer.frameLength)
                samples = Array(UnsafeBufferPointer(start: channelData, count: n))
            }

            Task { [weak self] in
                await self?.processAudioSamples(samples)
            }
        }
        
        // Prepare and start engine
        audioEngine.prepare()
        
        do {
            try audioEngine.start()
            isRunning = true
        } catch {
            throw RealtimeError.microphoneSetupFailed(error.localizedDescription)
        }
    }
    
    public func stop() async {
        guard isRunning else { return }
        
        audioEngine?.inputNode.removeTap(onBus: 0)
        audioEngine?.stop()
        audioEngine = nil
        converter = nil
        outputFormat = nil
        isRunning = false
        continuation?.finish()

        #if os(iOS)
        do {
            try AVAudioSession.sharedInstance().setActive(false)
        } catch {
            // Best-effort: don't surface stop-time audio session errors.
            Qwen3ASRDebug.log("MicrophoneAudioSource: AVAudioSession deactivation failed: \(error)")
        }
        #endif
    }

    private func setContinuation(_ continuation: AsyncStream<[Float]>.Continuation) {
        self.continuation = continuation
    }

    private func processAudioSamples(_ samples: [Float]) {
        guard let continuation else { return }
        pendingSamples.append(contentsOf: samples)

        // Emit fixed-size frames.
        while (pendingSamples.count - pendingStartIndex) >= frameSize {
            let start = pendingStartIndex
            let end = start + frameSize
            let frame = Array(pendingSamples[start..<end])
            pendingStartIndex = end
            continuation.yield(frame)
        }

        // Periodically compact to avoid unbounded growth.
        if pendingStartIndex > 8192 {
            pendingSamples.removeFirst(pendingStartIndex)
            pendingStartIndex = 0
        }
    }

    #if os(iOS)
    private func configureAudioSession() async throws {
        let session = AVAudioSession.sharedInstance()

        // Request permission early so start() fails with an actionable error.
        let granted = await withCheckedContinuation { (cont: CheckedContinuation<Bool, Never>) in
            session.requestRecordPermission { ok in
                cont.resume(returning: ok)
            }
        }
        guard granted else {
            throw RealtimeError.microphoneSetupFailed("Microphone permission denied (check NSMicrophoneUsageDescription in the host app).")
        }

        do {
            try session.setCategory(.playAndRecord, mode: .measurement, options: [.defaultToSpeaker, .allowBluetooth])
            try session.setPreferredSampleRate(Double(sampleRate))
            try session.setPreferredIOBufferDuration(0.02) // ~20ms
            try session.setActive(true)
        } catch {
            throw RealtimeError.microphoneSetupFailed("AVAudioSession setup failed: \(error)")
        }
    }
    #endif
}

/// Simulated audio source for testing (reads from file)
public actor FileAudioSource: AudioFrameSource {
    public let sampleRate: Int
    private let audioData: [Float]
    private let frameSize: Int
    private var currentPosition = 0
    private var isRunning = false
    private var continuation: AsyncStream<[Float]>.Continuation?
    private var playbackTask: Task<Void, Never>?
    private var startRequested: Bool = false
    
    public init(audioData: [Float], sampleRate: Int, frameSizeMs: Double = 20.0) {
        self.audioData = audioData
        self.sampleRate = sampleRate
        self.frameSize = Int(Double(sampleRate) * frameSizeMs / 1000.0)
    }
    
    public func frames() -> AsyncStream<[Float]> {
        AsyncStream(bufferingPolicy: .bufferingNewest(50)) { continuation in
            Task { [weak self] in
                await self?.setContinuation(continuation)
            }
        }
    }
    
    public func start() async throws {
        guard !isRunning else { return }
        isRunning = true
        currentPosition = 0
        startRequested = true
        maybeStartPlayback()
    }
    
    public func stop() async {
        isRunning = false
        playbackTask?.cancel()
        playbackTask = nil
        startRequested = false
        continuation?.finish()
    }

    private func setContinuation(_ continuation: AsyncStream<[Float]>.Continuation) {
        self.continuation = continuation
        maybeStartPlayback()
    }

    private func maybeStartPlayback() {
        guard startRequested, isRunning, continuation != nil else { return }
        guard playbackTask == nil else { return }
        playbackTask = Task { [weak self] in
            await self?.runPlayback()
        }
    }

    private func runPlayback() async {
        let frameDurationNs = UInt64((Double(frameSize) / Double(sampleRate)) * 1_000_000_000.0)

        while isRunning && currentPosition < audioData.count && !Task.isCancelled {
            guard let continuation else { break }
            let endPosition = min(currentPosition + frameSize, audioData.count)
            let frame = Array(audioData[currentPosition..<endPosition])
            currentPosition = endPosition
            continuation.yield(frame)

            if frameDurationNs > 0 {
                try? await Task.sleep(nanoseconds: frameDurationNs)
            }
        }

        continuation?.finish()
    }
}

/// Errors for realtime translation
public enum RealtimeError: Error, LocalizedError {
    case microphoneSetupFailed(String)
    case metalUnavailable
    
    public var errorDescription: String? {
        switch self {
        case .microphoneSetupFailed(let reason):
            return "Microphone setup failed: \(reason)"
        case .metalUnavailable:
            return "Metal is not available on this device"
        }
    }
}
