import Foundation
import MLX
import Qwen3ASR
#if canImport(Metal)
import Metal
#endif

// MARK: - Realtime Command

private enum TranslationProvider: String {
    case auto
    case model
    case google
    case off
}

private actor RealtimeEventPrinter {
    let format: OutputFormat

    init(format: OutputFormat) {
        self.format = format
    }

    func emit(_ event: RealtimeTranslationEvent) {
        switch format {
        case .plain:
            printPlain(event)
        case .jsonl:
            printJSONL(event)
        }
    }
}

private actor GoogleTranslationThrottler {
    private let printer: RealtimeEventPrinter
    private let apiKey: String
    private let sourceLanguage: String?
    private let targetLanguage: String

    private var inFlight: Bool = false
    private var pendingText: String? = nil

    init(
        printer: RealtimeEventPrinter,
        apiKey: String,
        sourceLanguage: String?,
        targetLanguage: String
    ) {
        self.printer = printer
        self.apiKey = apiKey
        self.sourceLanguage = sourceLanguage
        self.targetLanguage = targetLanguage
    }

    func submit(_ text: String) {
        if inFlight {
            // If translation can't keep up, keep only the latest segment.
            pendingText = text
            return
        }
        inFlight = true

        let apiKey = self.apiKey
        let sourceLanguage = self.sourceLanguage
        let targetLanguage = self.targetLanguage

        Task(priority: .utility) { [weak self] in
            do {
                let translated = try await GoogleCloudTranslation.translate(
                    text,
                    apiKey: apiKey,
                    sourceLanguage: sourceLanguage,
                    targetLanguage: targetLanguage
                )
                let cleaned = translated.trimmingCharacters(in: .whitespacesAndNewlines)
                if !cleaned.isEmpty {
                    await self?.printer.emit(.init(
                        kind: .translation,
                        transcript: text,
                        translation: cleaned,
                        isStable: true
                    ))
                }
            } catch {
                await self?.printer.emit(.init(
                    kind: .metrics,
                    transcript: "",
                    translation: nil,
                    isStable: true,
                    metadata: ["error": "Google translation failed: \(String(describing: error))"]
                ))
            }
            await self?.finish()
        }
    }

    private func finish() {
        inFlight = false
        if let next = pendingText {
            pendingText = nil
            submit(next)
        }
    }
}

private func runRealtime(
    targetLanguage: String,
    sourceLanguage: String?,
    modelId: String,
    windowSeconds: Double,
    stepMs: Int,
    enableVAD: Bool,
    enableTranslation: Bool,
    translationProvider: TranslationProvider,
    format: OutputFormat
) async throws {
    print("Loading model: \(modelId)")

    let model = try await Qwen3ASRModel.fromPretrained(modelId: modelId) { progress, status in
        print("  [\(Int(progress * 100))%] \(status)")
    }

    // Normalize CLI language identifiers (e.g. "en" -> "English", "cn" -> "Chinese").
    let normalizedTo = Qwen3ASRLanguage.normalize(targetLanguage)
    let normalizedFrom = Qwen3ASRLanguage.normalizeOptional(sourceLanguage)
    let googleTo = Qwen3ASRLanguage.googleCode(targetLanguage)
    let googleFrom = Qwen3ASRLanguage.googleCodeOptional(sourceLanguage)

    let env = ProcessInfo.processInfo.environment
    let hasGoogleKey = (env["QWEN3_ASR_GOOGLE_TRANSLATE_API_KEY"]?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false)

    let provider: TranslationProvider = {
        switch translationProvider {
        case .auto:
            // Prefer Google when configured so translation doesn't consume MLX compute.
            return hasGoogleKey ? .google : .model
        case .model, .google, .off:
            return translationProvider
        }
    }()

    print("Starting realtime transcription...")
    print("Target language: \(normalizedTo)")
    print("Source language: \(normalizedFrom ?? "auto-detect")")
    print("Window: \(windowSeconds)s, Step: \(stepMs)ms")
    print("Press Ctrl+C to stop\n")

    let printer = RealtimeEventPrinter(format: format)

    #if DEBUG
    await printer.emit(.init(
        kind: .metrics,
        transcript: "",
        translation: nil,
        isStable: true,
        metadata: ["note": "Debug build is slower and can hurt realtime ASR accuracy. Prefer `swift run -c release qwen3-asr-cli ...` or `.build/release/qwen3-asr-cli ...`."]
    ))
    #endif

    let options = RealtimeTranslationOptions(
        targetLanguage: normalizedTo,
        sourceLanguage: normalizedFrom,
        windowSeconds: windowSeconds,
        stepMs: stepMs,
        enableVAD: enableVAD,
        enableTranslation: (enableTranslation && provider == .model)
    )

    let audioSource = MicrophoneAudioSource(frameSizeMs: 20)
    let stream = await model.realtimeTranslate(
        audioSource: audioSource,
        options: options
    )

    let googleTranslator: GoogleTranslationThrottler? = {
        guard enableTranslation, provider == .google else { return nil }
        guard let apiKey = env["QWEN3_ASR_GOOGLE_TRANSLATE_API_KEY"],
              !apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else { return nil }

        return GoogleTranslationThrottler(
            printer: printer,
            apiKey: apiKey,
            sourceLanguage: googleFrom,
            targetLanguage: googleTo
        )
    }()

    for await event in stream {
        await printer.emit(event)

        guard enableTranslation, provider == .google else { continue }
        guard event.kind == .final else { continue }

        let src = event.transcript.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !src.isEmpty else { continue }

        guard let googleTranslator else {
            await printer.emit(.init(
                kind: .metrics,
                transcript: "",
                translation: nil,
                isStable: true,
                metadata: ["error": "Missing QWEN3_ASR_GOOGLE_TRANSLATE_API_KEY (required for Google translation)."]
            ))
            continue
        }

        await googleTranslator.submit(src)
    }
}

private func printPlain(_ event: RealtimeTranslationEvent) {
    switch event.kind {
    case .partial:
        let stable = event.isStable ? "✓" : "…"
        print("[\(stable)] \(event.transcript)")
    case .final:
        print("[FINAL] \(event.transcript)")
    case .translation:
        if let translation = event.translation {
            print("[TRANS] \(translation)")
        }
    case .metrics:
        if let metadata = event.metadata {
            print("[METRICS] \(metadata)")
        }
    }
}

private func printJSONL(_ event: RealtimeTranslationEvent) {
    let json: [String: Any] = [
        "timestamp": ISO8601DateFormatter().string(from: event.timestamp),
        "kind": String(describing: event.kind),
        "transcript": event.transcript,
        "translation": event.translation as Any,
        "isStable": event.isStable
    ]

    if let data = try? JSONSerialization.data(withJSONObject: json),
       let jsonString = String(data: data, encoding: .utf8) {
        print(jsonString)
    }
}

enum OutputFormat {
    case plain
    case jsonl
}

// MARK: - Transcribe Command

private func runTranscribe(audioPath: String, modelId: String) async throws {
    print("Loading model: \(modelId)")

    let model = try await Qwen3ASRModel.fromPretrained(modelId: modelId) { progress, status in
        print("  [\(Int(progress * 100))%] \(status)")
    }

    print("Loading audio: \(audioPath)")
    let audio = try AudioFileLoader.load(url: URL(fileURLWithPath: audioPath), targetSampleRate: 24000)
    let seconds = Double(audio.count) / 24000.0
    print(String(format: "  Loaded %d samples (%.2fs)", audio.count, seconds))

    print("Transcribing...")
    let result = model.transcribe(
        audio: audio,
        sampleRate: 24000
    )
    print("Result: \(result)")
}

// MARK: - Metal Setup

private func requireMetalOrExit() {
    #if canImport(Metal)
    if MTLCreateSystemDefaultDevice() == nil {
        fputs("Error: Metal device is unavailable. MLX on macOS requires Metal.\n", stderr)
        fputs("If you are running inside a restricted sandbox/CI, run outside the sandbox or on a Mac with Metal enabled.\n", stderr)
        exit(1)
    }
    #endif
}

private func requireMLXMetallibOrExit() {
    guard let exeURL = Bundle.main.executableURL else { return }
    let binDir = exeURL.deletingLastPathComponent()

    let candidates: [URL] = [
        binDir.appendingPathComponent("mlx.metallib"),
        binDir.appendingPathComponent("default.metallib"),
        binDir.appendingPathComponent("Resources/mlx.metallib"),
        binDir.appendingPathComponent("Resources/default.metallib"),
    ]

    if candidates.contains(where: { FileManager.default.fileExists(atPath: $0.path) }) {
        return
    }

    fputs("Error: MLX Metal library (mlx.metallib) not found next to the executable.\n", stderr)
    fputs("Fix:\n", stderr)
    fputs("  1) Install Metal toolchain (once): xcodebuild -downloadComponent MetalToolchain\n", stderr)
    fputs("  2) From the repo root, run:\n", stderr)
    fputs("     swift build -c release --disable-sandbox\n", stderr)
    fputs("     ./scripts/build_mlx_metallib.sh release\n", stderr)
    exit(1)
}

// MARK: - Argument Parsing

private struct Arguments {
    let command: Command
    let modelId: String
    let device: String?

    enum Command {
        case transcribe(audioPath: String)
        case realtime(
            targetLanguage: String,
            sourceLanguage: String?,
            windowSeconds: Double,
            stepMs: Int,
            enableVAD: Bool,
            enableTranslation: Bool,
            translationProvider: TranslationProvider,
            format: OutputFormat
        )
    }
}

private func parseArguments() -> Arguments? {
    let args = CommandLine.arguments

    guard args.count >= 2 else {
        printUsage()
        return nil
    }

    let subcommand = args[1]
    let modelId = ProcessInfo.processInfo.environment["QWEN3_ASR_MODEL"]
        ?? "mlx-community/Qwen3-ASR-0.6B-4bit"
    let device = ProcessInfo.processInfo.environment["QWEN3_ASR_DEVICE"]

    switch subcommand {
    case "transcribe":
        guard args.count >= 3 else {
            print("Usage: qwen3-asr-cli transcribe <audio-file>")
            return nil
        }
        return Arguments(
            command: .transcribe(audioPath: args[2]),
            modelId: modelId,
            device: device
        )

    case "realtime":
        // Parse realtime options
        var targetLanguage: String = "en"
        var sourceLanguage: String? = nil
        var windowSeconds: Double = 10.0
        var stepMs: Int = 500
        var enableVAD: Bool = true
        var enableTranslation: Bool = true
        var translationProvider: TranslationProvider = .auto
        var format: OutputFormat = .plain

        var i = 2
        while i < args.count {
            switch args[i] {
            case "--to":
                if i + 1 < args.count {
                    targetLanguage = args[i + 1]
                    i += 2
                } else {
                    print("Error: --to requires a language (e.g. en, English, zh, Chinese)")
                    return nil
                }
            case "--from":
                if i + 1 < args.count {
                    let lang = args[i + 1]
                    sourceLanguage = lang == "auto" ? nil : lang
                    i += 2
                } else {
                    print("Error: --from requires a language (e.g. auto, zh, Chinese)")
                    return nil
                }
            case "--window":
                if i + 1 < args.count, let seconds = Double(args[i + 1]) {
                    windowSeconds = seconds
                    i += 2
                } else {
                    print("Error: --window requires a number")
                    return nil
                }
            case "--step":
                if i + 1 < args.count, let ms = Int(args[i + 1]) {
                    stepMs = ms
                    i += 2
                } else {
                    print("Error: --step requires a number")
                    return nil
                }
            case "--no-vad":
                enableVAD = false
                i += 1
            case "--no-translate":
                enableTranslation = false
                translationProvider = .off
                i += 1
            case "--translate-provider":
                if i + 1 < args.count {
                    let raw = args[i + 1].lowercased()
                    guard let p = TranslationProvider(rawValue: raw) else {
                        print("Error: --translate-provider must be one of: auto, model, google, off")
                        return nil
                    }
                    translationProvider = p
                    i += 2
                } else {
                    print("Error: --translate-provider requires a value (auto|model|google|off)")
                    return nil
                }
            case "--jsonl":
                format = .jsonl
                i += 1
            case "--help":
                printRealtimeHelp()
                return nil
            default:
                print("Unknown option: \(args[i])")
                printRealtimeHelp()
                return nil
            }
        }

        return Arguments(
            command: .realtime(
                targetLanguage: targetLanguage,
                sourceLanguage: sourceLanguage,
                windowSeconds: windowSeconds,
                stepMs: stepMs,
                enableVAD: enableVAD,
                enableTranslation: enableTranslation,
                translationProvider: translationProvider,
                format: format
            ),
            modelId: modelId,
            device: device
        )

    case "--help", "-h":
        printUsage()
        return nil

    default:
        // Legacy: single argument is treated as transcribe
        if args.count == 2 && !subcommand.hasPrefix("--") {
            return Arguments(
                command: .transcribe(audioPath: subcommand),
                modelId: modelId,
                device: device
            )
        }

        print("Unknown command: \(subcommand)")
        printUsage()
        return nil
    }
}

private func printUsage() {
    print("Usage: qwen3-asr-cli <command> [options]")
    print("")
    print("Commands:")
    print("  transcribe <audio-file>    Transcribe an audio file")
    print("  realtime                   Start realtime transcription from microphone")
    print("")
    print("Realtime options:")
    print("  --to <lang>               Target translation language (default: en)")
    print("  --from <lang>             Source language hint (default: auto)")
    print("  --window <seconds>        Sliding window size (default: 10)")
    print("  --step <milliseconds>     Update interval (default: 500)")
    print("  --no-vad                  Disable voice activity detection")
    print("  --no-translate            Disable translation (transcription only)")
    print("  --translate-provider <p>  Translation provider: auto|model|google|off (default: auto)")
    print("  --jsonl                   Output in JSONL format")
    print("")
    print("Language values:")
    print("  You can pass common codes or names. The CLI normalizes them for the model prompt.")
    print("  Examples: en -> English, zh/cn -> Chinese, ja -> Japanese")
    print("")
    print("Environment variables:")
    print("  QWEN3_ASR_MODEL           Model ID (default: mlx-community/Qwen3-ASR-0.6B-4bit)")
    print("  QWEN3_ASR_DEVICE          Set to 'cpu' to force CPU mode")
    print("  QWEN3_ASR_GOOGLE_TRANSLATE_API_KEY  Google Translate API key (used when --translate-provider google, or auto)")
    print("")
    print("Examples:")
    print("  qwen3-asr-cli transcribe audio.wav")
    print("  qwen3-asr-cli realtime --to en --from auto")
    print("  qwen3-asr-cli realtime --to ja --window 15 --jsonl")
}

private func printRealtimeHelp() {
    print("Usage: qwen3-asr-cli realtime [options]")
    print("")
    print("Options:")
    print("  --to <lang>      Target translation language (required)")
    print("  --from <lang>    Source language hint (default: auto)")
    print("  --window <sec>   Audio window size in seconds (default: 10)")
    print("  --step <ms>      Update interval in milliseconds (default: 500)")
    print("  --no-vad         Disable voice activity detection")
    print("  --no-translate   Transcribe only, no translation")
    print("  --translate-provider <p>  auto|model|google|off (default: auto)")
    print("  --jsonl          Output structured JSONL")
    print("")
    print("Language values:")
    print("  Examples: --to en, --to English, --from zh, --from Chinese")
}

// MARK: - Entry Point

Task {
    requireMetalOrExit()
    requireMLXMetallibOrExit()

    guard let args = parseArguments() else {
        exit(1)
    }

    do {
        switch args.command {
        case .transcribe(let audioPath):
            if args.device == "cpu" {
                try await Device.withDefaultDevice(.cpu) {
                    try await runTranscribe(audioPath: audioPath, modelId: args.modelId)
                }
            } else {
                try await runTranscribe(audioPath: audioPath, modelId: args.modelId)
            }

        case .realtime(let targetLang, let sourceLang, let window, let step, let vad, let translate, let provider, let format):
            if args.device == "cpu" {
                try await Device.withDefaultDevice(.cpu) {
                    try await runRealtime(
                        targetLanguage: targetLang,
                        sourceLanguage: sourceLang,
                        modelId: args.modelId,
                        windowSeconds: window,
                        stepMs: step,
                        enableVAD: vad,
                        enableTranslation: translate,
                        translationProvider: provider,
                        format: format
                    )
                }
            } else {
                try await runRealtime(
                    targetLanguage: targetLang,
                    sourceLanguage: sourceLang,
                    modelId: args.modelId,
                    windowSeconds: window,
                    stepMs: step,
                    enableVAD: vad,
                    enableTranslation: translate,
                    translationProvider: provider,
                    format: format
                )
            }
        }

        exit(0)
    } catch {
        print("Error: \(error)")
        exit(1)
    }
}

// Keep the main thread alive
RunLoop.main.run()
