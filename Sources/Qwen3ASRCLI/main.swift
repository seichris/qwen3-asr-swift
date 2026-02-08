import Foundation
import MLX
import Qwen3ASR
#if canImport(Metal)
import Metal
#endif

// MARK: - Realtime Command

private func runRealtime(
    targetLanguage: String,
    sourceLanguage: String?,
    modelId: String,
    windowSeconds: Double,
    stepMs: Int,
    enableVAD: Bool,
    enableTranslation: Bool,
    format: OutputFormat
) async throws {
    print("Loading model: \(modelId)")

    let model = try await Qwen3ASRModel.fromPretrained(modelId: modelId) { progress, status in
        print("  [\(Int(progress * 100))%] \(status)")
    }

    // Normalize CLI language identifiers (e.g. "en" -> "English", "cn" -> "Chinese").
    let normalizedTo = Qwen3ASRLanguage.normalize(targetLanguage)
    let normalizedFrom = Qwen3ASRLanguage.normalizeOptional(sourceLanguage)

    print("Starting realtime transcription...")
    print("Target language: \(normalizedTo)")
    print("Source language: \(normalizedFrom ?? "auto-detect")")
    print("Window: \(windowSeconds)s, Step: \(stepMs)ms")
    print("Press Ctrl+C to stop\n")

    let options = RealtimeTranslationOptions(
        targetLanguage: normalizedTo,
        sourceLanguage: normalizedFrom,
        windowSeconds: windowSeconds,
        stepMs: stepMs,
        enableVAD: enableVAD,
        enableTranslation: enableTranslation
    )

    let audioSource = MicrophoneAudioSource(frameSizeMs: 20)
    let stream = await model.realtimeTranslate(
        audioSource: audioSource,
        options: options
    )

    for await event in stream {
        switch format {
        case .plain:
            printPlain(event)
        case .jsonl:
            printJSONL(event)
        }
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
                i += 1
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
    print("  --jsonl                   Output in JSONL format")
    print("")
    print("Language values:")
    print("  You can pass common codes or names. The CLI normalizes them for the model prompt.")
    print("  Examples: en -> English, zh/cn -> Chinese, ja -> Japanese")
    print("")
    print("Environment variables:")
    print("  QWEN3_ASR_MODEL           Model ID (default: mlx-community/Qwen3-ASR-0.6B-4bit)")
    print("  QWEN3_ASR_DEVICE          Set to 'cpu' to force CPU mode")
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

        case .realtime(let targetLang, let sourceLang, let window, let step, let vad, let translate, let format):
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
