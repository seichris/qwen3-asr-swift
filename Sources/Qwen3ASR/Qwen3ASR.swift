import Foundation
import Dispatch
import MLX
import MLXNN
import MLXFast

/// Special token IDs for Qwen3-ASR
public struct Qwen3ASRTokens {
    public static let audioTokenId = 151676        // <|audio_pad|>
    public static let audioStartTokenId = 151669   // <|audio_start|>
    public static let audioEndTokenId = 151670     // <|audio_end|>
    public static let eosTokenId = 151645          // <|im_end|>
    public static let padTokenId = 151643          // <|endoftext|>
    public static let imStartTokenId = 151644      // <|im_start|>
    public static let imEndTokenId = 151645        // <|im_end|>
}

/// Main Qwen3-ASR model for speech recognition
public class Qwen3ASRModel {
    public let audioEncoder: Qwen3AudioEncoder
    public let featureExtractor: WhisperFeatureExtractor
    public var textDecoder: QuantizedTextModel?

    /// Tokenizer for decoding output tokens
    private var tokenizer: Qwen3Tokenizer?

    /// Text decoder config
    public let textConfig: TextDecoderConfig

    public init(
        audioConfig: Qwen3AudioEncoderConfig = .default,
        textConfig: TextDecoderConfig = .small
    ) {
        self.audioEncoder = Qwen3AudioEncoder(config: audioConfig)
        self.featureExtractor = WhisperFeatureExtractor()
        self.textConfig = textConfig
        // Text decoder will be initialized when loading weights
        self.textDecoder = nil
    }

    /// Set tokenizer for text decoding
    public func setTokenizer(_ tokenizer: Qwen3Tokenizer) {
        self.tokenizer = tokenizer
    }

    /// Initialize text decoder (called after loading)
    public func initializeTextDecoder() {
        self.textDecoder = QuantizedTextModel(config: textConfig)
    }

    /// Transcribe audio to text
    /// - Parameters:
    ///   - audio: Float audio samples
    ///   - sampleRate: Sample rate of input audio (default 24000)
    ///   - language: Target output language (nil = auto-detect and transcribe in source language)
    ///   - maxTokens: Maximum tokens to generate
    public func transcribe(
        audio: [Float],
        sampleRate: Int = 24000,
        language: String? = nil,
        maxTokens: Int = 448
    ) -> String {
        // Extract mel features
        let melFeatures = featureExtractor.process(audio, sampleRate: sampleRate)

        if Qwen3ASRDebug.enabled {
            Qwen3ASRDebug.log("Mel features shape: \(melFeatures.shape)")
            if Qwen3ASRDebug.tensorStatsEnabled {
                let melFlat = melFeatures.flattened()
                Qwen3ASRDebug.logTensorStats("Mel features - mean: \(mean(melFlat).item(Float.self)), std: \(sqrt(variance(melFlat)).item(Float.self))")
            }
        }

        // Add batch dimension: [mel, time] -> [1, mel, time]
        let batchedFeatures = melFeatures.expandedDimensions(axis: 0)

        // Encode audio - returns [time, features] without batch dim (matching Python)
        var audioEmbeds = audioEncoder(batchedFeatures)

        // Add batch dimension for consistency: [time, features] -> [1, time, features]
        audioEmbeds = audioEmbeds.expandedDimensions(axis: 0)

        if Qwen3ASRDebug.enabled {
            Qwen3ASRDebug.log("Audio embeds shape: \(audioEmbeds.shape)")
            if Qwen3ASRDebug.tensorStatsEnabled {
                let embedsFlat = audioEmbeds.flattened()
                Qwen3ASRDebug.logTensorStats("Audio embeds - mean: \(mean(embedsFlat).item(Float.self)), std: \(sqrt(variance(embedsFlat)).item(Float.self))")
                Qwen3ASRDebug.logTensorStats("Audio embeds - min: \(min(embedsFlat).item(Float.self)), max: \(max(embedsFlat).item(Float.self))")
            }
        }

        // Check if text decoder is loaded
        guard let textDecoder = textDecoder else {
            let shape = audioEmbeds.shape
            return "[Audio encoded: \(shape)] - Text decoder not loaded"
        }

        // Generate text using the text decoder
        return generateText(
            audioEmbeds: audioEmbeds,
            textDecoder: textDecoder,
            language: language,
            maxTokens: maxTokens
        )
    }

    /// Generate text from audio embeddings
    /// - Parameters:
    ///   - audioEmbeds: Audio embeddings from encoder
    ///   - textDecoder: Text decoder model
    ///   - language: Target language (nil = let model auto-detect and transcribe in source language)
    ///   - maxTokens: Maximum tokens to generate
    private func generateText(
        audioEmbeds: MLXArray,
        textDecoder: QuantizedTextModel,
        language: String?,
        maxTokens: Int
    ) -> String {
        return _generateText(
            audioEmbeds: audioEmbeds,
            textDecoder: textDecoder,
            language: language,
            maxTokens: maxTokens,
            isTextOnly: false,
            prompt: nil
        )
    }

    /// Generate text from audio embeddings (internal implementation)
    private func _generateText(
        audioEmbeds: MLXArray?,
        textDecoder: QuantizedTextModel,
        language: String?,
        maxTokens: Int,
        isTextOnly: Bool,
        prompt: String?
    ) -> String {
        let env = ProcessInfo.processInfo.environment
        let shouldScaleAudioEmbeds: Bool = {
            // Scaling requires expensive global variance computations.
            // Default off on iOS where it can dominate realtime latency; opt-in via env var.
            #if os(iOS)
            let raw = env["QWEN3_ASR_SCALE_AUDIO_EMBEDS"]?
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased()
            return raw == "1" || raw == "true" || raw == "yes" || raw == "on"
            #else
            let raw = env["QWEN3_ASR_DISABLE_EMBED_SCALE"]?
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased()
            return !(raw == "1" || raw == "true" || raw == "yes" || raw == "on")
            #endif
        }()

        // Qwen3-ASR prompt format (from mlx-audio implementation):
        // <|im_start|>system\n<|im_end|>\n
        // <|im_start|>user\n<|audio_start|><|audio_pad|>...<|audio_end|><|im_end|>\n
        // <|im_start|>assistant\n[language X<asr_text>] <- model generates this if not specified
        //
        // If language is specified: <|im_start|>assistant\nlanguage {lang}<asr_text>
        // If language is nil: <|im_start|>assistant\n (let model output "language X<asr_text>...")

        // Special token IDs
        let imStartId = 151644      // <|im_start|>
        let imEndId = 151645        // <|im_end|>
        let audioStartId = 151669   // <|audio_start|>
        let audioEndId = 151670     // <|audio_end|>
        let audioPadId = 151676     // <|audio_pad|> - placeholder for audio embeddings
        let asrTextId = 151704      // <asr_text>
        let newlineId = 198         // \n

        // Token IDs for "system", "user", "assistant"
        // Verified from vocab.json via tokenizer.debugTokenMappings()
        let systemId = 8948        // "system"
        let userId = 872           // "user"
        let assistantId = 77091    // "assistant"

        // Number of audio tokens (from audio encoder output, 0 for text-only)
        let numAudioTokens = audioEmbeds?.dim(1) ?? 0

        // Build input_ids array.
        var inputIds: [Int32] = []
        let audioStartIndex: Int
        let audioEndIndex: Int

        // <|im_start|>system\n<|im_end|>\n
        inputIds.append(contentsOf: [imStartId, systemId, newlineId, imEndId, newlineId].map { Int32($0) })

        if isTextOnly {
            guard let tokenizer else {
                Qwen3ASRDebug.log("Tokenizer not loaded; cannot run text-only generation")
                return ""
            }
            guard let prompt else {
                Qwen3ASRDebug.log("Missing prompt; cannot run text-only generation")
                return ""
            }

            // <|im_start|>user\n{prompt}<|im_end|>\n
            inputIds.append(contentsOf: [imStartId, userId, newlineId].map { Int32($0) })
            let promptTokens = tokenizer.encode(prompt)
            inputIds.append(contentsOf: promptTokens.map { Int32($0) })
            inputIds.append(contentsOf: [imEndId, newlineId].map { Int32($0) })

            // <|im_start|>assistant\n
            inputIds.append(contentsOf: [imStartId, assistantId, newlineId].map { Int32($0) })

            audioStartIndex = inputIds.count
            audioEndIndex = inputIds.count
        } else {
            // <|im_start|>user\n<|audio_start|><|audio_pad|>...<|audio_end|><|im_end|>\n
            inputIds.append(contentsOf: [imStartId, userId, newlineId, audioStartId].map { Int32($0) })

            let start = inputIds.count
            for _ in 0..<numAudioTokens {
                inputIds.append(Int32(audioPadId))
            }
            let end = inputIds.count

            inputIds.append(contentsOf: [audioEndId, imEndId, newlineId].map { Int32($0) })

            // <|im_start|>assistant\n
            inputIds.append(contentsOf: [imStartId, assistantId, newlineId].map { Int32($0) })

            // Language handling (audio prompt only).
            if let lang = language, let tokenizer = tokenizer {
                let langPrefix = "language \(lang)"
                let langTokens = tokenizer.encode(langPrefix)
                inputIds.append(contentsOf: langTokens.map { Int32($0) })
                inputIds.append(Int32(asrTextId))
                Qwen3ASRDebug.log("Forcing language: \(lang)")
            } else {
                Qwen3ASRDebug.log("Auto-detect mode - model generates full language tag + text")
            }

            audioStartIndex = start
            audioEndIndex = end
        }

        Qwen3ASRDebug.log("Input IDs length: \(inputIds.count), audio tokens: \(numAudioTokens) at positions \(audioStartIndex)..<\(audioEndIndex)")

        // Get text embeddings for all tokens
        let inputIdsTensor = MLXArray(inputIds).expandedDimensions(axis: 0)  // [1, seq_len]
        var inputEmbeds = textDecoder.embedTokens(inputIdsTensor)  // [1, seq_len, hidden]

	        // If we have audio embeddings, insert them into the prompt
	        if let audioEmbeds = audioEmbeds, numAudioTokens > 0 {
            var scaledAudioEmbeds = audioEmbeds
            if shouldScaleAudioEmbeds {
                // Compare embedding scales (expensive; keep optional).
                let textEmbedsFlat = inputEmbeds.flattened()
                let textStd = sqrt(variance(textEmbedsFlat)).item(Float.self)

                let audioEmbedsFlat = audioEmbeds.flattened()
                let audioStd = sqrt(variance(audioEmbedsFlat)).item(Float.self)

                if Qwen3ASRDebug.tensorStatsEnabled {
                    Qwen3ASRDebug.logTensorStats("Text embeds - mean: \(mean(textEmbedsFlat).item(Float.self)), std: \(textStd)")
                    Qwen3ASRDebug.logTensorStats("Audio embeds (before scaling) - mean: \(mean(audioEmbedsFlat).item(Float.self)), std: \(audioStd)")
                }

                // Scale audio embeddings to match text embedding variance
                if audioStd > 0 {
                    let scaleFactor = textStd / audioStd
                    scaledAudioEmbeds = audioEmbeds * scaleFactor
                    Qwen3ASRDebug.log("Scaling audio embeds by \(scaleFactor) to match text std")
                }
            }

            // Replace audio_pad token positions with actual audio embeddings
            // audioEmbeds shape: [1, numAudioTokens, hidden]
            // We need to replace inputEmbeds[0, audioStartIndex:audioEndIndex, :] with audioEmbeds[0, :, :]

            // Build the final embeddings by concatenating parts
            let beforeAudio = inputEmbeds[0..., 0..<audioStartIndex, 0...]  // [1, audioStartIndex, hidden]
            let afterAudio = inputEmbeds[0..., audioEndIndex..., 0...]  // [1, remaining, hidden]

            inputEmbeds = concatenated([beforeAudio, scaledAudioEmbeds, afterAudio], axis: 1)
	        }

	        Qwen3ASRDebug.log("Prompt structure - before_audio:\(audioStartIndex), audio:\(numAudioTokens), after_audio:\(inputIds.count - audioEndIndex)")

        // Initialize KV cache
        var cache: [(MLXArray, MLXArray)]? = nil

        // Generate tokens
        var generatedTokens: [Int32] = []

        // First pass: process the full input embeddings
        var (hiddenStates, newCache) = textDecoder(inputsEmbeds: inputEmbeds, cache: cache)
        cache = newCache

        // Get logits from the last position using embedding as LM head (tied weights)
        // hiddenStates shape: [1, seq_len, hidden]
        let seqLen = hiddenStates.dim(1)
        let lastHidden = hiddenStates[0..., (seqLen-1)..<seqLen, 0...]  // [1, 1, hidden]
        var logits = textDecoder.embedTokens.asLinear(lastHidden)
        var nextToken = argMax(logits, axis: -1).squeezed().item(Int32.self)
        generatedTokens.append(nextToken)

        if Qwen3ASRDebug.enabled {
            let logitsFlat = logits.squeezed()
            Qwen3ASRDebug.log("Logits shape: \(logitsFlat.shape)")
            if Qwen3ASRDebug.tensorStatsEnabled {
                Qwen3ASRDebug.logTensorStats("Logits stats - mean: \(mean(logitsFlat).item(Float.self)), max: \(max(logitsFlat).item(Float.self)), min: \(min(logitsFlat).item(Float.self))")
            }

            Qwen3ASRDebug.log("First token generated: \(nextToken) (EOS=\(Qwen3ASRTokens.eosTokenId))")
            Qwen3ASRDebug.log("Input embeds shape: \(inputEmbeds.shape), hidden states shape: \(hiddenStates.shape)")
        }

        // Continue generating
        for iteration in 1..<maxTokens {
            // Check for EOS
            if nextToken == Int32(Qwen3ASRTokens.eosTokenId) {
                Qwen3ASRDebug.log("EOS token reached at iteration \(iteration)")
                break
            }

            // Get embedding for the new token
            let tokenEmbeds = textDecoder.embedTokens(MLXArray([nextToken]).expandedDimensions(axis: 0))

            // Forward pass with cache
            (hiddenStates, newCache) = textDecoder(inputsEmbeds: tokenEmbeds, cache: cache)
            cache = newCache

            // Get next token
            let lastHiddenNext = hiddenStates[0..., (-1)..., .ellipsis]
            logits = textDecoder.embedTokens.asLinear(lastHiddenNext)
            nextToken = argMax(logits, axis: -1).squeezed().item(Int32.self)
            generatedTokens.append(nextToken)
        }

        if Qwen3ASRDebug.enabled {
            Qwen3ASRDebug.log("Generated \(generatedTokens.count) tokens: \(Array(generatedTokens.prefix(20)))")
        }

        // Decode tokens to text
        if let tokenizer = tokenizer {
            return tokenizer.decode(tokens: generatedTokens.map { Int($0) })
        } else {
            // Fallback: return token IDs
            return generatedTokens.map { String($0) }.joined(separator: " ")
        }
    }

    /// Generate text from a text prompt (no audio)
    /// Used for translation and other text-to-text tasks
    public func generateTextOnly(prompt: String, maxTokens: Int = 200) -> String? {
        guard let textDecoder = textDecoder else {
            Qwen3ASRDebug.log("Text decoder not loaded")
            return nil
        }

        // Use the internal generator without audio embeddings
        let out = _generateText(
            audioEmbeds: nil,
            textDecoder: textDecoder,
            language: nil,
            maxTokens: maxTokens,
            isTextOnly: true,
            prompt: prompt
        )
        return out.isEmpty ? nil : out
    }
}

// The model is used across concurrency boundaries (realtime loops, UI tasks).
// We treat it as effectively immutable after construction/weight load.
extension Qwen3ASRModel: @unchecked Sendable {}

// MARK: - Model Loading

public extension Qwen3ASRModel {
    /// Load model from HuggingFace hub with automatic weight downloading
    static func fromPretrained(
        modelId: String = "mlx-community/Qwen3-ASR-0.6B-4bit",
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> Qwen3ASRModel {
        progressHandler?(0.1, "Downloading model...")

        // Get cache directory
        let cacheDir = try getCacheDirectory(for: modelId)

        // Download weights if needed
        if !weightsExist(in: cacheDir) {
            try await downloadWeights(modelId: modelId, to: cacheDir, progressHandler: { progress in
                progressHandler?(0.1 + progress * 0.4, "Downloading weights...")
            })
        }

        progressHandler?(0.5, "Loading tokenizer...")

        // Create model with default config
        let model = Qwen3ASRModel()

        // Do CPU-heavy synchronous work off the caller's executor (important for iOS UI responsiveness).
        try await runBlocking {
            // Load tokenizer from vocab.json
            let vocabPath = cacheDir.appendingPathComponent("vocab.json")
            if FileManager.default.fileExists(atPath: vocabPath.path) {
                let tokenizer = Qwen3Tokenizer()
                try tokenizer.load(from: vocabPath)
                if Qwen3ASRDebug.enabled {
                    tokenizer.debugTokenMappings()  // verify token IDs
                }
                model.setTokenizer(tokenizer)
            }
        }

        progressHandler?(0.6, "Loading audio encoder weights...")

        // Load audio encoder weights (synchronous / large IO)
        try await runBlocking {
            try WeightLoader.loadWeights(into: model.audioEncoder, from: cacheDir)
        }

        progressHandler?(0.75, "Loading text decoder weights...")

        // Initialize and load text decoder
        model.initializeTextDecoder()
        if let textDecoder = model.textDecoder {
            try await runBlocking {
                try WeightLoader.loadTextDecoderWeights(into: textDecoder, from: cacheDir)
            }
        }

        progressHandler?(1.0, "Ready")

        return model
    }

    private static func runBlocking(_ work: @escaping () throws -> Void) async throws {
        // MLX uses TaskLocal state (device/stream). Running outside Swift Concurrency (e.g. GCD)
        // can bypass that state and lead to incorrect behavior on some platforms.
        try await Task.detached(priority: .userInitiated) {
            try work()
        }.value
    }

    // MARK: Cache Management

    /// Returns the on-disk cache directory for a given modelId.
    ///
    /// Note: This does not create the directory. `fromPretrained` will create it as needed.
    static func cacheDirectoryURL(modelId: String) throws -> URL {
        try _cacheDirectoryURL(for: modelId, create: false)
    }

    /// Returns the total size (in bytes) of the cached files for `modelId`.
    static func cachedModelSizeBytes(modelId: String) throws -> Int64 {
        let dir = try _cacheDirectoryURL(for: modelId, create: false)
        guard FileManager.default.fileExists(atPath: dir.path) else { return 0 }
        return try _directorySizeBytes(dir)
    }

    /// Deletes all cached files for `modelId` (config/tokenizer + weights).
    static func deleteCachedModel(modelId: String) throws {
        let dir = try _cacheDirectoryURL(for: modelId, create: false)
        guard FileManager.default.fileExists(atPath: dir.path) else { return }
        try FileManager.default.removeItem(at: dir)
    }

    /// Deletes the entire `qwen3-asr/` cache root (all models).
    static func deleteAllCachedModels() throws {
        let base = try _baseCacheDirectoryURL(create: false)
        let root = base.appendingPathComponent("qwen3-asr", isDirectory: true)
        guard FileManager.default.fileExists(atPath: root.path) else { return }
        try FileManager.default.removeItem(at: root)
    }

    private static func getCacheDirectory(for modelId: String) throws -> URL {
        try _cacheDirectoryURL(for: modelId, create: true)
    }

    private static func _cacheDirectoryURL(for modelId: String, create: Bool) throws -> URL {
        let cacheKey = _sanitizedCacheKey(for: modelId)
        let fm = FileManager.default

        let baseCacheDir = try _baseCacheDirectoryURL(create: create)
        let cacheDir = baseCacheDir
            .appendingPathComponent("qwen3-asr", isDirectory: true)
            .appendingPathComponent(cacheKey, isDirectory: true)

        if create {
            try fm.createDirectory(at: cacheDir, withIntermediateDirectories: true)
        }
        return cacheDir
    }

    private static func _baseCacheDirectoryURL(create: Bool) throws -> URL {
        let fm = FileManager.default

        // Allow callers (and CI/sandboxes) to override the cache location.
        // Uses a directory path; model subfolders are created underneath it.
        let baseCacheDir: URL
        if let override = ProcessInfo.processInfo.environment["QWEN3_ASR_CACHE_DIR"],
           !override.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            baseCacheDir = URL(fileURLWithPath: override, isDirectory: true)
        } else {
            baseCacheDir = fm.urls(for: .cachesDirectory, in: .userDomainMask).first!
        }

        if create {
            try fm.createDirectory(at: baseCacheDir, withIntermediateDirectories: true)
        }
        return baseCacheDir
    }

    private static func _directorySizeBytes(_ directory: URL) throws -> Int64 {
        let fm = FileManager.default
        let keys: Set<URLResourceKey> = [.isRegularFileKey, .fileSizeKey]
        guard let e = fm.enumerator(at: directory, includingPropertiesForKeys: Array(keys)) else { return 0 }

        var total: Int64 = 0
        for case let url as URL in e {
            let rv = try url.resourceValues(forKeys: keys)
            guard rv.isRegularFile == true else { continue }
            total += Int64(rv.fileSize ?? 0)
        }
        return total
    }

    /// Convert an arbitrary modelId into a single, safe path component for on-disk caching.
    /// This avoids path traversal (`..`) and keeps cache paths stable across runs.
    static func _sanitizedCacheKey(for modelId: String) -> String {
        // Keep the historical behavior for common HF model IDs while disallowing path separators.
        let replaced = modelId.replacingOccurrences(of: "/", with: "_")

        let allowed = CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
        var scalars: [UnicodeScalar] = []
        scalars.reserveCapacity(replaced.unicodeScalars.count)
        for s in replaced.unicodeScalars {
            scalars.append(allowed.contains(s) ? s : "_")
        }

        var cleaned = String(String.UnicodeScalarView(scalars))
        cleaned = cleaned.trimmingCharacters(in: CharacterSet(charactersIn: "._"))

        if cleaned.isEmpty || cleaned == "." || cleaned == ".." {
            cleaned = "model"
        }

        return cleaned
    }

    private static func weightsExist(in directory: URL) -> Bool {
        let fm = FileManager.default
        let contents = (try? fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? []
        return contents.contains { $0.pathExtension == "safetensors" }
    }

    static func _validatedRemoteFileName(_ file: String) throws -> String {
        // Reject any attempt to provide a path instead of a single file name.
        let base = URL(fileURLWithPath: file).lastPathComponent
        guard base == file else {
            throw DownloadError.invalidRemoteFileName(file)
        }

        // Disallow hidden files and path traversal.
        guard !base.isEmpty, !base.hasPrefix("."), !base.contains("..") else {
            throw DownloadError.invalidRemoteFileName(file)
        }

        // Keep this strict: we only expect HF artifact names like `model-00001-of-00002.safetensors`.
        guard base.range(of: #"^[A-Za-z0-9._-]+$"#, options: .regularExpression) != nil else {
            throw DownloadError.invalidRemoteFileName(file)
        }

        return base
    }

    static func _validatedLocalPath(directory: URL, fileName: String) throws -> URL {
        let local = directory.appendingPathComponent(fileName, isDirectory: false)
        let dirPath = directory.standardizedFileURL.path
        let localPath = local.standardizedFileURL.path
        let prefix = dirPath.hasSuffix("/") ? dirPath : (dirPath + "/")
        guard localPath.hasPrefix(prefix) else {
            throw DownloadError.invalidRemoteFileName(fileName)
        }
        return local
    }

    private static func downloadWeights(
        modelId: String,
        to directory: URL,
        progressHandler: ((Double) -> Void)?
    ) async throws {
        let baseURL = "https://huggingface.co/\(modelId)/resolve/main"
        let session = URLSession(configuration: .ephemeral)

        // Files to download (config and tokenizer)
        var filesToDownload = [
            "config.json",
            "vocab.json",
            "tokenizer_config.json"
        ]

        // Determine model file(s) to download
        let indexPath = directory.appendingPathComponent("model.safetensors.index.json")

        if !FileManager.default.fileExists(atPath: indexPath.path) {
            let indexURL = URL(string: "\(baseURL)/model.safetensors.index.json")!
            if let (indexData, indexResponse) = try? await session.data(from: indexURL),
               let httpResponse = indexResponse as? HTTPURLResponse,
               httpResponse.statusCode == 200 {
                try indexData.write(to: indexPath)
            }
        }

        // Check if we have an index file and get model files from it
        var modelFiles: [String] = []
        if FileManager.default.fileExists(atPath: indexPath.path),
           let indexData = try? Data(contentsOf: indexPath),
           let index = try? JSONSerialization.jsonObject(with: indexData) as? [String: Any],
           let weightMap = index["weight_map"] as? [String: String] {
            let uniqueFiles = Set(weightMap.values)
            modelFiles = Array(uniqueFiles).sorted()
            Qwen3ASRDebug.log("Found \(modelFiles.count) model file(s) from index: \(modelFiles)")
        } else {
            modelFiles = ["model.safetensors"]
        }

        filesToDownload.append(contentsOf: modelFiles)

        for (index, file) in filesToDownload.enumerated() {
            let safeFile = try _validatedRemoteFileName(file)
            let localPath = try _validatedLocalPath(directory: directory, fileName: safeFile)

            if FileManager.default.fileExists(atPath: localPath.path) {
                progressHandler?(Double(index + 1) / Double(filesToDownload.count))
                continue
            }

            Qwen3ASRDebug.log("Downloading: \(file)")

            let url = URL(string: "\(baseURL)/\(safeFile)")!
            let (data, response) = try await session.data(from: url)

            guard let httpResponse = response as? HTTPURLResponse,
                  httpResponse.statusCode == 200 else {
                throw DownloadError.failedToDownload(file)
            }

            try data.write(to: localPath)

            progressHandler?(Double(index + 1) / Double(filesToDownload.count))
        }
    }
}

// MARK: - Errors

public enum DownloadError: Error, LocalizedError {
    case failedToDownload(String)
    case invalidRemoteFileName(String)

    public var errorDescription: String? {
        switch self {
        case .failedToDownload(let file):
            return "Failed to download: \(file)"
        case .invalidRemoteFileName(let file):
            return "Refusing to write unsafe remote file name: \(file)"
        }
    }
}
