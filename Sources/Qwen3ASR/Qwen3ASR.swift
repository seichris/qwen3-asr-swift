import Foundation
import Dispatch
#if canImport(CryptoKit)
import CryptoKit
#endif
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
        let env = ProcessInfo.processInfo.environment
        let rawDevice = env["QWEN3_ASR_DEVICE"]?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()

        // MLX CPU backend is not available on iOS. Default to GPU; allow forcing CPU on macOS/CLI.
        let device: Device = {
            #if os(iOS)
            if rawDevice == "cpu" {
                Qwen3ASRDebug.log("QWEN3_ASR_DEVICE=cpu requested on iOS, but MLX CPU backend is not supported; using GPU.")
            }
            return .gpu
            #else
            return (rawDevice == "cpu") ? .cpu : .gpu
            #endif
        }()

        if Qwen3ASRDebug.enabled {
            Qwen3ASRDebug.log("Qwen3ASRModel.transcribe: device=\(device)")
            Qwen3ASRDebug.log("Qwen3ASRModel.transcribe: audio_samples=\(audio.count) sampleRate=\(sampleRate)")
            if !audio.isEmpty {
                // Quick amplitude sanity (helps detect out-of-range audio on iOS).
                var minV = Float.greatestFiniteMagnitude
                var maxV = -Float.greatestFiniteMagnitude
                var sumSq: Double = 0
                for v in audio {
                    if v < minV { minV = v }
                    if v > maxV { maxV = v }
                    sumSq += Double(v) * Double(v)
                }
                let rms = sqrt(sumSq / Double(audio.count))
                Qwen3ASRDebug.log(String(format: "Qwen3ASRModel.transcribe: audio_min=%.6f audio_max=%.6f audio_rms=%.6f", minV, maxV, rms))
            }
        }

        return Device.withDefaultDevice(device) {
            // Extract mel features
            let melFeatures = featureExtractor.process(audio, sampleRate: sampleRate)

            if Qwen3ASRDebug.enabled {
                Qwen3ASRDebug.log("Mel features shape: \(melFeatures.shape)")
                Qwen3ASRDebug.log("Mel features dtype: \(melFeatures.dtype)")
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
                Qwen3ASRDebug.log("Audio embeds dtype: \(audioEmbeds.dtype)")
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
        let debugAudioInfluence: Bool = {
            let raw = env["QWEN3_ASR_DEBUG_AUDIO_INFLUENCE"]?
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased()
            switch raw {
            case "1", "true", "yes", "y", "on":
                return true
            default:
                return false
            }
        }()
        let shouldScaleAudioEmbeds: Bool = {
            // Default ON everywhere (matches CLI). Disable via env var when needed for performance.
            let raw = env["QWEN3_ASR_DISABLE_EMBED_SCALE"]?
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased()
            return !(raw == "1" || raw == "true" || raw == "yes" || raw == "on")
        }()

        let debugTopK: Int = {
            let raw = env["QWEN3_ASR_DEBUG_TOPK"]?
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased()
            guard let raw, !raw.isEmpty else { return 0 }
            if raw == "1" || raw == "true" || raw == "yes" || raw == "on" { return 8 }
            if let n = Int(raw), n > 0 { return min(n, 50) }
            return 0
        }()

        let debugLMHeadCheck: Bool = {
            let raw = env["QWEN3_ASR_DEBUG_LMHEAD_CHECK"]?
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased()
            switch raw {
            case "1", "true", "yes", "y", "on":
                return true
            default:
                return false
            }
        }()

        let debugLMHeadCheckN: Int = {
            let raw = env["QWEN3_ASR_DEBUG_LMHEAD_CHECK_N"]?
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased()
            guard let raw, !raw.isEmpty else { return 256 }
            if let n = Int(raw), n > 0 { return min(max(n, 32), 2048) }
            return 256
        }()

        // Qwen3-ASR prompt format (align with reference processor):
        // <|im_start|>user\n<|audio_start|><|audio_pad|>...<|audio_end|><|im_end|>\n
        // <|im_start|>assistant\n[ language X<asr_text>]
        //
        // Note the leading space before "language" when forced.

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

        if isTextOnly {
            guard let tokenizer else {
                Qwen3ASRDebug.log("Tokenizer not loaded; cannot run text-only generation")
                return ""
            }
            guard let prompt else {
                Qwen3ASRDebug.log("Missing prompt; cannot run text-only generation")
                return ""
            }

            // Keep the legacy text-only chat template.
            inputIds.append(contentsOf: [imStartId, systemId, newlineId, imEndId, newlineId].map { Int32($0) })

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
	            // Match chat_template.json: always include a system message, even if empty.
	            // <|im_start|>system\n<|im_end|>\n
	            inputIds.append(contentsOf: [imStartId, systemId, newlineId, imEndId, newlineId].map { Int32($0) })

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
                // Keep the leading space before "language" to match processor behavior.
                let langPrefix = " language \(lang)<asr_text>"
                var langTokens = tokenizer.encode(langPrefix)
                if !langTokens.contains(asrTextId) {
                    langTokens.append(asrTextId)
                }
                inputIds.append(contentsOf: langTokens.map { Int32($0) })
                Qwen3ASRDebug.log("Forcing language: \(lang)")
            } else {
                Qwen3ASRDebug.log("Auto-detect mode - model generates full language tag + text")
            }

            audioStartIndex = start
            audioEndIndex = end
        }

        Qwen3ASRDebug.log("Input IDs length: \(inputIds.count), audio tokens: \(numAudioTokens) at positions \(audioStartIndex)..<\(audioEndIndex)")

        Qwen3ASRDebug.logMemory("after_prompt_build")

        // Get text embeddings for all tokens
        let inputIdsTensor = MLXArray(inputIds).expandedDimensions(axis: 0)  // [1, seq_len]
        var inputEmbeds = textDecoder.embedTokens(inputIdsTensor)  // [1, seq_len, hidden]
        let inputEmbedsNoAudio = inputEmbeds

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
            // Avoid open-ended slices on iOS/Metal; some builds have had slicing quirks.
            let seqLenEmbeds = inputEmbeds.dim(1)
            let beforeAudio = inputEmbeds[0..., 0..<audioStartIndex, 0...]  // [1, audioStartIndex, hidden]
            let afterAudio = inputEmbeds[0..., audioEndIndex..<seqLenEmbeds, 0...]  // [1, remaining, hidden]

            inputEmbeds = concatenated([beforeAudio, scaledAudioEmbeds, afterAudio], axis: 1)
	        }

	        Qwen3ASRDebug.log("Prompt structure - before_audio:\(audioStartIndex), audio:\(numAudioTokens), after_audio:\(inputIds.count - audioEndIndex)")
        Qwen3ASRDebug.logMemory("after_embed_insert")

        // Initialize KV cache
        var cache: [(MLXArray, MLXArray)]? = nil

        // Generate tokens
        var generatedTokens: [Int32] = []

        // Optional debug: run a second step-0 pass without inserting audio embeddings and compare.
        // This is expensive; keep behind QWEN3_ASR_DEBUG_AUDIO_INFLUENCE=1.
        var debugNoAudioLogits: MLXArray? = nil
        var debugNoAudioNextToken: Int32? = nil

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

        if Qwen3ASRDebug.enabled, debugLMHeadCheck {
            // Validate lm-head (quantizedMM transpose path) on a subset of vocab rows:
            // Compare `asLinear(lastHidden)` vs `matmul(lastHidden, dequantized(rows).T)` for selected ids.
            //
            // This is intended to catch Metal backend issues that might not show up in generic diagnostics.
            let vocabSize = textDecoder.config.vocabSize
            let n = min(debugLMHeadCheckN, vocabSize)
            let topM = min(64, n)

            // Materialize logits on CPU once (vocab ~151k floats => ~0.6MB).
            let flatAll = logits.squeezed().asArray(Float.self)

            // Extract topM token ids from the quantized logits.
            var best: [(idx: Int, value: Float)] = []
            best.reserveCapacity(topM)
            for (i, v) in flatAll.enumerated() {
                if best.count < topM {
                    best.append((i, v))
                    if best.count == topM { best.sort { $0.value > $1.value } }
                    continue
                }
                if v <= best[topM - 1].value { continue }
                best[topM - 1] = (i, v)
                var j = topM - 1
                while j > 0 && best[j].value > best[j - 1].value {
                    best.swapAt(j, j - 1)
                    j -= 1
                }
            }

            var ids: [Int32] = best.map { Int32($0.idx) }

            // Fill remaining ids with evenly spaced samples over vocab (deterministic).
            if ids.count < n {
                let remaining = n - ids.count
                let stride = max(1, vocabSize / remaining)
                var x = 0
                while ids.count < n {
                    ids.append(Int32(x))
                    x = (x + stride) % vocabSize
                }
            }

            // Compute reference logits on this subset via dequantized rows.
            let idsTensor = MLXArray(ids)
            let wRows = textDecoder.embedTokens.dequantizeRows(idsTensor, dtype: .float32) // [n, hidden]
            let h = lastHidden.squeezed(axis: 0).squeezed(axis: 0).asType(.float32) // [hidden]
            let ref = matmul(h.expandedDimensions(axis: 0), wRows.transposed(1, 0)).squeezed().asType(.float32) // [n]

            // Gather quantized logits for the same ids.
            let q = logits.squeezed()[idsTensor].asType(.float32) // [n]

            let diff = abs(q - ref).flattened()
            let maxAbs = max(diff).item(Float.self)
            let meanAbs = mean(diff).item(Float.self)
            Qwen3ASRDebug.log(String(format: "LMHeadCheck: n=%d max_abs=%.6f mean_abs=%.6f", n, maxAbs, meanAbs))
        }

        if Qwen3ASRDebug.enabled {
            let logitsFlat = logits.squeezed()
            Qwen3ASRDebug.log("Logits shape: \(logitsFlat.shape)")
            if Qwen3ASRDebug.tensorStatsEnabled {
                Qwen3ASRDebug.logTensorStats("Logits stats - mean: \(mean(logitsFlat).item(Float.self)), max: \(max(logitsFlat).item(Float.self)), min: \(min(logitsFlat).item(Float.self))")
            }

            Qwen3ASRDebug.log("First token generated: \(nextToken) (EOS=\(Qwen3ASRTokens.eosTokenId))")
            Qwen3ASRDebug.log("Input embeds shape: \(inputEmbeds.shape), hidden states shape: \(hiddenStates.shape)")
        }

        if Qwen3ASRDebug.enabled, debugAudioInfluence, numAudioTokens > 0 {
            // Compare step-0 logits with and without audio insertion. If these are nearly identical,
            // the decoder is effectively ignoring audio (masking/op bug).
            let (hsNoAudio, _) = textDecoder(inputsEmbeds: inputEmbedsNoAudio, cache: nil)
            let seqLenNoAudio = hsNoAudio.dim(1)
            let lastHiddenNoAudio = hsNoAudio[0..., (seqLenNoAudio - 1)..<seqLenNoAudio, 0...]  // [1,1,hidden]
            let logitsNoAudio = textDecoder.embedTokens.asLinear(lastHiddenNoAudio)
            let nextNoAudio = argMax(logitsNoAudio, axis: -1).squeezed().item(Int32.self)

            debugNoAudioLogits = logitsNoAudio
            debugNoAudioNextToken = nextNoAudio

            // Cosine similarity + L1/L2 deltas between last hidden states.
            let a = lastHidden.squeezed(axis: 0).squeezed(axis: 0)
            let b = lastHiddenNoAudio.squeezed(axis: 0).squeezed(axis: 0)
            let dot = sum(a * b).item(Float.self)
            let normA = sqrt(sum(a * a)).item(Float.self)
            let normB = sqrt(sum(b * b)).item(Float.self)
            let cos = (normA > 0 && normB > 0) ? (dot / (normA * normB)) : 0

            let diff = abs(a - b)
            let l1 = sum(diff).item(Float.self)
            let l2 = sqrt(sum(diff * diff)).item(Float.self)

            Qwen3ASRDebug.log(String(format: "AudioInfluence: lastHidden cos=%.6f l1=%.3f l2=%.3f next_with_audio=%d next_no_audio=%d", cos, l1, l2, nextToken, nextNoAudio))
        }

        if Qwen3ASRDebug.enabled, debugTopK > 0, let tokenizer = tokenizer {
            // Debugging helper: show what the model thinks the top candidates are at step 0.
            // This helps distinguish "bad audio embeddings/prompt" vs "bad sampling".
            let flat = logits.squeezed().asArray(Float.self)
            var best: [(idx: Int, value: Float)] = []
            best.reserveCapacity(debugTopK)
            for (i, v) in flat.enumerated() {
                if best.count < debugTopK {
                    best.append((i, v))
                    if best.count == debugTopK {
                        best.sort { $0.value > $1.value }
                    }
                    continue
                }
                if v <= best[debugTopK - 1].value { continue }
                best[debugTopK - 1] = (i, v)
                var j = debugTopK - 1
                while j > 0 && best[j].value > best[j - 1].value {
                    best.swapAt(j, j - 1)
                    j -= 1
                }
            }

            Qwen3ASRDebug.log("Top-\(debugTopK) tokens @ step0:")
            for (rank, entry) in best.enumerated() {
                let id = entry.idx
                let rawTok = tokenizer.getToken(for: id) ?? "<?>"
                // Single-token decode is imperfect (Tokenizer trims); still useful for spotting obvious junk.
                let decoded = tokenizer.decode(tokens: [id])
                let valStr = String(format: "%.4f", entry.value)
                Qwen3ASRDebug.log("  #\(rank + 1): id=\(id) logit=\(valStr) raw='\(rawTok)' decoded='\(decoded)'")
            }

            if Qwen3ASRDebug.enabled, debugAudioInfluence, let debugNoAudioLogits {
                let flatNoAudio = debugNoAudioLogits.squeezed().asArray(Float.self)
                var bestNoAudio: [(idx: Int, value: Float)] = []
                bestNoAudio.reserveCapacity(debugTopK)
                for (i, v) in flatNoAudio.enumerated() {
                    if bestNoAudio.count < debugTopK {
                        bestNoAudio.append((i, v))
                        if bestNoAudio.count == debugTopK {
                            bestNoAudio.sort { $0.value > $1.value }
                        }
                        continue
                    }
                    if v <= bestNoAudio[debugTopK - 1].value { continue }
                    bestNoAudio[debugTopK - 1] = (i, v)
                    var j = debugTopK - 1
                    while j > 0 && bestNoAudio[j].value > bestNoAudio[j - 1].value {
                        bestNoAudio.swapAt(j, j - 1)
                        j -= 1
                    }
                }

                Qwen3ASRDebug.log("Top-\(debugTopK) tokens @ step0 (no audio):")
                for (rank, entry) in bestNoAudio.enumerated() {
                    let id = entry.idx
                    let rawTok = tokenizer.getToken(for: id) ?? "<?>"
                    let decoded = tokenizer.decode(tokens: [id])
                    let valStr = String(format: "%.4f", entry.value)
                    Qwen3ASRDebug.log("  #\(rank + 1): id=\(id) logit=\(valStr) raw='\(rawTok)' decoded='\(decoded)'")
                }
                if let debugNoAudioNextToken {
                    Qwen3ASRDebug.log("AudioInfluence: next_no_audio=\(debugNoAudioNextToken)")
                }
            }
        }

        let streamTokens: Bool = {
            let raw = env["QWEN3_ASR_DEBUG_STREAM_TOKENS"]?
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased()
            return raw == "1" || raw == "true" || raw == "yes" || raw == "on"
        }()

        // Continue generating
        // In forced-language mode, we already include `<asr_text>` in the prompt, so we are
        // effectively "inside" the ASR text region from the start of generation.
        let promptHasASRText = (!isTextOnly && language != nil)
        var seenASRTextTag = promptHasASRText
        for iteration in 1..<maxTokens {
            // Check for EOS
            // Hugging Face generation_config.json lists eos_token_id: [151643, 151645]
            // i.e. <|endoftext|> and <|im_end|>. Stop on either.
            if nextToken == Int32(Qwen3ASRTokens.eosTokenId) || nextToken == Int32(Qwen3ASRTokens.padTokenId) {
                Qwen3ASRDebug.log("EOS token reached at iteration \(iteration)")
                break
            }

            if nextToken == Int32(151704) { // <asr_text>
                seenASRTextTag = true
            }

            // Get embedding for the new token
            let tokenEmbeds = textDecoder.embedTokens(MLXArray([nextToken]).expandedDimensions(axis: 0))

            // Forward pass with cache
            (hiddenStates, newCache) = textDecoder(inputsEmbeds: tokenEmbeds, cache: cache)
            cache = newCache

            // Get next token
            // hiddenStates is [1, 1, hidden] here; avoid negative-index slicing which has
            // had backend-specific quirks on some iOS/Metal builds.
            let lastHiddenNext = hiddenStates
            logits = textDecoder.embedTokens.asLinear(lastHiddenNext)
            nextToken = argMax(logits, axis: -1).squeezed().item(Int32.self)
            generatedTokens.append(nextToken)

            // Stop on any special control token to avoid drifting into chat-template junk.
            // (The decoder can emit these when it gets confused / doesn't emit EOS.)
            if nextToken == Int32(Qwen3ASRTokens.imStartTokenId) ||
                nextToken == Int32(Qwen3ASRTokens.audioStartTokenId) ||
                nextToken == Int32(Qwen3ASRTokens.audioEndTokenId) ||
                nextToken == Int32(Qwen3ASRTokens.audioTokenId) {
                Qwen3ASRDebug.log("Stopping early due to control token: \(nextToken)")
                break
            }

            // Early-stop heuristic for ASR: after we've entered the ASR text region, if the model
            // starts repeating the same token many times, it's usually a degeneration loop.
            if seenASRTextTag && generatedTokens.count >= 32 {
                let last = generatedTokens.last!
                var run = 0
                for t in generatedTokens.reversed() {
                    if t == last { run += 1 } else { break }
                }
                if run >= 16 {
                    Qwen3ASRDebug.log("Stopping early due to repetition loop: token=\(last) run=\(run)")
                    break
                }
            }

            // Lower-diversity tail guard (catches loops like "cke cke cke..." even if run-length is short).
            if seenASRTextTag && generatedTokens.count >= 64 {
                let tail = generatedTokens.suffix(64)
                let uniq = Set(tail).count
                if uniq <= 4 {
                    Qwen3ASRDebug.log("Stopping early due to low-diversity tail: unique=\(uniq)")
                    break
                }
            }

            if Qwen3ASRDebug.enabled, streamTokens, iteration % 16 == 0 {
                Qwen3ASRDebug.log("Tokens@\(iteration): \(Array(generatedTokens.suffix(16)))")
            }
        }

        if Qwen3ASRDebug.enabled {
            Qwen3ASRDebug.log("Generated \(generatedTokens.count) tokens: \(Array(generatedTokens.prefix(20)))")
        }

        // Decode tokens to text
        if let tokenizer = tokenizer {
            let raw = tokenizer.decode(tokens: generatedTokens.map { Int($0) })
            let out = postProcessASROutput(raw)
            if Qwen3ASRDebug.enabled {
                let preview = out.prefix(200)
                Qwen3ASRDebug.log("ASR_TEXT(len=\(out.count)): \(preview)")
            }
            return out
        } else {
            // Fallback: return token IDs
            return generatedTokens.map { String($0) }.joined(separator: " ")
        }
    }

    /// Qwen3-ASR raw output is often: "language {lang}<asr_text>{transcription}".
    /// This strips everything up to and including the last "<asr_text>" tag.
    private func postProcessASROutput(_ text: String) -> String {
        let tag = "<asr_text>"
        guard let r = text.range(of: tag, options: .backwards) else {
            return text
        }
        let out = text[r.upperBound...]
        return out.trimmingCharacters(in: .whitespacesAndNewlines)
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

        #if os(iOS)
        // MLX uses a buffer recycling cache which can grow during inference.
        // On iOS, constraining this cache reduces jetsam (OOM) risk.
        configureMLXMemoryForiOS()
        #endif

        // Get cache directory
        let cacheDir = try getCacheDirectory(for: modelId)

        // Download artifacts if needed.
        // Existing caches may contain weights but miss tokenizer files (e.g. merges.txt).
        if !weightsExist(in: cacheDir) || !tokenizerArtifactsExist(in: cacheDir) {
            try await downloadWeights(modelId: modelId, to: cacheDir, progressHandler: { progress in
                progressHandler?(0.1 + progress * 0.4, "Downloading weights...")
            })
        }

        if Qwen3ASRDebug.enabled {
            _debugLogCachedArtifacts(cacheDir: cacheDir)
        }

	        progressHandler?(0.5, "Loading tokenizer...")

	        // Create model with config parsed from Hugging Face config.json when available.
	        // Rotary (RoPE) flags in config.json materially affect correctness.
	        let textConfig = (try? loadTextConfigIfPresent(cacheDir: cacheDir)) ?? .small
	        let model = Qwen3ASRModel(textConfig: textConfig)

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

        #if os(iOS)
        // iOS-only kernel sanity checks. These print when either:
        // - `QWEN3_ASR_DIAGNOSTICS=1`, or
        // - `QWEN3_ASR_DEBUG=1` (common during iOS debugging).
        _ = Device.withDefaultDevice(.gpu) {
            Qwen3ASRDiagnostics.runOnDeviceAfterLoad(textDecoder: model.textDecoder)
        }
        #endif

	        progressHandler?(1.0, "Ready")

	        return model
	    }

	    /// Load key text-decoder configuration from Hugging Face `config.json`.
	    /// We intentionally only consume the subset that affects correctness in our implementation.
	    private static func loadTextConfigIfPresent(cacheDir: URL) throws -> TextDecoderConfig? {
	        let url = cacheDir.appendingPathComponent("config.json")
	        guard FileManager.default.fileExists(atPath: url.path) else { return nil }

	        let data = try Data(contentsOf: url)
	        guard let rootAny = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
	        guard
	            let thinker = rootAny["thinker_config"] as? [String: Any],
	            let text = thinker["text_config"] as? [String: Any]
	        else { return nil }

	        var cfg = TextDecoderConfig.small

	        func int(_ key: String) -> Int? { text[key] as? Int }
	        func float(_ key: String) -> Float? {
	            if let f = text[key] as? Double { return Float(f) }
	            if let f = text[key] as? Float { return f }
	            if let f = text[key] as? Int { return Float(f) }
	            return nil
	        }

	        if let v = int("vocab_size") { cfg.vocabSize = v }
	        if let v = int("hidden_size") { cfg.hiddenSize = v }
	        if let v = int("num_hidden_layers") { cfg.numLayers = v }
	        if let v = int("num_attention_heads") { cfg.numHeads = v }
	        if let v = int("num_key_value_heads") { cfg.numKVHeads = v }
	        if let v = int("head_dim") { cfg.headDim = v }
	        if let v = int("intermediate_size") { cfg.intermediateSize = v }
	        if let v = int("max_position_embeddings") { cfg.maxPositionEmbeddings = v }
	        if let v = float("rms_norm_eps") { cfg.rmsNormEps = v }
	        if let v = float("rope_theta") { cfg.ropeTheta = v }

	        if let ropeScaling = text["rope_scaling"] as? [String: Any] {
	            if let v = ropeScaling["interleaved"] as? Bool { cfg.ropeInterleaved = v }
	            if let v = ropeScaling["mrope_interleaved"] as? Bool { cfg.mropeInterleaved = v }
	            if let v = ropeScaling["mrope_section"] as? [Int] { cfg.mropeSection = v }
	        }

	        if let quant = rootAny["quantization"] as? [String: Any] {
	            if let v = quant["group_size"] as? Int { cfg.groupSize = v }
	            if let v = quant["bits"] as? Int { cfg.bits = v }
	        }

	        return cfg
	    }

	    #if os(iOS)
	    private static func configureMLXMemoryForiOS() {
	        let env = ProcessInfo.processInfo.environment

        func parseInt(_ key: String) -> Int? {
            guard let raw = env[key]?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !raw.isEmpty
            else { return nil }
            return Int(raw)
        }

        // Allow explicit overrides:
        // - `QWEN3_ASR_MLX_CACHE_LIMIT_MB`: cache limit in MB
        // - `QWEN3_ASR_MLX_CACHE_LIMIT_BYTES`: cache limit in bytes
        let bytesOverride = parseInt("QWEN3_ASR_MLX_CACHE_LIMIT_BYTES")
        let mbOverride = parseInt("QWEN3_ASR_MLX_CACHE_LIMIT_MB")

        let limitBytes: Int = {
            if let b = bytesOverride, b >= 0 { return b }
            if let mb = mbOverride, mb >= 0 { return mb * 1024 * 1024 }
            // Default: keep cache modest to reduce jetsam risk.
            // This does NOT cap active memory (weights + live tensors).
            return 256 * 1024 * 1024
        }()

        if MLX.Memory.cacheLimit != limitBytes {
            MLX.Memory.cacheLimit = limitBytes
            Qwen3ASRDebug.log("MLX.Memory.cacheLimit set to \(limitBytes) bytes (iOS)")
        }
    }
    #endif

    private static func _debugLogCachedArtifacts(cacheDir: URL) {
        let env = ProcessInfo.processInfo.environment
        let wantHashes: Bool = {
            let raw = env["QWEN3_ASR_DEBUG_FILE_HASH"]?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            return raw == "1" || raw == "true" || raw == "yes" || raw == "on"
        }()
        let wantModelHash: Bool = {
            let raw = env["QWEN3_ASR_DEBUG_MODEL_HASH"]?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            return raw == "1" || raw == "true" || raw == "yes" || raw == "on"
        }()

        let files = [
            "config.json",
            "vocab.json",
            "tokenizer_config.json",
            "merges.txt",
            "model.safetensors",
        ]

        for name in files {
            let url = cacheDir.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: url.path) else { continue }

            let size = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.int64Value ?? -1
            Qwen3ASRDebug.log("Cache artifact: \(name) size=\(size) bytes")

            guard wantHashes else { continue }
            if name == "model.safetensors" && !wantModelHash { continue }
            if let sha = _sha256Hex(url: url) {
                Qwen3ASRDebug.log("Cache artifact: \(name) sha256=\(sha)")
            } else {
                Qwen3ASRDebug.log("Cache artifact: \(name) sha256=<unavailable>")
            }
        }
    }

    private static func _sha256Hex(url: URL) -> String? {
        #if canImport(CryptoKit)
        guard let fh = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? fh.close() }

        var hasher = SHA256()
        while true {
            guard let data = try? fh.read(upToCount: 1024 * 1024), !data.isEmpty else {
                break
            }
            hasher.update(data: data)
        }

        let digest = hasher.finalize()
        return digest.map { String(format: "%02x", $0) }.joined()
        #else
        return nil
        #endif
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

    private static func tokenizerArtifactsExist(in directory: URL) -> Bool {
        let fm = FileManager.default
        let required = [
            "vocab.json",
            "tokenizer_config.json",
            "merges.txt",
        ]
        return required.allSatisfy { name in
            fm.fileExists(atPath: directory.appendingPathComponent(name).path)
        }
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
            "tokenizer_config.json",
            "merges.txt",
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
