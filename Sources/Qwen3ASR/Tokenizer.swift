import Foundation

// MARK: - Errors

public enum TokenizerError: Error, LocalizedError {
    case invalidFormat(String)

    public var errorDescription: String? {
        switch self {
        case .invalidFormat(let reason):
            return "Invalid tokenizer format: \(reason)"
        }
    }
}

/// Simple tokenizer for Qwen3-ASR that loads from vocab.json
public class Qwen3Tokenizer {
    private struct Pair: Hashable {
        let first: String
        let second: String
    }

    internal var idToToken: [Int: String] = [:]
    internal var tokenToId: [String: Int] = [:]
    private var bpeRanks: [Pair: Int] = [:]
    private var bpeCache: [String: [String]] = [:]
    private var specialTokensSorted: [String] = []

    private static let pretokenizationRegex: NSRegularExpression = {
        // GPT-2 / byte-level BPE pre-tokenization pattern.
        let pattern = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
        return try! NSRegularExpression(pattern: pattern, options: [])
    }()

    public var eosTokenId: Int = 151643
    public var padTokenId: Int = 151643
    public var bosTokenId: Int = 151644

    public init() {}

    /// Load tokenizer from vocab.json file (direct token->id mapping)
    public func load(from url: URL) throws {
        let data = try Data(contentsOf: url)

        // vocab.json is a direct {token: id} mapping
        guard let vocab = try JSONSerialization.jsonObject(with: data) as? [String: Int] else {
            throw TokenizerError.invalidFormat("Expected {token: id} dictionary")
        }

        for (token, id) in vocab {
            idToToken[id] = token
            tokenToId[token] = id
        }

        // Also load added tokens from tokenizer_config.json if it exists
        let configUrl = url.deletingLastPathComponent().appendingPathComponent("tokenizer_config.json")
        if FileManager.default.fileExists(atPath: configUrl.path) {
            try loadAddedTokens(from: configUrl)
        }

        // Load BPE merge ranks (required for correct byte-level BPE tokenization).
        let mergesUrl = url.deletingLastPathComponent().appendingPathComponent("merges.txt")
        guard FileManager.default.fileExists(atPath: mergesUrl.path) else {
            throw TokenizerError.invalidFormat("Missing merges.txt next to vocab.json")
        }
        try loadMerges(from: mergesUrl)

        // Special tokens should be matched as whole strings before BPE.
        specialTokensSorted = tokenToId.keys
            .filter { $0.hasPrefix("<") && $0.hasSuffix(">") }
            .sorted { $0.count > $1.count }

        Qwen3ASRDebug.log("Loaded tokenizer with \(idToToken.count) tokens")
    }

    /// Load added tokens from tokenizer_config.json
    private func loadAddedTokens(from url: URL) throws {
        let data = try Data(contentsOf: url)

        guard let config = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return // Not a valid config, skip
        }

        // added_tokens_decoder is a dict with string keys (token IDs) and object values with "content" field
        if let addedTokens = config["added_tokens_decoder"] as? [String: [String: Any]] {
            var addedCount = 0
            for (idString, tokenInfo) in addedTokens {
                guard let id = Int(idString),
                      let content = tokenInfo["content"] as? String else {
                    continue
                }

                // Add to our mappings (overwrite if exists)
                idToToken[id] = content
                tokenToId[content] = id
                addedCount += 1
            }
            Qwen3ASRDebug.log("Loaded \(addedCount) added tokens from tokenizer_config.json")
        }
    }

    /// Load GPT-2 style BPE merges.
    private func loadMerges(from url: URL) throws {
        let contents = try String(contentsOf: url, encoding: .utf8)
        bpeRanks.removeAll(keepingCapacity: true)
        bpeCache.removeAll(keepingCapacity: true)

        var rank = 0
        for rawLine in contents.split(whereSeparator: \.isNewline) {
            let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            if line.isEmpty || line.hasPrefix("#") {
                continue
            }
            let parts = line.split(separator: " ", maxSplits: 1, omittingEmptySubsequences: true)
            guard parts.count == 2 else {
                continue
            }
            let pair = Pair(first: String(parts[0]), second: String(parts[1]))
            bpeRanks[pair] = rank
            rank += 1
        }
        Qwen3ASRDebug.log("Loaded \(rank) BPE merges from merges.txt")
    }

    /// Decode token IDs to text
    public func decode(tokens: [Int]) -> String {
        // IMPORTANT:
        // Byte-level BPE tokens represent *bytes* (mapped to unicode). A single UTF-8 codepoint
        // can span multiple bytes, and the BPE boundary can fall in the middle of that sequence.
        // Therefore decoding must be done on the concatenated byte stream, not token-by-token.

        var stitched = ""
        stitched.reserveCapacity(tokens.count * 2)

        for tokenId in tokens {
            guard let token = idToToken[tokenId] else { continue }

            // Skip chat special tokens like <|im_start|>, <|im_end|>, etc.
            if token.hasPrefix("<|") && token.hasSuffix("|>") {
                continue
            }

            // Preserve markers like <asr_text> (these are ASCII and decode cleanly).
            if token.hasPrefix("<") && token.hasSuffix(">") && !token.contains("|") {
                stitched += token
                continue
            }

            // GPT-2 byte-level space indicator.
            if token.hasPrefix("Ġ") {
                stitched += " "
                stitched += String(token.dropFirst())
            } else {
                stitched += token
            }
        }

        let decoded = decodeByteLevelToken(stitched)
        return decoded.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Byte-to-unicode mapping table (GPT-2 style)
    /// Built lazily on first use
    private static var byteToUnicode: [UInt8: Character] = {
        var mapping: [UInt8: Character] = [:]
        var n = 0

        // Printable ASCII and some extended chars map directly
        let ranges: [(ClosedRange<UInt8>)] = [
            (UInt8(ascii: "!")...UInt8(ascii: "~")),  // 33-126
            (0xA1...0xAC),  // 161-172
            (0xAE...0xFF),  // 174-255
        ]

        for range in ranges {
            for b in range {
                mapping[b] = Character(UnicodeScalar(b))
            }
        }

        // Remaining bytes (0-32, 127-160, 173) map to U+0100 onwards
        for b: UInt8 in 0...255 {
            if mapping[b] == nil {
                mapping[b] = Character(UnicodeScalar(0x100 + n)!)
                n += 1
            }
        }

        return mapping
    }()

    /// Unicode-to-byte reverse mapping
    private static var unicodeToByte: [Character: UInt8] = {
        var reverse: [Character: UInt8] = [:]
        for (byte, char) in byteToUnicode {
            reverse[char] = byte
        }
        return reverse
    }()

    /// Decode byte-level BPE token to proper UTF-8 string
    private func decodeByteLevelToken(_ token: String) -> String {
        var bytes: [UInt8] = []

        for char in token {
            if let byte = Self.unicodeToByte[char] {
                bytes.append(byte)
            } else {
                // Character not in mapping - keep as UTF-8
                bytes.append(contentsOf: String(char).utf8)
            }
        }

        // Decode bytes as UTF-8
        if let decoded = String(bytes: bytes, encoding: .utf8) {
            return decoded
        } else {
            // Fallback to original if decoding fails
            return token
        }
    }

    /// Encode text to token IDs using BPE
    /// - Parameter text: Input text to encode
    /// - Returns: Array of token IDs
    public func encode(_ text: String) -> [Int] {
        guard !text.isEmpty else { return [] }

        var tokenIds: [Int] = []
        var cursor = text.startIndex

        while cursor < text.endIndex {
            if let (token, tokenId) = matchSpecialToken(in: text, at: cursor) {
                tokenIds.append(tokenId)
                cursor = text.index(cursor, offsetBy: token.count)
                continue
            }

            let nextSpecial = nextSpecialTokenStart(in: text, from: cursor) ?? text.endIndex
            let segment = String(text[cursor..<nextSpecial])
            tokenIds.append(contentsOf: encodeOrdinaryText(segment))
            cursor = nextSpecial
        }

        return tokenIds
    }

    private func matchSpecialToken(in text: String, at index: String.Index) -> (String, Int)? {
        let suffix = text[index...]
        for token in specialTokensSorted {
            guard suffix.hasPrefix(token), let id = tokenToId[token] else { continue }
            return (token, id)
        }
        return nil
    }

    private func nextSpecialTokenStart(in text: String, from index: String.Index) -> String.Index? {
        var earliest: String.Index? = nil
        for token in specialTokensSorted {
            guard let range = text.range(of: token, range: index..<text.endIndex) else { continue }
            if earliest == nil || range.lowerBound < earliest! {
                earliest = range.lowerBound
            }
        }
        return earliest
    }

    private func encodeOrdinaryText(_ text: String) -> [Int] {
        guard !text.isEmpty else { return [] }

        let nsText = text as NSString
        let fullRange = NSRange(location: 0, length: nsText.length)
        let matches = Self.pretokenizationRegex.matches(in: text, options: [], range: fullRange)
        var tokenIds: [Int] = []

        for match in matches {
            guard let range = Range(match.range, in: text) else { continue }
            let piece = String(text[range])
            if piece.isEmpty { continue }

            let bytes = [UInt8](piece.utf8)
            let mapped = String(bytes.map { Self.byteToUnicode[$0]! })
            let bpeTokens = applyBPE(to: mapped)

            for token in bpeTokens {
                if let id = tokenToId[token] {
                    tokenIds.append(id)
                }
            }
        }

        return tokenIds
    }

    private func applyBPE(to token: String) -> [String] {
        if let cached = bpeCache[token] {
            return cached
        }
        if token.count <= 1 {
            bpeCache[token] = [token]
            return [token]
        }

        var word = token.map { String($0) }
        var pairs = getPairs(word)

        while !pairs.isEmpty {
            var bestPair: Pair? = nil
            var bestRank = Int.max
            for pair in pairs {
                guard let rank = bpeRanks[pair], rank < bestRank else { continue }
                bestRank = rank
                bestPair = pair
            }

            guard let pairToMerge = bestPair else { break }

            var merged: [String] = []
            merged.reserveCapacity(word.count)
            var i = 0
            while i < word.count {
                if i < word.count - 1 &&
                    word[i] == pairToMerge.first &&
                    word[i + 1] == pairToMerge.second {
                    merged.append(pairToMerge.first + pairToMerge.second)
                    i += 2
                } else {
                    merged.append(word[i])
                    i += 1
                }
            }

            word = merged
            if word.count == 1 { break }
            pairs = getPairs(word)
        }

        bpeCache[token] = word
        return word
    }

    private func getPairs(_ word: [String]) -> Set<Pair> {
        guard word.count >= 2 else { return [] }
        var pairs = Set<Pair>()
        pairs.reserveCapacity(word.count - 1)
        for i in 0..<(word.count - 1) {
            pairs.insert(Pair(first: word[i], second: word[i + 1]))
        }
        return pairs
    }

    /// Get token ID for a specific token string
    public func getTokenId(for token: String) -> Int? {
        return tokenToId[token]
    }

    /// Get token string for a specific ID
    public func getToken(for id: Int) -> String? {
        return idToToken[id]
    }

    /// Debug: print token mappings for common words
    public func debugTokenMappings() {
        let commonTokens = [
            "<|im_start|>", "<|im_end|>", "<|audio_start|>", "<|audio_end|>",
            "<|audio_pad|>", "<asr_text>", "<|endoftext|>",
            "system", "user", "assistant", "language", "English",
            "Ġsystem", "Ġuser", "Ġassistant", "Ġlanguage", "ĠEnglish",
            "\n", "Ċ"  // newline representations
        ]

        print("Token ID mappings:")
        for token in commonTokens {
            if let id = tokenToId[token] {
                print("  '\(token)' -> \(id)")
            } else {
                print("  '\(token)' -> NOT FOUND")
            }
        }
    }
}

/// Protocol for tokenizer to allow different implementations
public protocol TokenizerProtocol {
    func decode(tokens: [Int]) -> String
    func encode(_ text: String) -> [Int]
}

extension Qwen3Tokenizer: TokenizerProtocol {}
