import Foundation

/// Normalizes user-provided language hints (CLI/UI) into the strings the model
/// is most likely to understand in the `language {X}<asr_text>` prompt prefix.
///
/// The model frequently emits language names like "English" / "Chinese" in its
/// output, so we map common ISO codes and aliases to those names.
public enum Qwen3ASRLanguage {
    /// Normalize a language identifier.
    ///
    /// - Returns: A best-effort normalized language name (e.g. "English").
    ///            If the input is unknown, returns it unchanged.
    public static func normalize(_ input: String) -> String {
        let trimmed = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return input }

        let key = canonicalKey(trimmed)
        if let mapped = aliasMap[key] {
            return mapped
        }

        // If user passed a full language name, preserve it (but normalize casing a bit).
        // Keep the original if it contains spaces or non-letters (likely model-specific).
        if trimmed.range(of: #"[^A-Za-z ]"#, options: .regularExpression) != nil {
            return trimmed
        }
        return trimmed.prefix(1).uppercased() + trimmed.dropFirst().lowercased()
    }

    public static func normalizeOptional(_ input: String?) -> String? {
        guard let input else { return nil }
        let trimmed = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        return normalize(trimmed)
    }

    private static func canonicalKey(_ s: String) -> String {
        // Lowercase, strip whitespace, and normalize separators.
        let lowered = s.lowercased()
        let replaced = lowered
            .replacingOccurrences(of: "_", with: "-")
            .replacingOccurrences(of: " ", with: "")
        return replaced
    }

    // Add more as we confirm the model supports them. Unknown values are passed through.
    private static let aliasMap: [String: String] = [
        "auto": "auto",

        "en": "English",
        "eng": "English",
        "english": "English",

        "zh": "Chinese",
        "zho": "Chinese",
        "chi": "Chinese",
        "cn": "Chinese",
        "zh-cn": "Chinese",
        "zh-hans": "Chinese",
        "chinese": "Chinese",
        "mandarin": "Chinese",

        "ja": "Japanese",
        "jpn": "Japanese",
        "jp": "Japanese",
        "japanese": "Japanese",

        "ko": "Korean",
        "kor": "Korean",
        "korean": "Korean",

        "fr": "French",
        "fra": "French",
        "fre": "French",
        "french": "French",

        "de": "German",
        "deu": "German",
        "ger": "German",
        "german": "German",

        "es": "Spanish",
        "spa": "Spanish",
        "spanish": "Spanish",

        "it": "Italian",
        "ita": "Italian",
        "italian": "Italian",

        "pt": "Portuguese",
        "por": "Portuguese",
        "portuguese": "Portuguese",
        "pt-br": "Portuguese",

        "ru": "Russian",
        "rus": "Russian",
        "russian": "Russian",

        "ar": "Arabic",
        "ara": "Arabic",
        "arabic": "Arabic",

        "hi": "Hindi",
        "hin": "Hindi",
        "hindi": "Hindi",
    ]
}

