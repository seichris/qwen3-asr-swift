import Foundation

struct SupportedLanguage: Identifiable, Hashable {
    let id: String              // ISO-ish code for Apple Translation / Locale.Language
    let displayName: String     // UI label
    let modelName: String       // Qwen3-ASR prompt language name (e.g. "Chinese")

    /// Auto-detect source language (pass nil to the model).
    static let autoDetect = SupportedLanguage(id: "auto", displayName: "Auto-detect", modelName: "auto")

    static let chinese = SupportedLanguage(id: "zh", displayName: "Chinese", modelName: "Chinese")
    static let english = SupportedLanguage(id: "en", displayName: "English", modelName: "English")
    static let japanese = SupportedLanguage(id: "ja", displayName: "Japanese", modelName: "Japanese")
    static let korean = SupportedLanguage(id: "ko", displayName: "Korean", modelName: "Korean")
    static let french = SupportedLanguage(id: "fr", displayName: "French", modelName: "French")
    static let german = SupportedLanguage(id: "de", displayName: "German", modelName: "German")
    static let spanish = SupportedLanguage(id: "es", displayName: "Spanish", modelName: "Spanish")
    static let italian = SupportedLanguage(id: "it", displayName: "Italian", modelName: "Italian")
    static let portuguese = SupportedLanguage(id: "pt", displayName: "Portuguese", modelName: "Portuguese")
    static let russian = SupportedLanguage(id: "ru", displayName: "Russian", modelName: "Russian")
    static let arabic = SupportedLanguage(id: "ar", displayName: "Arabic", modelName: "Arabic")
    static let hindi = SupportedLanguage(id: "hi", displayName: "Hindi", modelName: "Hindi")

    // Conservative “intersection” list (Qwen3-ASR supports these; Apple Translation availability may vary by OS).
    static let all: [SupportedLanguage] = [
        .chinese,
        .english,
        .japanese,
        .korean,
        .french,
        .german,
        .spanish,
        .italian,
        .portuguese,
        .russian,
        .arabic,
        .hindi,
    ]

    static let sources: [SupportedLanguage] = [.autoDetect] + all
    static let targets: [SupportedLanguage] = all

    /// Pass to `Qwen3ASRModel.transcribe(language:)` / `RealtimeTranslationOptions.sourceLanguage`.
    var modelNameOptional: String? {
        id == "auto" ? nil : modelName
    }
}
