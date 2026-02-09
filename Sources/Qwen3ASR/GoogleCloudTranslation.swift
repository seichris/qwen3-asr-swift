import Foundation

/// Minimal client for Google Cloud Translation API (v2) using an API key.
///
/// This is intended for lightweight app/CLI integrations where OAuth is not required.
/// Provide the API key via an environment variable in your host app (recommended), or pass it directly.
public enum GoogleCloudTranslation {
    public enum TranslationError: Swift.Error, LocalizedError, Sendable {
        case missingAPIKey
        case invalidHTTPResponse
        case httpStatus(code: Int, body: String)
        case emptyTranslation

        public var errorDescription: String? {
            switch self {
            case .missingAPIKey:
                return "Missing Google Cloud Translation API key."
            case .invalidHTTPResponse:
                return "Invalid HTTP response from Google Cloud Translation."
            case .httpStatus(let code, let body):
                if body.isEmpty { return "Google Cloud Translation failed with HTTP \(code)." }
                return "Google Cloud Translation failed with HTTP \(code): \(body)"
            case .emptyTranslation:
                return "Google Cloud Translation returned an empty translation."
            }
        }
    }

    public static func translate(
        _ text: String,
        apiKey: String,
        sourceLanguage: String? = nil,
        targetLanguage: String,
        session: URLSession = .shared
    ) async throws -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "" }

        let key = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !key.isEmpty else { throw TranslationError.missingAPIKey }

        var comps = URLComponents(string: "https://translation.googleapis.com/language/translate/v2")!
        comps.queryItems = [URLQueryItem(name: "key", value: key)]

        var req = URLRequest(url: comps.url!)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let body = TranslateRequest(
            q: [trimmed],
            target: targetLanguage,
            source: sourceLanguage,
            format: "text"
        )
        req.httpBody = try JSONEncoder().encode(body)

        let (data, resp) = try await session.data(for: req)
        guard let http = resp as? HTTPURLResponse else {
            throw TranslationError.invalidHTTPResponse
        }
        guard (200..<300).contains(http.statusCode) else {
            let bodyStr = String(data: data, encoding: .utf8) ?? ""
            throw TranslationError.httpStatus(code: http.statusCode, body: bodyStr)
        }

        let decoded = try JSONDecoder().decode(TranslateResponse.self, from: data)
        let raw = decoded.data.translations.first?.translatedText ?? ""
        let cleaned = decodeHTMLEntities(raw).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleaned.isEmpty else { throw TranslationError.emptyTranslation }
        return cleaned
    }

    // MARK: - Types

    private struct TranslateRequest: Codable {
        let q: [String]
        let target: String
        let source: String?
        let format: String?
    }

    private struct TranslateResponse: Decodable {
        let data: DataContainer

        struct DataContainer: Decodable {
            let translations: [Translation]
        }

        struct Translation: Decodable {
            let translatedText: String
            let detectedSourceLanguage: String?
        }
    }

    private static func decodeHTMLEntities(_ s: String) -> String {
        // Even with `format=text`, some responses may contain entities.
        var out = s
        let replacements: [(String, String)] = [
            ("&quot;", "\""),
            ("&#39;", "'"),
            ("&apos;", "'"),
            ("&lt;", "<"),
            ("&gt;", ">"),
            ("&amp;", "&"),
        ]
        for (k, v) in replacements {
            out = out.replacingOccurrences(of: k, with: v)
        }
        return out
    }
}

