import SwiftUI
import Qwen3ASR
import UniformTypeIdentifiers

@available(iOS 18.0, macOS 15.0, *)
struct SettingsView: View {
    @ObservedObject var vm: LiveTranslateViewModel
    let modelId: String
    let from: SupportedLanguage

    @Environment(\.dismiss) private var dismiss

    @State private var cacheBytes: Int64 = 0
    @State private var isRefreshingSize: Bool = false
    @State private var isDeleting: Bool = false
    @State private var showDeleteConfirm: Bool = false
    @State private var errorMessage: String?
    @State private var showFileImporter: Bool = false
    @AppStorage("DASHSCOPE_API_KEY") private var dashScopeAPIKey: String = ""
    @AppStorage("DASHSCOPE_API_KEY_SG") private var dashScopeAPIKeySG: String = ""
    @AppStorage("QWEN3_ASR_GOOGLE_TRANSLATE_API_KEY") private var googleTranslateAPIKey: String = ""

    var body: some View {
        NavigationStack {
            Form {
                Section("Downloads") {
                    LabeledContent("Model") {
                        Text(modelId)
                            .font(.system(.footnote, design: .monospaced))
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.trailing)
                    }

                    LabeledContent("On-Disk Size") {
                        if isRefreshingSize {
                            ProgressView()
                        } else {
                            Text(ByteCountFormatter.string(fromByteCount: cacheBytes, countStyle: .file))
                                .foregroundStyle(cacheBytes > 0 ? .primary : .secondary)
                        }
                    }

                    Button(role: .destructive) {
                        showDeleteConfirm = true
                    } label: {
                        Text("Delete Downloaded Model Files")
                    }
                    .disabled(vm.isRunning || isDeleting || cacheBytes == 0)

                    if vm.isRunning {
                        Text("Stop listening to delete downloaded files.")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                }

                Section("Diagnostics") {
                    Button {
                        showFileImporter = true
                    } label: {
                        Text("Transcribe Audio File…")
                    }
                    .disabled(vm.isRunning || vm.debugIsTranscribingFile)

                    if vm.debugIsTranscribingFile {
                        HStack(spacing: 10) {
                            ProgressView()
                            Text("Transcribing…")
                                .foregroundStyle(.secondary)
                        }
                    }

                    if !vm.debugFileInfo.isEmpty {
                        Text(vm.debugFileInfo)
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                            .lineLimit(2)
                    }

                    if !vm.debugFileTranscript.isEmpty {
                        Text(vm.debugFileTranscript)
                            .textSelection(.enabled)
                            .font(.footnote)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }

                    Text("This is the simplest ASR path (closest to the CLI): load an audio file, run one transcription, and show the raw ASR text. It bypasses realtime microphone capture, VAD, and transcript stabilization.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                Section("API Keys") {
                    SecureField("DashScope Mainland (DASHSCOPE_API_KEY)", text: $dashScopeAPIKey)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()

                    SecureField("DashScope Singapore (DASHSCOPE_API_KEY_SG)", text: $dashScopeAPIKeySG)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()

                    SecureField("Google Translate (QWEN3_ASR_GOOGLE_TRANSLATE_API_KEY)", text: $googleTranslateAPIKey)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()

                    Text("Keys saved here are stored in app preferences and used when Xcode environment variables are not present (for example, launching from the Home Screen).")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                Section("About") {
                    Text("The app downloads model weights (`*.safetensors`) from Hugging Face and caches them in the app’s Caches directory. Tokenizer/config files are small; weights are the large part.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)

                    Text("If you use Google Cloud translation, set QWEN3_ASR_GOOGLE_TRANSLATE_API_KEY either in Xcode scheme environment variables or in the API Keys section above.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)

                    Text("If you use DashScope hosted ASR, set DASHSCOPE_API_KEY (Mainland) or DASHSCOPE_API_KEY_SG (Singapore) in Xcode scheme environment variables or in the API Keys section above. Then choose the matching ASR source in the ASR dropdown. You can optionally set DASHSCOPE_REALTIME_MODEL to override the default model.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)

                    Text("Audio source dropdown: choose Microphone or Device Audio. Device Audio uses ReplayKit capture on iOS and requires screen-capture permission from the system prompt.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }
            .navigationTitle("Settings")
            .toolbar {
                #if os(iOS)
                ToolbarItem(placement: .cancellationAction) { Button("Done") { dismiss() } }
                #else
                ToolbarItem(placement: .automatic) { Button("Done") { dismiss() } }
                #endif
            }
            .task { await refreshCacheSize() }
            .fileImporter(
                isPresented: $showFileImporter,
                allowedContentTypes: [.audio],
                allowsMultipleSelection: false
            ) { result in
                switch result {
                case .success(let urls):
                    guard let url = urls.first else { return }
                    Task { @MainActor in
                        // Best-effort: handle security-scoped URLs (iOS Files).
                        let ok = url.startAccessingSecurityScopedResource()
                        defer { if ok { url.stopAccessingSecurityScopedResource() } }
                        await vm.transcribeAudioFile(modelId: modelId, from: from, url: url)
                    }
                case .failure(let error):
                    errorMessage = String(describing: error)
                }
            }
            .alert("Delete Downloads?", isPresented: $showDeleteConfirm) {
                Button("Cancel", role: .cancel) {}
                Button("Delete", role: .destructive) {
                    Task { await deleteCachedModel() }
                }
            } message: {
                Text("This removes the downloaded model files from on-device storage. The app will re-download them next time you start transcription.")
            }
            .alert("Error", isPresented: Binding(get: { errorMessage != nil }, set: { _ in errorMessage = nil })) {
                Button("OK", role: .cancel) {}
            } message: {
                Text(errorMessage ?? "")
            }
        }
    }

    private func refreshCacheSize() async {
        guard !isRefreshingSize else { return }
        isRefreshingSize = true
        defer { isRefreshingSize = false }
        do {
            cacheBytes = try Qwen3ASRModel.cachedModelSizeBytes(modelId: modelId)
        } catch {
            errorMessage = String(describing: error)
        }
    }

    private func deleteCachedModel() async {
        guard !isDeleting else { return }
        isDeleting = true
        defer { isDeleting = false }

        do {
            // Ensure we don't hold onto a model instance while removing its files.
            await MainActor.run { vm.unloadModel() }
            try Qwen3ASRModel.deleteCachedModel(modelId: modelId)
            await refreshCacheSize()
        } catch {
            errorMessage = String(describing: error)
        }
    }
}
