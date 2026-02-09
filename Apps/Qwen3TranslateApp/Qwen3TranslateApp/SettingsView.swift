import SwiftUI
import Qwen3ASR

@available(iOS 18.0, macOS 15.0, *)
struct SettingsView: View {
    @ObservedObject var vm: LiveTranslateViewModel
    let modelId: String

    @Environment(\.dismiss) private var dismiss

    @State private var cacheBytes: Int64 = 0
    @State private var isRefreshingSize: Bool = false
    @State private var isDeleting: Bool = false
    @State private var showDeleteConfirm: Bool = false
    @State private var errorMessage: String?

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

                Section("About") {
                    Text("The app downloads model weights (`*.safetensors`) from Hugging Face and caches them in the app’s Caches directory. Tokenizer/config files are small; weights are the large part.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)

                    Text("If you use Google Cloud translation, set QWEN3_ASR_GOOGLE_TRANSLATE_API_KEY in the app's environment (Xcode scheme) so the app can call the Translation API.")
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
