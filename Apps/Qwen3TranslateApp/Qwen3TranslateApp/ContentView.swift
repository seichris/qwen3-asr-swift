import SwiftUI
import Qwen3ASR
import Translation

@available(iOS 18.0, macOS 15.0, *)
struct ContentView: View {
    @StateObject private var vm = LiveTranslateViewModel()

    private struct RunTaskID: Equatable {
        var isRunning: Bool
        var provider: TranslationProvider
    }

    private enum TranslationProvider: String, CaseIterable, Identifiable, Equatable {
        case apple
        case googleCloud
        case off

        var id: String { rawValue }

        var displayName: String {
            switch self {
            case .apple: return "Apple"
            case .googleCloud: return "Google Cloud"
            case .off: return "Off"
            }
        }
    }

    @State private var from = SupportedLanguage.chinese
    @State private var to = SupportedLanguage.english
    @State private var translationProvider: TranslationProvider
    @State private var showSettings = false

    // Keeping this in state makes `.translationTask` restart when languages change.
    @State private var translationConfig: TranslationSession.Configuration? = .init(
        source: .init(identifier: SupportedLanguage.chinese.id),
        target: .init(identifier: SupportedLanguage.english.id)
    )

    private let modelIdDefault = "mlx-community/Qwen3-ASR-0.6B-4bit"

    init() {
        let key = ProcessInfo.processInfo.environment["QWEN3_ASR_GOOGLE_TRANSLATE_API_KEY"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let defaultProvider: TranslationProvider = (key?.isEmpty == false) ? .googleCloud : .apple
        _translationProvider = State(initialValue: defaultProvider)
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                languageBar
                transcriptPane
                micBar
            }
            .padding()
            .navigationTitle("Live Translate")
            .toolbar {
                #if os(iOS)
                ToolbarItem(placement: .topBarTrailing) { settingsButton }
                #else
                ToolbarItem(placement: .automatic) { settingsButton }
                #endif
            }
        }
        .sheet(isPresented: $showSettings) {
            SettingsView(vm: vm, modelId: modelIdDefault)
        }
        .onChange(of: from) { _, _ in
            rebuildTranslationConfig()
            if vm.isRunning { vm.requestStop() }
        }
        .onChange(of: to) { _, _ in
            rebuildTranslationConfig()
            if vm.isRunning { vm.requestStop() }
        }
        .onChange(of: translationProvider) { _, _ in
            if vm.isRunning { vm.requestStop() }
        }
        .translationTask((vm.isRunning && translationProvider == .apple) ? translationConfig : nil) { session in
            // Runs while active; cancelled automatically when `translationConfig` becomes nil.
            await vm.run(
                translationSession: session,
                modelId: modelIdDefault,
                from: from,
                to: to
            )
        }
        .task(id: RunTaskID(isRunning: vm.isRunning, provider: translationProvider)) {
            guard vm.isRunning else { return }
            switch translationProvider {
            case .off:
                await vm.runNoTranslation(modelId: modelIdDefault, from: from)
            case .googleCloud:
                await vm.runGoogleTranslation(modelId: modelIdDefault, from: from, to: to)
            case .apple:
                // Handled by `.translationTask`.
                break
            }
        }
    }

    private var settingsButton: some View {
        Button {
            showSettings = true
        } label: {
            Image(systemName: "gearshape")
        }
        .disabled(vm.isRunning)
        .accessibilityLabel("Settings")
    }

    private var languageBar: some View {
        HStack(spacing: 12) {
            Picker("From", selection: $from) {
                ForEach(SupportedLanguage.all) { lang in
                    Text(lang.displayName).tag(lang)
                }
            }
            .pickerStyle(.menu)
            .disabled(vm.isRunning)

            Button {
                let tmp = from
                from = to
                to = tmp
            } label: {
                Image(systemName: "arrow.left.arrow.right")
                    .font(.system(size: 18, weight: .semibold))
            }
            .buttonStyle(.bordered)
            .disabled(vm.isRunning)

            Picker("To", selection: $to) {
                ForEach(SupportedLanguage.all) { lang in
                    Text(lang.displayName).tag(lang)
                }
            }
            .pickerStyle(.menu)
            .disabled(vm.isRunning)

            Picker("Translate", selection: $translationProvider) {
                ForEach(TranslationProvider.allCases) { p in
                    Text(p.displayName).tag(p)
                }
            }
            .pickerStyle(.menu)
            .disabled(vm.isRunning)
        }
    }

    private var transcriptPane: some View {
        VStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 8) {
                Text("Transcript")
                    .font(.headline)
                if !vm.partialTranscript.isEmpty {
                    CopyableTextBox(
                        text: vm.partialTranscript,
                        selectable: false,
                        onDoubleClickCopy: false
                    )
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            Divider()

            ScrollView {
                LazyVStack(alignment: .leading, spacing: 12) {
                    ForEach(vm.segments) { seg in
                        VStack(alignment: .leading, spacing: 6) {
                            CopyableTextBox(text: seg.transcript)
                            if translationProvider != .off, let t = seg.translation, !t.isEmpty {
                                CopyableTextBox(text: t)
                            }
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
            }
        }
    }

    private var micBar: some View {
        HStack(spacing: 12) {
            statusView
                .frame(maxWidth: .infinity, alignment: .leading)

            Button {
                if vm.isRunning {
                    vm.requestStop()
                } else {
                    vm.start()
                }
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: vm.isRunning ? "stop.fill" : "mic.fill")
                    Text(vm.isStopping ? "Stopping…" : (vm.isRunning ? "Stop" : "Start"))
                }
                .font(.system(size: 16, weight: .semibold))
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
            }
            .buttonStyle(.borderedProminent)
            .disabled(vm.isStopping)
        }
    }

    @ViewBuilder
    private var statusView: some View {
        switch vm.status {
        case .idle:
            Text("Idle")
                .foregroundStyle(.secondary)
        case .loadingModel(let p, let s):
            VStack(alignment: .leading, spacing: 2) {
                Text("Loading model… \(Int(p * 100))%")
                Text(s).font(.caption).foregroundStyle(.secondary)
            }
        case .ready:
            Text("Ready")
                .foregroundStyle(.secondary)
        case .running:
            Text("Listening…")
                .foregroundStyle(.secondary)
        case .error(let msg):
            Text("Error: \(msg)")
                .foregroundStyle(.red)
        }
    }

    private func rebuildTranslationConfig() {
        translationConfig = .init(
            source: .init(identifier: from.id),
            target: .init(identifier: to.id)
        )
    }
}
