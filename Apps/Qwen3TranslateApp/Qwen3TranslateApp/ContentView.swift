import SwiftUI
import Qwen3ASR
import Translation

@available(iOS 18.0, macOS 15.0, *)
struct ContentView: View {
    @StateObject private var vm = LiveTranslateViewModel()

#if os(iOS)
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass
#endif

    private struct RunTaskID: Equatable {
        var isRunning: Bool
        var asrProvider: ASRProvider
        var translationProvider: TranslationProvider
        var inputSource: RealtimeInputAudioSource
    }

    private enum ASRProvider: String, CaseIterable, Identifiable, Equatable {
        case local
        case dashScopeMainland
        case dashScopeSingapore

        var id: String { rawValue }

        var displayName: String {
            switch self {
            case .local: return "Local ASR"
            case .dashScopeMainland: return "Alibaba Mainland"
            case .dashScopeSingapore: return "Alibaba Singapore"
            }
        }

        var dashScopeWorkspace: DashScopeWorkspace? {
            switch self {
            case .local:
                return nil
            case .dashScopeMainland:
                return .mainland
            case .dashScopeSingapore:
                return .singapore
            }
        }
    }

    private enum TranslationProvider: String, CaseIterable, Identifiable, Equatable {
        case apple
        case googleCloud
        case off

        var id: String { rawValue }

        var displayName: String {
            switch self {
            case .apple: return "Apple Local Translate"
            case .googleCloud: return "Google Cloud Translate"
            case .off: return "Off"
            }
        }
    }

    @State private var from = SupportedLanguage.chinese
    @State private var to = SupportedLanguage.english
    @State private var asrProvider: ASRProvider
    @State private var inputSource: RealtimeInputAudioSource = .microphone
    @State private var translationProvider: TranslationProvider
    @State private var showSettings = false

    // Keeping this in state makes `.translationTask` restart when languages change.
    @State private var translationConfig: TranslationSession.Configuration? = .init(
        source: .init(identifier: SupportedLanguage.chinese.id),
        target: .init(identifier: SupportedLanguage.english.id)
    )

    private let modelIdDefault = "mlx-community/Qwen3-ASR-0.6B-4bit"

    init() {
        let env = ProcessInfo.processInfo.environment
        func readCredential(_ key: String) -> String? {
            let envValue = env[key]?.trimmingCharacters(in: .whitespacesAndNewlines)
            if let envValue, !envValue.isEmpty { return envValue }

            let storedValue = UserDefaults.standard.string(forKey: key)?
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if let storedValue, !storedValue.isEmpty { return storedValue }

            return nil
        }

        let dashScopeKey = readCredential(DashScopeRealtimeClient.apiKeyEnvironmentVariable)
        let googleKey = readCredential("QWEN3_ASR_GOOGLE_TRANSLATE_API_KEY")
        let dashScopeKeySG = readCredential("DASHSCOPE_API_KEY_SG")
        let defaultASRProvider: ASRProvider = {
            if dashScopeKeySG?.isEmpty == false { return .dashScopeSingapore }
            if dashScopeKey?.isEmpty == false { return .dashScopeMainland }
            return .local
        }()
        let defaultTranslationProvider: TranslationProvider = {
            if googleKey?.isEmpty == false { return .googleCloud }
            return .apple
        }()
        _asrProvider = State(initialValue: defaultASRProvider)
        _translationProvider = State(initialValue: defaultTranslationProvider)
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                languageBar
                transcriptPane
                micBar
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .navigationTitle("Live Translate")
            #if os(iOS)
            .toolbarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                #if os(iOS)
                ToolbarItem(placement: .topBarTrailing) { settingsButton }
                #else
                ToolbarItem(placement: .automatic) { settingsButton }
                #endif
            }
        }
        .sheet(isPresented: $showSettings) {
            SettingsView(vm: vm, modelId: modelIdDefault, from: from)
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
        .onChange(of: asrProvider) { _, _ in
            if vm.isRunning { vm.requestStop() }
        }
        .onChange(of: inputSource) { _, _ in
            if vm.isRunning { vm.requestStop() }
        }
        .translationTask((vm.isRunning && translationProvider == .apple) ? translationConfig : nil) { session in
            // Runs while active; cancelled automatically when `translationConfig` becomes nil.
            switch asrProvider {
            case .local:
                await vm.run(
                    translationSession: session,
                    modelId: modelIdDefault,
                    from: from,
                    to: to,
                    inputSource: inputSource
                )
            case .dashScopeMainland, .dashScopeSingapore:
                guard let workspace = asrProvider.dashScopeWorkspace else { return }
                await vm.runDashScopeHosted(
                    translationSession: session,
                    from: from,
                    to: to,
                    workspace: workspace,
                    inputSource: inputSource
                )
            }
        }
        .task(id: RunTaskID(isRunning: vm.isRunning, asrProvider: asrProvider, translationProvider: translationProvider, inputSource: inputSource)) {
            guard vm.isRunning else { return }

            switch asrProvider {
            case .local:
                switch translationProvider {
                case .off:
                    await vm.runNoTranslation(modelId: modelIdDefault, from: from, inputSource: inputSource)
                case .googleCloud:
                    await vm.runGoogleTranslation(modelId: modelIdDefault, from: from, to: to, inputSource: inputSource)
                case .apple:
                    // Handled by `.translationTask`.
                    break
                }
            case .dashScopeMainland, .dashScopeSingapore:
                guard let workspace = asrProvider.dashScopeWorkspace else { return }
                switch translationProvider {
                case .off:
                    await vm.runDashScopeHosted(from: from, workspace: workspace, inputSource: inputSource)
                case .googleCloud:
                    await vm.runDashScopeHostedGoogle(from: from, to: to, workspace: workspace, inputSource: inputSource)
                case .apple:
                    // Handled by `.translationTask`.
                    break
                }
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
        Group {
            #if os(iOS)
            if horizontalSizeClass == .compact {
                VStack(spacing: 10) {
                    HStack(spacing: 10) {
                        fromPicker
                        swapButton
                        toPicker
                    }
                    HStack(spacing: 10) {
                        asrProviderPicker
                        inputSourcePicker
                        translationProviderPicker
                    }
                }
            } else {
                HStack(spacing: 12) {
                    fromPicker
                    swapButton
                    toPicker
                    asrProviderPicker
                    inputSourcePicker
                    translationProviderPicker
                }
            }
            #else
            HStack(spacing: 12) {
                fromPicker
                swapButton
                toPicker
                asrProviderPicker
                inputSourcePicker
                translationProviderPicker
            }
            #endif
        }
    }

    private var fromPicker: some View {
        Picker("From", selection: $from) {
            ForEach(SupportedLanguage.sources) { lang in
                Text(lang.displayName)
                    .lineLimit(1)
                    .truncationMode(.tail)
                    .tag(lang)
            }
        }
        .pickerStyle(.menu)
        .labelsHidden()
        .frame(maxWidth: .infinity, alignment: .leading)
        .disabled(vm.isRunning)
    }

    private var toPicker: some View {
        Picker("To", selection: $to) {
            ForEach(SupportedLanguage.targets) { lang in
                Text(lang.displayName)
                    .lineLimit(1)
                    .truncationMode(.tail)
                    .tag(lang)
            }
        }
        .pickerStyle(.menu)
        .labelsHidden()
        .frame(maxWidth: .infinity, alignment: .leading)
        .disabled(vm.isRunning)
    }

    private var asrProviderPicker: some View {
        Picker("ASR", selection: $asrProvider) {
            ForEach(ASRProvider.allCases) { p in
                Text(p.displayName)
                    .lineLimit(1)
                    .truncationMode(.tail)
                    .tag(p)
            }
        }
        .pickerStyle(.menu)
        .labelsHidden()
        .frame(maxWidth: .infinity, alignment: .leading)
        .disabled(vm.isRunning)
    }

    private var translationProviderPicker: some View {
        Picker("Translate", selection: $translationProvider) {
            ForEach(TranslationProvider.allCases) { p in
                Text(p.displayName)
                    .lineLimit(1)
                    .truncationMode(.tail)
                    .tag(p)
            }
        }
        .pickerStyle(.menu)
        .labelsHidden()
        .frame(maxWidth: .infinity, alignment: .leading)
        .disabled(vm.isRunning)
    }

    private var inputSourcePicker: some View {
        Picker("Audio", selection: $inputSource) {
            ForEach(availableInputSources) { source in
                Text(source.displayName)
                    .lineLimit(1)
                    .truncationMode(.tail)
                    .tag(source)
            }
        }
        .pickerStyle(.menu)
        .labelsHidden()
        .frame(maxWidth: .infinity, alignment: .leading)
        .disabled(vm.isRunning)
    }

    private var swapButton: some View {
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
    }

    private var transcriptPane: some View {
        VStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 8) {
                Text("Transcript")
                    .font(.headline)
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            ScrollView {
                if combinedTranscriptText.isEmpty {
                    Text("No transcript yet.")
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.vertical, 8)
                } else {
                    CopyableTextBox(text: combinedTranscriptText)
                }
            }
        }
    }

    private var combinedTranscriptText: String {
        var rows: [String] = []

        for seg in vm.segments {
            let transcript = seg.transcript.trimmingCharacters(in: .whitespacesAndNewlines)
            if !transcript.isEmpty {
                rows.append(transcript)
            }

            if translationProvider != .off {
                let translation = seg.translation?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                if !translation.isEmpty {
                    rows.append(translation)
                }
            }
        }

        let partial = vm.partialTranscript.trimmingCharacters(in: .whitespacesAndNewlines)
        if !partial.isEmpty {
            rows.append(partial)
        }

        return rows.joined(separator: "\n")
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

    private var availableInputSources: [RealtimeInputAudioSource] {
        #if os(iOS) && canImport(ReplayKit)
        return RealtimeInputAudioSource.allCases
        #else
        return [.microphone]
        #endif
    }
}
