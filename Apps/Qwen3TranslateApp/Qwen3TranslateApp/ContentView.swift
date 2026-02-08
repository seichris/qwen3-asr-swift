import SwiftUI
import Qwen3ASR
import Translation

@available(iOS 18.0, macOS 15.0, *)
struct ContentView: View {
    @StateObject private var vm = LiveTranslateViewModel()

    @State private var from = SupportedLanguage.chinese
    @State private var to = SupportedLanguage.english

    // Keeping this in state makes `.translationTask` restart when languages change.
    @State private var translationConfig: TranslationSession.Configuration? = .init(
        source: .init(identifier: SupportedLanguage.chinese.id),
        target: .init(identifier: SupportedLanguage.english.id)
    )

    @State private var isRunning = false

    private let modelIdDefault = "mlx-community/Qwen3-ASR-0.6B-4bit"

    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                languageBar
                transcriptPane
                micBar
            }
            .padding()
            .navigationTitle("Live Translate")
        }
        .onChange(of: from) { _, _ in
            rebuildTranslationConfig()
            if isRunning { isRunning = false }
        }
        .onChange(of: to) { _, _ in
            rebuildTranslationConfig()
            if isRunning { isRunning = false }
        }
        .translationTask(isRunning ? translationConfig : nil) { session in
            // Runs while active; cancelled automatically when `translationConfig` becomes nil.
            vm.start(
                translationSession: session,
                modelId: modelIdDefault,
                from: from,
                to: to
            )
        }
    }

    private var languageBar: some View {
        HStack(spacing: 12) {
            Picker("From", selection: $from) {
                ForEach(SupportedLanguage.all) { lang in
                    Text(lang.displayName).tag(lang)
                }
            }
            .pickerStyle(.menu)

            Button {
                let tmp = from
                from = to
                to = tmp
            } label: {
                Image(systemName: "arrow.left.arrow.right")
                    .font(.system(size: 18, weight: .semibold))
            }
            .buttonStyle(.bordered)

            Picker("To", selection: $to) {
                ForEach(SupportedLanguage.all) { lang in
                    Text(lang.displayName).tag(lang)
                }
            }
            .pickerStyle(.menu)
        }
    }

    private var transcriptPane: some View {
        VStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 8) {
                Text("Transcript")
                    .font(.headline)
                if !vm.partialTranscript.isEmpty {
                    Text(vm.partialTranscript)
                        .font(.body)
                        .foregroundStyle(.secondary)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            Divider()

            ScrollView {
                LazyVStack(alignment: .leading, spacing: 12) {
                    ForEach(vm.segments) { seg in
                        VStack(alignment: .leading, spacing: 6) {
                            Text(seg.transcript)
                                .font(.body)
                            if let t = seg.translation, !t.isEmpty {
                                Text(t)
                                    .font(.body)
                                    .foregroundStyle(.secondary)
                            } else {
                                Text("…")
                                    .font(.body)
                                    .foregroundStyle(.tertiary)
                            }
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(12)
                        .background(.thinMaterial)
                        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
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
                isRunning.toggle()
                if !isRunning {
                    vm.stop()
                } else {
                    vm.clear()
                }
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: isRunning ? "stop.fill" : "mic.fill")
                    Text(isRunning ? "Stop" : "Start")
                }
                .font(.system(size: 16, weight: .semibold))
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
            }
            .buttonStyle(.borderedProminent)
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

