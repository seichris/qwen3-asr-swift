import SwiftUI

#if os(macOS)
import AppKit
#elseif os(iOS)
import UIKit
#endif

struct CopyableTextBox: View {
    let text: String
    var selectable: Bool = true
    var onDoubleClickCopy: Bool = true

    var body: some View {
        Text(text)
            .font(.body)
            .textSelection(selectable ? .enabled : .disabled)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(10)
            .background(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(.thinMaterial)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .stroke(.quaternary, lineWidth: 1)
            )
            .contextMenu {
                Button("Copy") { copyToPasteboard(text) }
            }
            .onTapGesture(count: 2) {
                guard onDoubleClickCopy else { return }
                copyToPasteboard(text)
            }
    }

    private func copyToPasteboard(_ s: String) {
        let trimmed = s.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        #if os(macOS)
        let pb = NSPasteboard.general
        pb.clearContents()
        pb.setString(trimmed, forType: .string)
        #elseif os(iOS)
        UIPasteboard.general.string = trimmed
        #endif
    }
}
