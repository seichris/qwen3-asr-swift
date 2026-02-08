# Feature Plan: Cross-Platform Translate App (iPhone + Mac)

## Goal

Build a SwiftUI app that runs on:
- iPhone (iOS)
- Mac (native macOS target, not Catalyst)

The app provides realtime transcription + translation with a UI similar to Google Translate / Apple Translate:
- Choose source language (ASR hint)
- Choose target language (translation output)
- Mic button to start/stop live capture
- Live transcript (partial + final)
- Live translation (final segments translated)

Default behavior:
- Source: Chinese
- Target: English

## Why An App Is Required For Real Translation

The ASR model is good at transcription but unreliable for “speech translation” just by forcing `language English`.
For reliable translation we use Apple’s `Translation` framework, which is app-hosted (SwiftUI `.translationTask(...)` provides a `TranslationSession`).

## Scope / Non-Goals

- CLI translation improvements are out of scope here (CLI cannot reliably obtain a `TranslationSession`).
- No model fine-tuning or prompt engineering.

## Supported Languages In UI (Intersection)

We will offer a conservative set of major languages that:
- Qwen3-ASR supports (per repo README language list)
- Apple Translation is likely to support on-device

Initial picker set (can expand later):
- Chinese (zh)
- English (en)
- Japanese (ja)
- Korean (ko)
- French (fr)
- German (de)
- Spanish (es)
- Italian (it)
- Portuguese (pt)
- Russian (ru)
- Arabic (ar)
- Hindi (hi)

Notes:
- Apple Translation availability depends on the OS + installed language packs; translation can fail even if a language is in the picker.
- We’ll surface translation errors non-fatally in the UI (keep transcription running).

## Architecture

**UI**
- `ContentView`: language pickers, swap, mic button, transcript + translation panes, segment history.

**View model**
- `LiveTranslateViewModel` (`@MainActor`, `ObservableObject`)
- Loads `Qwen3ASRModel` once and reuses it between runs.
- Starts microphone capture using `MicrophoneAudioSource`.
- Streams events from `Qwen3ASRModel.realtimeTranslate(...)`.
- Uses Apple Translation wrapper (`realtimeTranslate(..., translationSession:)`) to emit `.translation` events for `.final` segments.

**Library integration**
- Use existing realtime pipeline in `Sources/Qwen3ASR/RealtimeTranslation.swift`.
- Use Apple Translation wrapper in `Sources/Qwen3ASR/RealtimeTranslationApple.swift` (requires iOS 18 / macOS 15 SDK availability).

## Project Setup

- Add `Apps/Qwen3TranslateApp/` Xcode project:
  - iOS app target (deployment: iOS 18.0)
  - macOS app target (deployment: macOS 15.0)
  - Depends on local Swift package product `Qwen3ASR`
- Provide `Info.plist` with `NSMicrophoneUsageDescription`.

## Implementation Steps

1. Add app source files:
   - `Apps/Qwen3TranslateApp/Qwen3TranslateApp/ContentView.swift`
   - `Apps/Qwen3TranslateApp/Qwen3TranslateApp/LiveTranslateViewModel.swift`
   - `Apps/Qwen3TranslateApp/Qwen3TranslateApp/SupportedLanguage.swift`
   - `Apps/Qwen3TranslateApp/Qwen3TranslateApp/Qwen3TranslateAppApp.swift`
2. Add minimal `Info.plist` and assets.
3. Wire translation session:
   - Use SwiftUI `.translationTask(...)` to obtain `TranslationSession` and run the streaming pipeline while active.
4. Add UX polish:
   - Swap languages
   - Clear button
   - Status line (loading model / listening / error)
5. Validate build:
   - `xcodebuild` with codesigning disabled (CI-friendly)

## Acceptance Criteria

- On iPhone: mic capture works, transcript updates live, final segments translate to English by default.
- On Mac (macOS): same behavior; prompts for mic permission.
- Language pickers work; changing languages restarts translation session and stream cleanly.
