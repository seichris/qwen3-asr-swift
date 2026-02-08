# Qwen3-ASR Swift

A Swift implementation of Qwen3-ASR speech recognition model using [MLX Swift](https://github.com/ml-explore/mlx-swift) for Apple Silicon.

## Overview

Qwen3-ASR is a state-of-the-art automatic speech recognition model from Alibaba/Qwen that offers:

- **52 languages**: 30 major languages + 22 Chinese dialects
- **Excellent noise robustness**: Outperforms Whisper and GPT-4o in noisy conditions
- **Fast inference**: 92ms TTFT, RTF 0.064 at high concurrency
- **On-device**: Runs locally on Apple Silicon Macs and iPhones

## Models

| Model | Parameters | Use Case |
|-------|-----------|----------|
| Qwen3-ASR-0.6B | 600M | Efficient, on-device |
| Qwen3-ASR-1.7B | 1.7B | Best accuracy |

## Installation

### Swift Package Manager

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/yourusername/qwen3-asr-swift", from: "0.1.0")
]
```

### Requirements

- macOS 14+ or iOS 17+
- Apple Silicon (M1/M2/M3/M4)
- Xcode 15+

## Run Locally (macOS)

### 1) Build MLX Metal Library (metallib)

If you see an error like `MLX error: Failed to load the default metallib ...`, generate `mlx.metallib` locally:

```bash
# Install Metal toolchain (once)
xcodebuild -downloadComponent MetalToolchain

# Build and generate mlx.metallib next to the SwiftPM output
swift build -c release --disable-sandbox
./scripts/build_mlx_metallib.sh release
```

### 2) (Optional) Configure Cache / Debug

The first run downloads model artifacts from Hugging Face and caches them under `~/Library/Caches/qwen3-asr/...`.

```bash
# Optional: override cache root (useful for CI/sandboxes).
# IMPORTANT: this must be a writable directory. Don't paste `/path/to/cache-root` literally.
# Examples:
#   export QWEN3_ASR_CACHE_DIR="$HOME/.cache"
#   export QWEN3_ASR_CACHE_DIR="$(mktemp -d)"
#
# Use the default cache location:
unset QWEN3_ASR_CACHE_DIR

# Optional: enable debug logging (very verbose; slows down inference)
export QWEN3_ASR_DEBUG=1

# Disable debug logging:
unset QWEN3_ASR_DEBUG
```

### 3) Build And Run The CLI

```bash
swift build -c release --disable-sandbox
.build/release/qwen3-asr-cli --help
```

Transcribe a file:

```bash
.build/release/qwen3-asr-cli transcribe /path/to/audio.wav
```

Realtime microphone transcription/translation:

```bash
# First time: macOS will prompt for microphone permission (grant it to your terminal app).
.build/release/qwen3-asr-cli realtime --to en --from auto

# Japanese translation with JSONL output
.build/release/qwen3-asr-cli realtime --to ja --window 15 --jsonl

# Transcription only (no translation)
.build/release/qwen3-asr-cli realtime --to en --no-translate
```

Notes:
- Realtime output is normalized to plain text (the library strips the model's `language ...<asr_text>` wrapper before emitting events).
- Translation (`[TRANS]`) is generated via text-only decoding on the same model (no extra service required).

Force CPU (slow; mostly for debugging):

```bash
export QWEN3_ASR_DEVICE=cpu
.build/release/qwen3-asr-cli transcribe /path/to/audio.wav
```

## Usage

### Basic Transcription

```swift
import Qwen3ASR

// Load model
let model = try await Qwen3ASRModel.fromPretrained(modelId: "Qwen/Qwen3-ASR-0.6B")

// Transcribe audio (24kHz mono float samples)
let transcription = model.transcribe(audio: audioSamples, sampleRate: 24000)
print(transcription)
```

### Realtime Translation

```swift
import Qwen3ASR

// Create audio source (microphone)
let audioSource = MicrophoneAudioSource(frameSizeMs: 20)

// Configure options
let options = RealtimeTranslationOptions(
	    targetLanguage: "en",
	    sourceLanguage: nil,
	    windowSeconds: 10.0,
	    stepMs: 500,
	    enableVAD: true
	)

// Start realtime translation
let stream = await model.realtimeTranslate(
    audioSource: audioSource,
    options: options
)

for await event in stream {
    switch event.kind {
    case .partial:
        print("Transcribing: \(event.transcript)")
    case .final:
        print("Final: \(event.transcript)")
    case .translation:
        print("Translation: \(event.translation ?? "")")
    default: break
    }
}
```

### CLI Tool

```bash
# Build CLI
swift build -c release

# Transcribe audio file
qwen3-asr-cli transcribe audio.wav

# Realtime microphone translation
qwen3-asr-cli realtime --to en --from auto

# Japanese translation with JSONL output
qwen3-asr-cli realtime --to ja --window 15 --jsonl

# Transcription only (no translation)
qwen3-asr-cli realtime --to en --no-translate
```

## Architecture

```
Audio Input (24kHz)
    │
    ▼
┌─────────────────┐
│  Mel Spectrogram│  (WhisperFeatureExtractor)
│  128 bins, 8ms  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Audio Encoder  │  (Conv2D + Transformer)
│  12/18 layers   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Projector     │  (2-layer MLP)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Text Decoder   │  (Qwen3 LLM)
│  28 layers      │
└────────┬────────┘
         │
         ▼
     Text Output
```

## Performance

### Benchmarks (Apple M3 Max)

| Model | TTFT | RTF |
|-------|------|-----|
| Qwen3-ASR-0.6B | ~100ms | ~0.08 |
| Qwen3-ASR-1.7B | ~200ms | ~0.15 |

### Word Error Rate

| Model | LibriSpeech (clean) | Noisy Conditions |
|-------|---------------------|------------------|
| Qwen3-ASR-0.6B | 2.11% | 17.88% |
| Qwen3-ASR-1.7B | 1.63% | 16.17% |
| Whisper-large-v3 | 1.51% | 63.17% |

## Supported Languages

### Major Languages (30)
Chinese, English, Cantonese, Arabic, German, French, Spanish, Portuguese, Indonesian, Italian, Korean, Russian, Thai, Vietnamese, Japanese, Turkish, Hindi, Malay, Dutch, Swedish, Danish, Finnish, Polish, Czech, Filipino, Persian, Greek, Hungarian, Macedonian, Romanian

### Chinese Dialects (22)
Anhui, Dongbei, Fujian, Gansu, Guizhou, Hebei, Henan, Hubei, Hunan, Jiangxi, Ningxia, Shandong, Shaanxi, Shanxi, Sichuan, Tianjin, Yunnan, Zhejiang, Cantonese (HK/Guangdong), Wu, Minnan

## Development Status

- [x] Configuration classes
- [x] Audio encoder (Conv2D + Transformer)
- [x] Text decoder (Qwen3)
- [x] Audio preprocessing (Mel spectrogram)
- [x] Weight loading infrastructure
- [ ] HuggingFace model download
- [ ] Tokenizer integration
- [ ] Streaming inference optimization
- [ ] iOS support

## License

Apache 2.0 (same as original Qwen3-ASR)

## Credits

- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) - Original model by Alibaba/Qwen
- [MLX Swift](https://github.com/ml-explore/mlx-swift) - Apple's ML framework for Swift
- [mlx-audio](https://github.com/ml-explore/mlx-audio) - Reference Python implementation
