import Foundation

/// Manages transcript stability and prefix commitment
public actor TranscriptStabilizer {
    /// Configuration for stabilization
    public struct Config {
        public let stabilityThreshold: Int  // Number of consecutive matches to consider stable
        public let minPrefixLength: Int     // Minimum characters to commit
        
        public init(
            stabilityThreshold: Int = 3,
            minPrefixLength: Int = 5
        ) {
            self.stabilityThreshold = stabilityThreshold
            self.minPrefixLength = minPrefixLength
        }
    }
    
    /// Represents a transcript state
    public struct State {
        public let committed: String   // Stable, committed text
        public let pending: String     // Current best guess (may change)
        public let isStable: Bool      // Whether pending has stabilized
        public let newlyCommitted: String // The segment committed by this update, if any
        
        public init(committed: String = "", pending: String = "", isStable: Bool = false, newlyCommitted: String = "") {
            self.committed = committed
            self.pending = pending
            self.isStable = isStable
            self.newlyCommitted = newlyCommitted
        }
    }
    
    private let config: Config
    private var committedText: String = ""
    private var pendingText: String = ""
    private var matchCount: Int = 0
    private var lastPending: String = ""
    
    public init(config: Config = Config()) {
        self.config = config
    }
    
    /// Update with a new transcript from the model
    /// - Parameter transcript: New transcript from current audio window
    /// - Returns: Updated state with committed and pending text
    public func update(transcript: String) -> State {
        let trimmed = transcript.trimmingCharacters(in: .whitespacesAndNewlines)

        // Best-effort alignment between the sliding-window transcript and our committed prefix.
        // If the window has moved and no longer includes the full committed text, we keep the
        // committed text and only append the non-overlapping suffix.
        let newContent: String
        if trimmed.hasPrefix(committedText) {
            newContent = String(trimmed.dropFirst(committedText.count))
        } else {
            let overlap = longestSuffixPrefixOverlap(committedText, trimmed)
            newContent = String(trimmed.dropFirst(overlap))
        }
        
        // Check if pending text is stabilizing
        if newContent == lastPending {
            matchCount += 1
        } else {
            matchCount = 0
            lastPending = newContent
        }
        
        // Commit if stable and meets minimum length
        let isStable = matchCount >= config.stabilityThreshold
        var newlyCommitted = ""
        if isStable && newContent.count >= config.minPrefixLength {
            // Commit the stable portion (append-only).
            committedText += newContent
            pendingText = ""
            lastPending = ""
            matchCount = 0
            newlyCommitted = newContent
        } else {
            pendingText = newContent
        }
        
        return State(
            committed: committedText,
            pending: pendingText,
            isStable: isStable,
            newlyCommitted: newlyCommitted
        )
    }
    
    /// Force commit all pending text (e.g., on silence/VAD)
    public func forceCommit() -> State {
        var newlyCommitted = ""
        if !pendingText.isEmpty {
            committedText += pendingText
            newlyCommitted = pendingText
            pendingText = ""
            lastPending = ""
            matchCount = 0
        }
        return State(
            committed: committedText,
            pending: "",
            isStable: true,
            newlyCommitted: newlyCommitted
        )
    }
    
    /// Reset the stabilizer (e.g., on new utterance)
    public func reset() {
        committedText = ""
        pendingText = ""
        lastPending = ""
        matchCount = 0
    }
    
    /// Get current state without modifying
    public func currentState() -> State {
        State(
            committed: committedText,
            pending: pendingText,
            isStable: matchCount >= config.stabilityThreshold,
            newlyCommitted: ""
        )
    }
    
    /// Find the longest overlap where a suffix of `committed` matches a prefix of `candidate`.
    private func longestSuffixPrefixOverlap(_ committed: String, _ candidate: String) -> Int {
        guard !committed.isEmpty, !candidate.isEmpty else { return 0 }

        let c1 = Array(committed)
        let c2 = Array(candidate)
        let maxLen = min(c1.count, c2.count)

        // Try longer overlaps first.
        for len in stride(from: maxLen, to: 0, by: -1) {
            let suffixStart = c1.count - len
            var ok = true
            for i in 0..<len {
                if c1[suffixStart + i] != c2[i] {
                    ok = false
                    break
                }
            }
            if ok { return len }
        }
        return 0
    }
}

/// Simple Voice Activity Detection (VAD)
public actor SimpleVAD {
    public struct Config {
        public let energyThreshold: Float  // RMS energy threshold
        public let silenceDurationMs: Int  // Duration to consider end of utterance
        public let minSpeechDurationMs: Int // Minimum speech duration
        
        public init(
            energyThreshold: Float = 0.01,
            silenceDurationMs: Int = 1000,
            minSpeechDurationMs: Int = 200
        ) {
            self.energyThreshold = energyThreshold
            self.silenceDurationMs = silenceDurationMs
            self.minSpeechDurationMs = minSpeechDurationMs
        }
    }
    
    public enum State: Equatable {
        case silence
        case speech(startTime: UInt64)
        case speechEnd(durationMs: Int)
    }
    
    private let config: Config
    private var state: State = .silence
    private var silenceStartTime: UInt64?
    private var lastSpeechEndTime: UInt64 = 0
    private let timebaseInfo: mach_timebase_info_data_t
    
    public init(config: Config = Config()) {
        self.config = config
        var info = mach_timebase_info_data_t()
        mach_timebase_info(&info)
        self.timebaseInfo = info
    }
    
    /// Process audio frame and return VAD state
    /// - Parameter frame: Audio samples
    /// - Returns: Current VAD state
    public func process(frame: [Float]) -> State {
        let energy = calculateRMS(frame)
        let currentTime = mach_absolute_time()
        
        switch state {
        case .silence:
            if energy > config.energyThreshold {
                state = .speech(startTime: currentTime)
            }
            
        case .speech(let startTime):
            if energy < config.energyThreshold {
                if silenceStartTime == nil {
                    silenceStartTime = currentTime
                } else {
                    let silenceDuration = elapsedMs(from: silenceStartTime!, to: currentTime)
                    if silenceDuration >= config.silenceDurationMs {
                        let speechDuration = elapsedMs(from: startTime, to: silenceStartTime!)
                        if speechDuration >= config.minSpeechDurationMs {
                            state = .speechEnd(durationMs: speechDuration)
                            lastSpeechEndTime = currentTime
                        } else {
                            state = .silence
                        }
                        silenceStartTime = nil
                    }
                }
            } else {
                silenceStartTime = nil
            }
            
        case .speechEnd:
            // Reset to silence after reporting end
            state = .silence
        }
        
        return state
    }
    
    /// Calculate RMS energy of audio frame
    private func calculateRMS(_ frame: [Float]) -> Float {
        guard !frame.isEmpty else { return 0 }
        var sum: Float = 0
        for x in frame {
            sum += x * x
        }
        return sqrt(sum / Float(frame.count))
    }
    
    /// Convert mach absolute time to milliseconds elapsed
    private func elapsedMs(from start: UInt64, to end: UInt64) -> Int {
        let elapsed = end - start
        let nanoseconds = elapsed * UInt64(timebaseInfo.numer) / UInt64(timebaseInfo.denom)
        return Int(nanoseconds / 1_000_000)
    }
    
    public func reset() {
        state = .silence
        silenceStartTime = nil
    }
}
