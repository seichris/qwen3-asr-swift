import Foundation

/// A fixed-capacity ring buffer that keeps only the newest `capacity` samples.
struct FloatRingBuffer: Sendable {
    private var storage: [Float]
    private var head: Int = 0
    private var count: Int = 0

    var capacity: Int { storage.count }
    var isEmpty: Bool { count == 0 }
    var size: Int { count }

    init(capacity: Int) {
        let cap = max(1, capacity)
        self.storage = [Float](repeating: 0, count: cap)
    }

    mutating func reset() {
        head = 0
        count = 0
    }

    mutating func append(contentsOf samples: [Float]) {
        guard !samples.isEmpty else { return }
        for s in samples {
            append(s)
        }
    }

    mutating func append(_ sample: Float) {
        if count < storage.count {
            storage[(head + count) % storage.count] = sample
            count += 1
        } else {
            storage[head] = sample
            head = (head + 1) % storage.count
        }
    }

    func toArray() -> [Float] {
        guard count > 0 else { return [] }
        if head + count <= storage.count {
            return Array(storage[head..<(head + count)])
        }
        let firstCount = storage.count - head
        var out: [Float] = []
        out.reserveCapacity(count)
        out.append(contentsOf: storage[head..<storage.count])
        out.append(contentsOf: storage[0..<(count - firstCount)])
        return out
    }
}

