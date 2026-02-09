import Foundation

#if canImport(Darwin)
import Darwin
#endif

enum Qwen3ASRRuntimeMetrics {
    /// Best-effort resident memory (MB). Useful for spotting leaks / runaway allocations in realtime.
    static func residentMemoryMB() -> Int? {
        #if canImport(Darwin)
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<integer_t>.size)
        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) { ptr in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
                task_info(
                    mach_task_self_,
                    task_flavor_t(MACH_TASK_BASIC_INFO),
                    rebound,
                    &count
                )
            }
        }
        guard kerr == KERN_SUCCESS else { return nil }
        return Int(info.resident_size / (1024 * 1024))
        #else
        return nil
        #endif
    }
}

