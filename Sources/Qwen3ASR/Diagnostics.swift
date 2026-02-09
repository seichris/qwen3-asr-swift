import Foundation
import MLX
import MLXNN

public enum Qwen3ASRDiagnostics {
    /// Enable via `QWEN3_ASR_DIAGNOSTICS=1`.
    public static var enabled: Bool = {
        let raw = ProcessInfo.processInfo.environment["QWEN3_ASR_DIAGNOSTICS"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        switch raw {
        case "1", "true", "yes", "y", "on":
            return true
        default:
            return false
        }
    }()

    /// Basic correctness check for MLX's 4-bit `quantizedMM` implementation.
    ///
    /// On some iOS devices/OS versions, the Metal backend for quantized matmul can misbehave,
    /// which would produce completely nonsensical ASR outputs (random token IDs).
    ///
    /// This test compares:
    /// - `quantizedMM(x, wq, ...)`
    /// - `matmul(x, dequantized(wq, ...).T)`
    ///
    /// It prints max/mean absolute error. If errors are large (e.g. > 1e-2), quantized kernels are suspect.
    public static func runQuantizedMMSanityCheck() {
        guard enabled else { return }

        let env = ProcessInfo.processInfo.environment
        let rawIters = env["QWEN3_ASR_DIAGNOSTICS_ITERS"]?.trimmingCharacters(in: .whitespacesAndNewlines)
        let iters = max(1, Int(rawIters ?? "") ?? 3)

        // Use small sizes to keep this fast and avoid huge allocations.
        let batch = 2
        let inDim = 128
        let outDim = 256
        let groupSize = 64
        let bits = 4

        print("Qwen3ASRDiagnostics: running quantizedMM sanity (iters=\(iters))")

        for i in 0..<iters {
            // Generate FP32 reference weights, then quantize using MLX so formats match the backend expectation.
            let x = MLXRandom.normal([batch, inDim]).asType(.float32)
            let w = MLXRandom.normal([outDim, inDim]).asType(.float32)
            let (wq, scales, biases) = MLX.quantized(w, groupSize: groupSize, bits: bits, mode: .affine)

            let yQuant = quantizedMM(
                x, wq,
                scales: scales,
                biases: biases,
                transpose: true,
                groupSize: groupSize,
                bits: bits,
                mode: .affine
            ).asType(.float32)

            let wDeq = dequantized(
                wq,
                scales: scales,
                biases: biases,
                groupSize: groupSize,
                bits: bits,
                mode: .affine,
                dtype: .float32
            )
            let yRef = matmul(x, wDeq.transposed(1, 0)).asType(.float32)

            let diff = abs(yQuant - yRef).flattened()
            let maxAbs = max(diff).item(Float.self)
            let meanAbs = mean(diff).item(Float.self)
            print("Qwen3ASRDiagnostics: iter=\(i) max_abs=\(maxAbs) mean_abs=\(meanAbs)")
        }
    }
}
