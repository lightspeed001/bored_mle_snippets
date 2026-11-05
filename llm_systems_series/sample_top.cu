// Maps: E12 (sampling) -> terminal DAG nodes -> T8 (SIMT, no Tensor Cores).
// Greedy is easy: one argmax reduction. Nucleus (top-p) needs a threshold.
// Sketch: find the probability mass cutoff with a parallel scan/sort.
__global__ void top_p_sample(const float* __restrict__ z,   // [V] FP32 logits
                             float temperature, float p,
                             int* __restrict__ out_token, int V) {
    // 1) softmax(z/T): max reduction -> exp -> sum reduction. V = 163,840.
    //    Two block-wide reductions plus a small device-level combine. The same
    //    SFU (MUFU.EX2) that bottlenecked FlashAttention shows up again here.
    // 2) sort probabilities (cub::DeviceSegmentedRadixSort) or, cheaper,
    //    binary-search a threshold tau such that sum_{p_i >= tau} p_i ≈ p.
    // 3) sample from the truncated distribution with a single RNG draw,
    //    then one atomic write of the chosen token id.
    //
    // WHY WE MENTION IT: this is ~163,840 FP32 values = 640 KB of logits read
    // for EVERY generated token, and it is entirely SIMT. A "model" article
    // that ends at the logits has skipped a real kernel on the critical path.
}
