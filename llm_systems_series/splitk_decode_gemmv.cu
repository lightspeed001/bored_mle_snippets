// Maps: T1 with M=1 -> DAG node g3 -> E3/E9, batch-1 decode.
// Problem: only 12 output CTAs for a 1536-wide output. 148 SMs available.
// Solution: split the K reduction into SPLITS chunks, then combine partials.
constexpr int SPLITS = 16;               // 7168 / 16 = 448 K-elements each
constexpr int N_TILES = 12;              // ceil(1536/128)
// grid = 12 x 16 = 192 CTAs  ->  1.3 waves, ~all SMs busy. Better.

__global__ void splitk_gemv(const __nv_fp8_e4m3* W, const half* x,
                            float* partial, int K, int N) {
    int n_tile = blockIdx.x, split = blockIdx.y;
    int k0 = split * (K / SPLITS), k1 = k0 + (K / SPLITS);

    // Each CTA owns a [128 output cols] x [448 K-rows] slice. Accumulate in
    // TMEM (or registers for this narrow shape), dequantising FP8 block scales.
    float acc[128] = {0.f};
    for (int k = k0; k < k1; k += 128) {
        fp8_tile_t a = tma_load(W, n_tile, k);              // 128x128 FP8
        half_tile_t b = tma_load(x, k);                     // 1x128 BF16
        tcgen05_mma_accumulate(acc, a, b, scale[k >> 7]);  // FP32 accum
    }
    // Write one partial vector per (n_tile, split). The second kernel reduces
    // 16 partials -> 1 output. That reduction is a NEW edge in the physical
    // graph that does NOT exist in the mathematical DAG: it is an artefact of
    // parallelism, not of the function being computed.
    store_partial(partial, n_tile, split, acc);
}

// NOTE THE ASYMMETRY: split-K adds traffic (16 x partial vectors, FP32) and a
// second kernel launch. At batch 1 that is worth it. At batch 256 it is
// actively harmful -- batching already fills the SMs. The right kernel depends
// on a SERVING decision (L5 scheduler), not on the equation. Remember this
// when someone asks why vLLM has six different GEMM paths for one operation.
