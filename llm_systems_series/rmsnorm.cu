// Maps: E2 (RMSNorm) -> DAG node g2 -> SIMT on CUDA cores / SFU.
// One CTA per token-row. d = 7168, 256 threads => 28 elements per thread.
// In a real stack (vLLM / TRT-LLM / CUTLASS) this is FUSED into the previous
// GEMM's epilogue, so h never takes an HBM round-trip. We write it standalone
// here because the reduction is the part worth seeing.
__global__ void rmsnorm(const half*  __restrict__ x,       // [N, d] BF16
                        const half*  __restrict__ gamma,   // [d] learned gain
                        half*        __restrict__ h,       // [N, d]
                        int d) {
    extern __shared__ float warp_ss[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const half2* xv = reinterpret_cast<const half2*>(x) + blockIdx.x * (d/2);

    // 1) Sum of squares. half2 loads keep the HBM/L1 transactions coalesced:
    //    one 32-lane warp moves 32 * 4 B = 128 B per instruction.
    float ss = 0.f;
    for (int i = tid; i < d/2; i += blockDim.x) {
        float2 f = __half22float2(xv[i]);
        ss = fmaf(f.x, f.x, fmaf(f.y, f.y, ss));   // 2 FMA + 1 add / iteration
    }

    // 2) Warp-level reduction. No shared memory, no barrier: shuffle only.
    for (int off = 16; off; off >>= 1)
        ss += __shfl_down_sync(0xffffffffu, ss, off);
    if (lane == 0) warp_ss[warp] = ss;             // 256 threads = 8 warps
    __syncthreads();                               // the ONE synchronisation

    // 3) Cross-warp reduction, then rsqrt on the SFU (not the Tensor Core).
    float total = (tid < 8) ? warp_ss[tid] : 0.f;
    for (int off = 4; off; off >>= 1)
        total += __shfl_down_sync(0xffffffffu, total, off);
    __shared__ float inv;
    if (tid == 0) inv = rsqrtf(total / d + 1e-6f);
    __syncthreads();

    // 4) Normalise + gain + BF16 store. This is the whole node g2.
    const half2* gv = reinterpret_cast<const half2*>(gamma);
    half2* out = reinterpret_cast<half2*>(h) + blockIdx.x * (d/2);
    for (int i = tid; i < d/2; i += blockDim.x) {
        float2 f = __half22float2(xv[i]), w = __half22float2(gv[i]);
        f.x *= inv * w.x;  f.y *= inv * w.y;
        out[i] = __float22half2_rn(f);
    }
}
