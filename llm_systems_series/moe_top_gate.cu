// Maps: E11 (router) -> DAG node g10 -> T6 on CUDA cores (SIMT).
// Kimi K2: E = 384 experts, top-8, SIGMOID scoring (not softmax), an additive
// per-expert bias b[e] for auxiliary-loss-free balancing, and alpha = 2.827.
// One CTA per token; 384 threads => one thread per expert.

__global__ void moe_gate(const half*  __restrict__ u,        // [d] BF16
                         const half*  __restrict__ Wr,       // [d x E], usually BF16
                         const float* __restrict__ bias,     // [E] learned
                         float*       __restrict__ score,    // [E] scratch in SMEM
                         int*         __restrict__ top_ids,  // [8]
                         float*       __restrict__ top_w) {  // [8]
    extern __shared__ float ss[];
    const int e = threadIdx.x;

    // (a) The 7168-d dot product is a GEMV, NOT a scalar loop. In production
    //     this is one tcgen05 GEMV or a cuBLAS call whose result feeds this
    //     kernel. Collapsed here so we can stare at the top-k instead.
    const float logit = gemv_dot(u, Wr, e);                  // [1 x d] x [d x E]

    // (b) Sigmoid scoring + additive correction bias. Kimi's "noaux_tc"
    //     balancing nudges b[e] during training instead of adding an aux loss,
    //     so at inference it is just a bias add. No softmax over all 384.
    ss[e] = 1.f / (1.f + expf(-logit)) + bias[e];

    // (c) Top-8: eight rounds of block-wide argmax, each followed by masking
    //     the winner. Branchy SIMT, warp divergence, ~8 * log2(384) shuffles.
    //     (Real kernels use a radix/bitonic select; the DAG node is the same.)
    __syncthreads();
    for (int r = 0; r < 8; ++r) {
        float best = ss[e];  int who = e;
        for (int off = 16; off; off >>= 1) {                 // warp reduce
            float v = __shfl_down_sync(0xffffffffu, best, off);
            int   w = __shfl_down_sync(0xffffffffu, who,  off);
            if (v > best) { best = v; who = w; }
        }
        // ... cross-warp reduce in SMEM, then one lane records the winner ...
        if (e == 0) { top_ids[r] = who_g; top_w[r] = best_g; }
        __syncthreads();
        if (e == top_ids[r]) ss[e] = -INFINITY;              // mask it out
        __syncthreads();
    }

    // (d) Renormalise the 8 selected scores, apply alpha = 2.827, then write
    //     the per-token (expert_id, weight) pairs. Downstream: a SCATTER of
    //     token-rows to expert owners (T7), then k independent grouped GEMMs.
}
