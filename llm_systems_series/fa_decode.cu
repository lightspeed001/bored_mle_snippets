// Maps: E8 (causal attention) -> DAG node g6 -> T4 + T5 on B200.
// Decode: Q is [1 x d_h] per head; K/V come from PAGED KV BLOCKS (vLLM style).
// The paged block table is an int array in HBM; each block holds ~16 tokens.
// This is why "context length" is a memory-management problem, not a math one.

float  m  = -INFINITY;      // running max, per query row
float  l  = 0.f;            // running softmax denominator
float* acc = tmem_acc;      // running output accumulator, FP32, IN TMEM

for (int blk = 0; blk < num_kv_blocks; ++blk) {
    int phys = block_table[blk];                    // SIMT pointer chase, L2-hot

    // (a) TMA: bring this block's K and V tiles into SMEM (128 B swizzled).
    //     For MLA the "K tile" is [c_KV 512 | k^R 64]; see the absorbed form.
    tma_load_K(K_smem, k_cache, phys);
    tma_load_V(V_smem, v_cache, phys);

    // (b) MMA #1: S_tile = Q @ K^T / sqrt(d_h)  -> FP32 in TMEM.
    //     Causal masking is applied in the epilogue of this MMA, not by
    //     materialising an N x N mask. Partial last block only.
    tcgen05_mma(acc_s, Q_smem, K_smem);
    apply_causal_scale(acc_s, 1.f / sqrtf(d_h), blk);

    // (c) ONLINE SOFTMAX in SIMT registers. This is the E8 recurrence.
    //     m_new/l_new/o_new are per-row scalars; exp is the SFU bottleneck
    //     FA4 replaces with an FMA polynomial on CUDA cores.
    float s_max = row_max(acc_s);                  // warp shuffle reduction
    float m_new = fmaxf(m, s_max);
    float scale = expf(m - m_new);                 // rescale old accumulator
    float p     = expf(acc_s - m_new);             // new probabilities
    l  = scale * l + row_sum(p);
    acc = scale * acc;                            // rescale TMEM accumulator

    // (d) MMA #2: acc += P @ V. P is BF16 (or FP8) fed back through SMEM;
    //     the accumulator stays in TMEM across the whole KV loop.
    tcgen05_mma(acc, P_smem, V_smem, accumulate = true);
    m = m_new;
}
store_o(acc / l);                                  // BF16 out; node g6's edge

// PREFILL runs the SAME loop with M = 12 instead of M = 1 and additionally
// WRITES K/V rows. No separate algorithm. No special "prefill kernel" in the
// mathematics -- only in the launch geometry. Remember this at the roofline.
