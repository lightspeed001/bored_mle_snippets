// Maps: T4 + T5 -> DAG nodes g6, g5 -> E8 (causal attention).
// Roles follow the FlashAttention-4 / CUTLASS Blackwell pattern: dedicated
// warpgroups for TMA, MMA, softmax, correction and epilogue. Accumulators
// live in TMEM so several MMAs can be in flight while CUDA cores do exp.

__global__ void fa4_decode(const half* Q, const half* k_cache, const half* v_cache,
                           const int* block_table, int ctx, float* O,
                           float* m_state, float* l_state) {
    // ---- role assignment (warp specialisation) ---------------------------
    // WG3 warp1 : TMA loads Q/K/V tiles GMEM -> SMEM
    // WG3 warp0 : issues BOTH tcgen05 MMAs (QK^T and PV)
    // WG3 warp2 : TMA store of final O tile
    // WG0/WG1   : softmax for Q-stage 0 / 1, ping-pong
    // WG2       : correction / rescaling warps

    float m = m_state[head], l = l_state[head];   // online softmax state
    for (int blk = 0; blk < ceil_div(ctx, BLOCK); ++blk) {
        int phys = block_table[blk];               // T5: SIMT pointer chase,
                                                   // L2-hot, ~4 B per block
        // (a) TMA: paged K/V tile -> SMEM. 128 B swizzle, double buffered.
        tma_load(K_smem, k_cache + phys*BLOCK*kv_dim);
        tma_load(V_smem, v_cache + phys*BLOCK*kv_dim);
        mbarrier_wait(tma_done);

        // (b) MMA #1: S = Q K^T / sqrt(d_h), accumulator in TMEM.
        //     MLA absorbed path: K tile is [c_KV 512 | k^R 64] = 576 wide.
        tcgen05_mma(S_tmem, Q_smem, K_smem);
        apply_causal_mask_and_scale(S_tmem, blk, 1.f/sqrtf(d_h));

        // (c) SOFTMAX in SIMT registers. tcgen05.ld brings S from TMEM.
        //     FA4 uses an FMA polynomial for exp instead of MUFU.EX2, and
        //     SKIPS rescaling unless (m_new - m) is large: ~10x fewer ops.
        float s_max = warp_row_max(tcgen05_ld(S_tmem));
        float m_new = fmaxf(m, s_max);
        float rescale = exp_fma_poly(m - m_new);       // NOT expf() on SFU
        float P = exp_fma_poly(S - m_new);
        l = rescale*l + warp_row_sum(P);
        tmem_rescale(O_tmem, rescale);                 // correction warp
        tmem_store_P(P_tmem, P);                       // P back into TMEM

        // (d) MMA #2: O += P V, accumulator stays in TMEM across all blocks.
        tcgen05_mma(O_tmem, P_tmem, V_smem, /*accumulate=*/true);
        m = m_new;
        mbarrier_arrive(next_stage);                   // pipeline handoff
    }
    tcgen05_ld_to_gmem(O, O_tmem, /*divide_by=*/l);   // normalised output
    m_state[head] = m; l_state[head] = l;
}

// PREFILL is the same kernel with M = 12 and an extra KV WRITE. There is no
// separate "prefill algorithm" in the mathematics -- only in launch geometry.
