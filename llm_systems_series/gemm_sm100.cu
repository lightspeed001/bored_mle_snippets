// Maps: E3/E9 (projection GEMM) -> DAG node g3 -> T1 on B200 (sm_100).
// READ THIS AS A SHAPE, NOT COMPILABLE CODE. Real code builds SMEM/TMEM
// descriptors with CuTe (CUTLASS SM100 collective builder); those encodings
// are omitted here because they are 40 lines of bit-fiddling, not insight.
//
// Blackwell change vs Hopper: wgmma was issued by a 128-thread WARPGROUP and
// accumulated in REGISTERS. tcgen05.mma is issued by ONE elected thread and
// accumulates in TMEM (256 KB/SM, 128 lanes x 512 columns). The warp is now
// an orchestrator; the Tensor Core runs asynchronously alongside SIMT work.

// Warp specialisation: three roles, one CTA, 256 threads (8 warps).
//   warp 0 : TMA producer   - cp.async.bulk.tensor HBM -> SMEM (double buffered)
//   warp 1 : MMA issuer     - one lane issues tcgen05.mma into TMEM
//   warps 2..7 : epilogue   - dequantise, cast, store, fused bias/RoPE if needed

__shared__ alignas(128) __nv_fp8_e4m3 A_smem[2][128][128];  // 128 B swizzle REQUIRED
__shared__ alignas(128) __nv_fp8_e4m3 B_smem[2][128][128];  // for tcgen05 descriptors
__shared__ uint64_t mbar[2];                                 // mbarrier, 2 stages
__shared__ uint32_t tmem_addr;                               // TMEM allocation handle

// mainloop over K tiles (K = 7168, BLOCK_K = 128 -> 56 iterations)
for (int k = 0; k < K / 128; ++k) {
    int stage = k & 1;

    // (a) one elected lane issues the async bulk copy. TMA is a DMA engine:
    //     no CUDA cores participate, registers are untouched.
    if (warp_id == 0 && elect_one()) {
        cp_async_bulk_tensor(A_smem[stage], W_DQ, tile_a(k));   // 16 KB FP8 A tile
        cp_async_bulk_tensor(B_smem[stage], X,     tile_b(k));   // 16 KB FP8 B tile
        mbarrier_arrive(mbar[stage]);                            // producer signal
    }
    mbarrier_wait(mbar[stage], phase);                           // consumer waits

    // (b) one elected lane issues the MMA. FP8 inputs, FP32 accumulator in TMEM.
    //     A/B are read straight from SMEM; results NEVER land in registers here.
    //     FP8 128x128 block scales are applied as part of the MMA descriptor.
    if (warp_id == 1 && elect_one()) {
        tcgen05_mma_f8(tmem_addr, desc(A_smem[stage]), desc(B_smem[stage]),
                       accumulate = (k != 0));
        tcgen05_commit(mbar[stage]);                             // async completion
    }
    // (c) meanwhile the SIMT warps can do RMSNorm / softmax / next TMA setup.
    //     That overlap is the entire point of Blackwell's async MMA.
}
tcgen05_wait(mbar[stage]);

// epilogue: TMEM -> registers (tcgen05.ld), apply FP8 block scales + bias,
// optionally fuse RoPE (see rope.cu) and a BF16 cast, then store to HBM.
// One HBM write of the [128x128] BF16 output tile. That is node g3's output edge.
