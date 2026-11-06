// Maps: T1 (GEMM) -> DAG node g3 -> E9 (c_Q = h W_DQ) on B200, sm_100.
// READ AS A SHAPE. Production code (CUTLASS SM100 collective builder / CuTe,
// or DeepGEMM's JIT FP8 kernels) builds SMEM + TMEM descriptors with ~40
// lines of bit-field encoding. Those bits are not the insight; this is.
//
// Blackwell vs Hopper in three lines:
//   wgmma   : warpgroup (128 threads) issues, accumulator in REGISTERS
//   tcgen05 : ONE elected lane issues, accumulator in TMEM
//   result  : MMA is fully async, so CUDA cores can work in parallel

// ---- resources (all three limit occupancy) -------------------------------
__shared__ alignas(128) __nv_fp8_e4m3 A[2][128][128];   // 128 B swizzle is
__shared__ alignas(128) __nv_fp8_e4m3 B[2][128][128];   // REQUIRED by tcgen05
__shared__ uint64_t mbar[2];        // one mbarrier per double-buffer stage
__shared__ uint32_t tmem_base;      // TMEM allocation, 128 lanes x N columns

// One warp allocates TMEM; the allocator is warp-collective, the MMA is not.
if (warp_id == 0 && elect_one()) tmem_base = tcgen05_alloc(/*columns*/ 128);
__syncthreads();

for (int k = 0; k < 56; ++k) {              // K = 7168, BLOCK_K = 128
    const int s = k & 1;                    // double-buffered stage

    // (a) TMA PRODUCER warp: a DMA engine, not CUDA cores. Brings an FP8 tile
    //     from HBM (via L2) straight into SMEM. Registers are untouched.
    if (warp_id == 0 && elect_one()) {
        cp_async_bulk_tensor(A[s], W_DQ, tile_desc_a(k));   // 16 KB
        cp_async_bulk_tensor(B[s], h,    tile_desc_b(k));   // 16 KB
        mbarrier_arrive(mbar[s]);                            // signal consumer
    }
    mbarrier_wait(mbar[s], phase);                            // GRAPH EDGE #1

    // (b) MMA ISSUER warp: ONE lane. FP8 x FP8 with 128x128 block scale
    //     factors baked into the instruction descriptor; FP32 accumulate.
    //     A/B are read from SMEM descriptors, never from registers.
    if (warp_id == 1 && elect_one()) {
        tcgen05_mma(tmem_base, smem_desc(A[s]), smem_desc(B[s]),
                    /*accumulate=*/ (k != 0), /*kind=*/ FP8_BLOCK_SCALED,
                    scale_a[k], scale_b[k]);
        tcgen05_commit(mbar[s]);                              // async completion
    }

    // (c) SIMT warps 2..7 are FREE here: RMSNorm of the previous layer,
    //     softmax of the previous attention tile, or next TMA setup. This
    //     overlap is the entire reason Blackwell changed the MMA issue model.
    tcgen05_wait(mbar[s]);                                    // GRAPH EDGE #2
    phase ^= 1;
}

// (d) EPILOGUE: TMEM -> registers, then BF16 out. This is where T2/T3 fuse.
float4 acc = tcgen05_ld(tmem_base);                          // tcgen05.ld
acc = apply_fp8_block_scale(acc) + bias;                     // dequant + bias
acc = fused_rmsnorm_and_rope(acc, gamma, cosT, sinT, pos);  // see next kernel
store_bf16(O_tile, acc);                                     // 32 KB -> HBM
tcgen05_dealloc(tmem_base);

// The 672 CTAs are independent; GigaThread spreads them over 148 SMs / 2 dies.
// No CTA knows another CTA exists. Only the mbarrier inside a CTA is a real
// dependency -- which is exactly why the DAG node g3 can be ONE node.
