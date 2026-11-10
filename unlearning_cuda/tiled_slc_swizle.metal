// step3_tiled_slc_swizzle.metal — Async Threadgroup Tiling + Bank-Conflict Padding + SLC Swizzle
#include <metal_stdlib>
#include <metal_compute>
using namespace metal;

constant constexpr uint BM = 64;
constant constexpr uint BN = 64;
constant constexpr uint BK = 32;
// CRITICAL PADDING: 32 halfs = 64 bytes (16x 4B banks). Without padding, strided column/fragment
// loads hit 4- or 8-way threadgroup bank conflicts. Adding +4 halfs (8 bytes = 2 banks) skews rows!
constant constexpr uint TG_STRIDE = BK + 4;
constant constexpr uint SWIZZLE_W = 8; // Group 8 N-tiles together to maximize SLC hit rate!

kernel void gated_gemm_step3(
    device const half* A      [[buffer(0)]], // [M, K] row-major
    device const half* B_gate [[buffer(1)]], // [N, K] transposed (K contiguous!)
    device const half* B_up   [[buffer(2)]], // [N, K] transposed
    device half*       Out    [[buffer(3)]], // [M, N]
    constant uint3&    dims   [[buffer(4)]], // {M, N, K}
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid  [[thread_position_in_threadgroup]], // {0..15, 0..7} = 128 threads (4 simdgroups)
    uint  sgid [[simdgroup_index_in_threadgroup]]
) {
    const uint M = dims.x, N = dims.y, K = dims.z;

    // --- SLC RASTER SWIZZLE ---
    // Remap linear 2D grid into 8-column vertical panels so concurrent GPU cores share
    // the exact same A-panel and B-panel inside the 16–48 MB System Level Cache!
    const uint num_pid_m = (M + BM - 1) / BM;
    const uint num_pid_n = (N + BN - 1) / BN;
    const uint group_sz  = num_pid_m * SWIZZLE_W;
    const uint linear_id = tgid.y * num_pid_n + tgid.x;
    const uint group_id  = linear_id / group_sz;
    const uint first_n   = group_id * SWIZZLE_W;
    const uint cur_swiz  = min(num_pid_n - first_n, SWIZZLE_W);
    const uint pid_m     = (linear_id % group_sz) / cur_swiz;
    const uint pid_n     = first_n + ((linear_id % group_sz) % cur_swiz);

    // Total TG allocation: 64*36*2B (A) + 2 * 64*36*2B (Gate+Up) = 13,824 Bytes (< 32 KB limit)
    threadgroup half As[BM * TG_STRIDE];
    threadgroup half Bg[BN * TG_STRIDE];
    threadgroup half Bu[BN * TG_STRIDE];

    // Per-thread 4x8 scalar register tile (before we switch to matrix instructions in Step 4)
    float acc_gate[4][8] = {{0.0f}};
    float acc_up[4][8]   = {{0.0f}};

    for (uint k0 = 0; k0 < K; k0 += BK) {
        // Hardware DMA copy from device/SLC -> padded threadgroup SRAM (only sgid 0 issues group copy)
        simdgroup_event ev[3];
        if (sgid == 0) {
            ev[0] = simdgroup_async_copy(
                As, TG_STRIDE, ulong2(BK, BM),
                A + pid_m * BM * K + k0, K, ulong2(BK, min(BM, M - pid_m * BM)));
            ev[1] = simdgroup_async_copy(
                Bg, TG_STRIDE, ulong2(BK, BN),
                B_gate + pid_n * BN * K + k0, K, ulong2(BK, min(BN, N - pid_n * BN)));
            ev[2] = simdgroup_async_copy(
                Bu, TG_STRIDE, ulong2(BK, BN),
                B_up + pid_n * BN * K + k0, K, ulong2(BK, min(BN, N - pid_n * BN)));
        }
        // Synchronize async DMA across all 4 simdgroups in the threadgroup
        if (sgid == 0) {
            simdgroup_event::wait(3, ev);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute 4x8 per-thread sub-tile from conflict-free threadgroup memory
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < BK; ++kk) {
            float a_frag[4], bg_frag[8], bu_frag[8];
            #pragma clang loop unroll(full)
            for (uint m = 0; m < 4; ++m)
                a_frag[m] = float(As[(tid.y * 8 + m) * TG_STRIDE + kk]); // simplified mapping
            #pragma clang loop unroll(full)
            for (uint n = 0; n < 8; ++n) {
                bg_frag[n] = float(Bg[(tid.x * 4 + n) * TG_STRIDE + kk]);
                bu_frag[n] = float(Bu[(tid.x * 4 + n) * TG_STRIDE + kk]);
            }
            #pragma clang loop unroll(full)
            for (uint m = 0; m < 4; ++m)
                for (uint n = 0; n < 8; ++n) {
                    acc_gate[m][n] = fma(a_frag[m], bg_frag[n], acc_gate[m][n]);
                    acc_up[m][n]   = fma(a_frag[m], bu_frag[n], acc_up[m][n]);
                }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // ... Fused SiLU(gate) * up epilogue store to Out[M, N] ...
}
