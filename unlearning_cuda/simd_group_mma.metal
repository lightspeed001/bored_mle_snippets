// step4a_m4_simdgroup_mma.metal — Optimal Metal 3 Path for M4 (and Portable Fallback)
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// Inside our 128-thread (4-simdgroup) kernel: each simdgroup computes a 32x32 output block
// (= 4x4 grid of 8x8 tiles) for BOTH Gate and Up projections -> 32 float8x8 accumulators.
inline void m4_simdgroup_inner_loop(
    threadgroup const half* As, // [64][TG_STRIDE]
    threadgroup const half* Bg, // [64][TG_STRIDE]
    uint sgid,                  // 0..3 -> maps to 2x2 quadrant of the 64x64 threadgroup tile
    thread simdgroup_float8x8 (&acc_g)[4][4] // MUST be float8x8, NEVER half8x8 at 4x4 grid!
) {
    const uint sg_row = (sgid / 2) * 32;
    const uint sg_col = (sgid % 2) * 32;

    for (uint kk = 0; kk < BK; kk += 8) {
        simdgroup_half8x8 a_frag[4];
        simdgroup_half8x8 b_frag[4];

        // Cooperative 32-lane load: 1 instruction loads an 8x8 half tile across the simdgroup
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            simdgroup_load(a_frag[i], As + (sg_row + i * 8) * TG_STRIDE + kk, TG_STRIDE);
            simdgroup_load(b_frag[i], Bg + (sg_col + i * 8) * TG_STRIDE + kk, TG_STRIDE,
                           ulong2(0, 0), /*transpose=*/true);
        }

        // 16x outer-product 8x8 MMAs (FP16 inputs -> FP32 accumulator on the 128-ALU pipe)
        #pragma clang loop unroll(full)
        for (uint m = 0; m < 4; ++m) {
            #pragma clang loop unroll(full)
            for (uint n = 0; n < 4; ++n) {
                simdgroup_multiply_accumulate(acc_g[m][n], a_frag[m], b_frag[n], acc_g[m][n]);
            }
        }
    }
}
