// sme_router_gemm.cpp — Zero-Dispatch CPU Matrix Microkernel on Apple M4/M5
// Compile: clang++ -O3 -std=c++20 -march=armv9.2-a+sme2 sme_router_gemm.cpp
#include <arm_sme.h>
#include <pthread/qos.h>
#include <stdint.h>

// Step 1: Pin benchmark/worker thread to the Performance-core cluster's full-speed SME unit!
// Without this, Darwin QoS defaults may schedule onto E-cores (~357 GF/s vs ~2,009 GF/s).
inline void pin_thread_to_p_cluster() {
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
}

// Step 2: __arm_locally_streaming emits `smstart sm` on entry and `smstop sm` on return (~15 ns).
// __arm_new("za") tells Clang this function owns a fresh 4,096-byte ZA matrix state (`smstart za`).
// Computes a 32x32 block of C += A_packed * B_packed:
//   A_packed: [K][32] contiguous column-panel (2x 16-float vectors per k-step)
//   B_packed: [K][32] contiguous row-panel    (2x 16-float vectors per k-step)
__arm_locally_streaming __arm_new("za")
void sme_gemm_32x32_fp32(const float* __restrict A_packed,
                         const float* __restrict B_packed,
                         float* __restrict C,
                         int64_t ldc,
                         int64_t K) {
    // SVL on M4/M5 is 512 bits = 16x FP32 lanes; svptrue_b32() predicates all 16 lanes active.
    const svbool_t pg = svptrue_b32();

    // Zero all four 16x16 FP32 tiles (ZA0.S, ZA1.S, ZA2.S, ZA3.S = 4,096 bytes total)
    svzero_za();

    for (int64_t k = 0; k < K; ++k) {
        // TWO-STEP FAST LOAD (~925 GiB/s from L1D/L2, NOT slow direct `ldr za`):
        // Load 32 floats of A[:, k] into two 512-bit Z registers (zA0 = rows 0..15, zA1 = rows 16..31)
        svfloat32_t zA0 = svld1_f32(pg, A_packed + k * 32 + 0);
        svfloat32_t zA1 = svld1_f32(pg, A_packed + k * 32 + 16);

        // Load 32 floats of B[k, :] into two 512-bit Z registers (zB0 = cols 0..15, zB1 = cols 16..31)
        svfloat32_t zB0 = svld1_f32(pg, B_packed + k * 32 + 0);
        svfloat32_t zB1 = svld1_f32(pg, B_packed + k * 32 + 16);

        // INTERLEAVE ALL FOUR ZA TILES to hide M4/M5's 4-cycle FMOPA pipeline latency!
        // Each svmopa_za32_f32_m executes 16 * 16 * 2 = 512 FLOPs -> 2,048 FLOPs per loop iter!
        svmopa_za32_f32_m(/*tile=*/0, pg, pg, zA0, zB0); // Top-left     C[ 0..15,  0..15]
        svmopa_za32_f32_m(/*tile=*/1, pg, pg, zA0, zB1); // Top-right    C[ 0..15, 16..31]
        svmopa_za32_f32_m(/*tile=*/2, pg, pg, zA1, zB0); // Bottom-left  C[16..31,  0..15]
        svmopa_za32_f32_m(/*tile=*/3, pg, pg, zA1, zB1); // Bottom-right C[16..31, 16..31]
    }

    // Drain ZA0..ZA3 horizontal slices back to coherent Unified Memory (lands warm in L2 / SLC!)
    for (uint32_t slice = 0; slice < 16; ++slice) {
        svst1_hor_za32(0, slice, pg, C + (slice +  0) * ldc +  0);
        svst1_hor_za32(1, slice, pg, C + (slice +  0) * ldc + 16);
        svst1_hor_za32(2, slice, pg, C + (slice + 16) * ldc +  0);
        svst1_hor_za32(3, slice, pg, C + (slice + 16) * ldc + 16);
    }
}
