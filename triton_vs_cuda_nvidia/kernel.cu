#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace {
constexpr int kGroup = 128;
constexpr int kPackedGroup = kGroup / 2;       // 64 bytes
constexpr int kGroupsPerIter = 8;
constexpr int kPackedIter = kGroupsPerIter * kPackedGroup; // 512 B
constexpr int kXIter = kGroupsPerIter * kGroup;            // 1,024 BF16

__device__ __forceinline__ float signed_nibble(uint32_t v) {
    v &= 0x0Fu;
    return v < 8u ? static_cast<float>(v)
                  : static_cast<float>(v) - 16.0f;
}

union X8 {
    uint4 vec;                 // 8 BF16 values = 16 bytes
    __nv_bfloat162 pair[4];
};

// Four packed bytes + eight BF16 values = eight products.
// All indices are compile-time after unrolling; verify that SASS contains no
// local-memory traffic from x.pair[].
__device__ __forceinline__ float dot8(uint32_t packed, uint4 x_bits) {
    X8 x;
    x.vec = x_bits;
    float d = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint32_t byte = (packed >> (8 * j)) & 0xFFu;
        const float x0 = __bfloat162float(x.pair[j].x);
        const float x1 = __bfloat162float(x.pair[j].y);
        d = fmaf(signed_nibble(byte),      x0, d);
        d = fmaf(signed_nibble(byte >> 4), x1, d);
    }
    return d;
}
} // namespace


template<int K>
__global__ void w4a16_gemv_cuda(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ w,
    const __half* __restrict__ scales,
    __nv_bfloat16* __restrict__ y,
    int N)
{
    static_assert(K % kXIter == 0, "Fast path requires K divisible by 1024");

    const int lane = threadIdx.x;              // blockDim.x == 32
    const int warp = threadIdx.y;              // eight warps in our launch
    const int row = blockIdx.x * blockDim.y + warp;
    if (row >= N) return;                      // Handles arbitrary N tails.

    constexpr int kGroups = K / kGroup;
    const uint8_t* w_row = w + static_cast<size_t>(row) * (K / 2);
    const __half* s_row = scales + static_cast<size_t>(row) * kGroups;
    float acc = 0.0f;

    for (int it = 0; it < kGroups / kGroupsPerIter; ++it) {
        // Lane l owns packed bytes [16*l, 16*l+16) within this 512-B slab.
        // The base is 16-B aligned because K and all slab strides are aligned.
        const size_t w_off = static_cast<size_t>(it) * kPackedIter
                           + lane * sizeof(uint4);
        const uint4 packed = *reinterpret_cast<const uint4*>(w_row + w_off);

        // One lane's 16 packed bytes represent 32 adjacent logical weights.
        // Four 16-B loads therefore fetch its 32 BF16 activation values.
        const int x_off = it * kXIter + lane * 32;
        const uint4 x0 = *reinterpret_cast<const uint4*>(x + x_off);
        const uint4 x1 = *reinterpret_cast<const uint4*>(x + x_off + 8);
        const uint4 x2 = *reinterpret_cast<const uint4*>(x + x_off + 16);
        const uint4 x3 = *reinterpret_cast<const uint4*>(x + x_off + 24);

        // Four lanes cover one 64-B group and read the same scale.
        const int group = it * kGroupsPerIter + lane / 4;
        const float scale = __half2float(__ldg(s_row + group));

        float d = dot8(packed.x, x0);
        d += dot8(packed.y, x1);
        d += dot8(packed.z, x2);
        d += dot8(packed.w, x3);
        acc = fmaf(scale, d, acc);
    }

    // All 32 lanes participated in the same row; reduce in registers.
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, offset);
    }
    if (lane == 0) y[row] = __float2bfloat16(acc);
}


void launch_cuda_4096(const __nv_bfloat16* x,
                      const uint8_t* w,
                      const __half* scales,
                      __nv_bfloat16* y,
                      int n,
                      cudaStream_t stream) {
    constexpr int kWarpsPerCta = 8;
    dim3 block(32, kWarpsPerCta);
    dim3 grid((n + kWarpsPerCta - 1) / kWarpsPerCta);
    w4a16_gemv_cuda<4096><<<grid, block, 0, stream>>>(x, w, scales, y, n);
}
