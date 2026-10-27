#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace {
constexpr int kGroup = 128;
constexpr int kGroupsPerIter = 8;
constexpr int kPackedIter = 512;       // 8 groups * 64 packed bytes
constexpr int kXIter = 1024;           // 8 groups * 128 BF16 values

__device__ __forceinline__ float signed_nibble(uint32_t v) {
    v &= 0x0Fu;
    return v < 8u ? static_cast<float>(v)
                  : static_cast<float>(v) - 16.0f;
}

union X8 {
    uint4 vec;
    __nv_bfloat162 pair[4];
};

// Four packed bytes and eight BF16 activations produce eight products.
__device__ __forceinline__ float dot8(uint32_t packed, uint4 x_bits) {
    X8 x;
    x.vec = x_bits;
    float d = 0.f;
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


template<int N, int K>
__global__ void grouped_w4a16_persistent_cuda(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ w,
    const __half* __restrict__ scales,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ active_experts,  // Sorted by descending count.
    const int* __restrict__ token_ids,
    __nv_bfloat16* __restrict__ y,
    int active_count,
    int* __restrict__ work_counter)          // Initialized to zero per launch.
{
    static_assert(N == 4096 && K == 4096,
                  "Primary fast path; other shapes use their own instantiation");
    constexpr int kWarpsPerCta = 8;
    constexpr int kRowBlocks = N / kWarpsPerCta;
    constexpr int kMaxAssignments = 4;

    __shared__ int work_s;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int total_work = active_count * kRowBlocks;

    while (true) {
        // One CTA leader reserves an item; the barrier publishes it to all warps.
        if (tid == 0) work_s = atomicAdd(work_counter, 1);
        __syncthreads();
        const int work = work_s;
        if (work >= total_work) break;        // Uniform across the whole CTA.

        const int active_i = work / kRowBlocks;
        const int row_block = work % kRowBlocks;
        const int expert = active_experts[active_i];
        const int start = expert_offsets[expert];
        const int count = expert_offsets[expert + 1] - start;

        // Preload token IDs into registers. The unrolled loops below must not
        // turn tokens[] or acc[] into local-memory arrays; inspect SASS.
        int tokens[kMaxAssignments];
#pragma unroll
        for (int t = 0; t < kMaxAssignments; ++t) {
            tokens[t] = t < count ? token_ids[start + t] : 0;
        }

        const int row = row_block * kWarpsPerCta + warp;
        const uint8_t* w_row =
            w + (static_cast<size_t>(expert) * N + row) * (K / 2);
        const __half* s_row =
            scales + (static_cast<size_t>(expert) * N + row) * (K / kGroup);

        float acc[kMaxAssignments] = {0.f, 0.f, 0.f, 0.f};

        for (int it = 0; it < K / kXIter; ++it) {
            const size_t w_off = static_cast<size_t>(it) * kPackedIter
                               + lane * sizeof(uint4);
            const uint4 packed = *reinterpret_cast<const uint4*>(w_row + w_off);
            const int group = it * kGroupsPerIter + lane / 4;
            const float scale = __half2float(__ldg(s_row + group));

            // Weights and scale are register-resident before the assignment loop.
#pragma unroll
            for (int t = 0; t < kMaxAssignments; ++t) {
                if (t < count) {
                    const int x_off = it * kXIter + lane * 32;
                    const __nv_bfloat16* x_t = x + tokens[t] * K + x_off;
                    const uint4 x0 = *reinterpret_cast<const uint4*>(x_t);
                    const uint4 x1 = *reinterpret_cast<const uint4*>(x_t + 8);
                    const uint4 x2 = *reinterpret_cast<const uint4*>(x_t + 16);
                    const uint4 x3 = *reinterpret_cast<const uint4*>(x_t + 24);

                    float d = dot8(packed.x, x0);
                    d += dot8(packed.y, x1);
                    d += dot8(packed.z, x2);
                    d += dot8(packed.w, x3);
                    acc[t] = fmaf(scale, d, acc[t]);
                }
            }
        }

        // Each warp reduces one output row independently for each live assignment.
#pragma unroll
        for (int t = 0; t < kMaxAssignments; ++t) {
            if (t < count) {
                float value = acc[t];
#pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) {
                    value += __shfl_down_sync(0xFFFFFFFFu, value, offset);
                }
                if (lane == 0) {
                    y[static_cast<size_t>(start + t) * N + row] =
                        __float2bfloat16(value);
                }
            }
        }

        // Keep the shared work slot safe from the next leader write. Work items
        // are large enough that this barrier should be cheap, but measure it.
        __syncthreads();
    }
}
