#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <math.h>

namespace {
union BF16x4 {
    uint2 vec;                  // Four BF16 values = 8 bytes.
    __nv_bfloat162 pair[2];
};

__device__ __forceinline__ float4 load_bf16x4(const __nv_bfloat16* p) {
    BF16x4 u;
    u.vec = *reinterpret_cast<const uint2*>(p);  // 8-B aligned by the layout.
    return make_float4(
        __bfloat162float(u.pair[0].x), __bfloat162float(u.pair[0].y),
        __bfloat162float(u.pair[1].x), __bfloat162float(u.pair[1].y));
}
} // namespace


template<int HKV, int GROUP, int D, int PAGE, int CHUNK>
__global__ void paged_decode_partial_cuda(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const int* __restrict__ page_ids,
    float* __restrict__ partial_o,       // [HQ, R, D]
    float* __restrict__ partial_m,       // [HQ, R]
    float* __restrict__ partial_l,       // [HQ, R]
    int length, int num_chunks, float softmax_scale)
{
    static_assert(HKV == 8 && GROUP == 8 && D == 128 && PAGE == 16,
                  "This draft fast path is specialized for the declared shape");
    static_assert(CHUNK % PAGE == 0, "Chunk must contain whole logical pages");
    constexpr int HQ = HKV * GROUP;

    // K + V = 8 KiB; scores = 512 B. Alignment permits uint4 shared stores.
    __shared__ alignas(16) __nv_bfloat16 k_s[PAGE * D];
    __shared__ alignas(16) __nv_bfloat16 v_s[PAGE * D];
    __shared__ float score_s[GROUP][PAGE];

    const int tid = threadIdx.x;         // Exactly 256 threads.
    const int lane = tid & 31;
    const int warp = tid >> 5;           // 0..7: one real GQA head per warp.
    const int chunk = blockIdx.x;
    const int hkv = blockIdx.y;
    const int h = hkv * GROUP + warp;
    const int d0 = lane * 4;             // 32 lanes * 4 dimensions = D.

    const float4 q_reg = load_bf16x4(q + h * D + d0);
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    float m = -INFINITY;
    float l = 0.f;

    const int token_begin = chunk * CHUNK;
    const int token_end = min(length, token_begin + CHUNK);
    const int first_page = token_begin / PAGE;
    const int last_page = (token_end + PAGE - 1) / PAGE;

    for (int logical_page = first_page; logical_page < last_page; ++logical_page) {
        // All threads read the same entry; the cache supplies a broadcast.
        const int physical_page = page_ids[logical_page];
        const size_t base = (static_cast<size_t>(physical_page) * HKV + hkv)
                          * (PAGE * D);

        // PAGE*D*2 B = 4 KiB per slab. 256 threads * 16 B = 4 KiB.
        const uint4* src_k = reinterpret_cast<const uint4*>(k_cache + base);
        const uint4* src_v = reinterpret_cast<const uint4*>(v_cache + base);
        reinterpret_cast<uint4*>(k_s)[tid] = src_k[tid];
        reinterpret_cast<uint4*>(v_s)[tid] = src_v[tid];
        __syncthreads();                  // K/V tile is visible to all eight warps.

        // QK: each warp computes one score per token, with D split across lanes.
        for (int t = 0; t < PAGE; ++t) {
            const float4 k_reg = load_bf16x4(k_s + t * D + d0);
            float dot = q_reg.x * k_reg.x;
            dot = fmaf(q_reg.y, k_reg.y, dot);
            dot = fmaf(q_reg.z, k_reg.z, dot);
            dot = fmaf(q_reg.w, k_reg.w, dot);
#pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                dot += __shfl_xor_sync(0xFFFFFFFFu, dot, offset);
            }
            if (lane == 0) score_s[warp][t] = dot * softmax_scale;
        }
        __syncwarp();                     // Publish this warp's score_s row.

        const int valid = min(PAGE, length - logical_page * PAGE);
        const float score = lane < valid ? score_s[warp][lane] : -INFINITY;

        float page_max = score;
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            page_max = fmaxf(page_max,
                __shfl_xor_sync(0xFFFFFFFFu, page_max, offset));
        }

        const float new_m = fmaxf(m, page_max);
        const float alpha = expf(m - new_m);  // exp(-inf) = 0 on the first page.
        float p = lane < valid ? expf(score - new_m) : 0.f;

        float p_sum = p;
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            p_sum += __shfl_xor_sync(0xFFFFFFFFu, p_sum, offset);
        }

        // Rescale once per page, not once per token.
        l = l * alpha + p_sum;
        acc.x *= alpha; acc.y *= alpha; acc.z *= alpha; acc.w *= alpha;
        m = new_m;

        // AV: probability p_t lives in lane t; broadcast it to the whole warp.
        for (int t = 0; t < valid; ++t) {
            const float p_t = __shfl_sync(0xFFFFFFFFu, p, t);
            const float4 v_reg = load_bf16x4(v_s + t * D + d0);
            acc.x = fmaf(p_t, v_reg.x, acc.x);
            acc.y = fmaf(p_t, v_reg.y, acc.y);
            acc.z = fmaf(p_t, v_reg.z, acc.z);
            acc.w = fmaf(p_t, v_reg.w, acc.w);
        }
        __syncthreads();                  // Nobody overwrites K/V while still in use.
    }

    // Each lane owns four contiguous FP32 output dimensions: one aligned float4.
    float4* dst = reinterpret_cast<float4*>(
        partial_o + (static_cast<size_t>(h) * num_chunks + chunk) * D + d0);
    *dst = acc;
    if (lane == 0) {
        partial_m[h * num_chunks + chunk] = m;
        partial_l[h * num_chunks + chunk] = l;
    }
}
