template<int D>
__global__ void merge_decode_cuda_32(
    const float* __restrict__ partial_o,
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    __nv_bfloat16* __restrict__ out,
    int num_chunks)                       // 1 <= R <= 32
{
    static_assert(D == 128, "Four dimensions per lane");
    const int h = blockIdx.x;             // One warp per query head.
    const int lane = threadIdx.x;         // blockDim.x == 32.

    const float mc = lane < num_chunks
        ? partial_m[h * num_chunks + lane] : -INFINITY;
    const float lc = lane < num_chunks
        ? partial_l[h * num_chunks + lane] : 0.f;

    float global_m = mc;
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        global_m = fmaxf(global_m,
            __shfl_xor_sync(0xFFFFFFFFu, global_m, offset));
    }

    const float weight = lane < num_chunks ? expf(mc - global_m) : 0.f;
    float denom = weight * lc;
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        denom += __shfl_xor_sync(0xFFFFFFFFu, denom, offset);
    }

    const int d0 = lane * 4;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int r = 0; r < num_chunks; ++r) {
        const float w_r = __shfl_sync(0xFFFFFFFFu, weight, r);
        const float4 o = *reinterpret_cast<const float4*>(
            partial_o + (static_cast<size_t>(h) * num_chunks + r) * D + d0);
        acc.x = fmaf(w_r, o.x, acc.x);
        acc.y = fmaf(w_r, o.y, acc.y);
        acc.z = fmaf(w_r, o.z, acc.z);
        acc.w = fmaf(w_r, o.w, acc.w);
    }

    out[h * D + d0 + 0] = __float2bfloat16(acc.x / denom);
    out[h * D + d0 + 1] = __float2bfloat16(acc.y / denom);
    out[h * D + d0 + 2] = __float2bfloat16(acc.z / denom);
    out[h * D + d0 + 3] = __float2bfloat16(acc.w / denom);
}
