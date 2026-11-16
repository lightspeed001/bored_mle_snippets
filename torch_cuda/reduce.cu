// src/kernels/reduce.cu
extern "C" __global__ void k_sum_blocks(
    const float* __restrict__ in, float* __restrict__ partials, int64_t n) {
  // Level 1: warp reduction with shuffle-down. No shared memory, no syncs
  // beyond the implicit warp sync: each step, lane i adds lane i+offset's
  // value into its own. log2(32)=5 steps and lane 0 holds the warp's sum.
  float v = 0.f;
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) v += in[i];          // grid-stride: each thread
                                                  // first reduces its own slice
  __shared__ float smem[32];                      // one slot per warp
  const unsigned FULL = 0xffffffffu;
  for (int off = 16; off > 0; off >>= 1)
    v += __shfl_down_sync(FULL, v, off);          // <- the warp tree

  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) smem[warp] = v;                  // warp leaders park in smem
  __syncthreads();

  // Level 2: first warp folds the per-warp partials. blockDim <= 1024 so
  // there are at most 32 warps, i.e. exactly one warp's worth of slots.
  if (warp == 0) {
    v = (lane < (blockDim.x >> 5)) ? smem[lane] : 0.f;
    for (int off = 16; off > 0; off >>= 1)
      v += __shfl_down_sync(FULL, v, off);
    if (lane == 0) partials[blockIdx.x] = v;      // ONE global write per block
  }
}

extern "C" __global__ void k_sum_final(const float* __restrict__ partials,
                                       float* __restrict__ out, int nb) {
  // Tiny second kernel folds <=4096 partials. Yes, a loop in one thread.
  // Yes, it's faster than atomics. No, you may not argue about it until Part 2.
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    float s = 0.f;
    for (int i = 0; i < nb; ++i) s += partials[i];
    out[0] = s;
  }
}
