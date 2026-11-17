// src/kernels/elementwise.cu — v1: the vectorized twin of Part 1's k_add
extern "C" __global__ void k_add_f4(const float* __restrict__ a,
                                    const float* __restrict__ b,
                                    float* __restrict__ out, int64_t n) {
  // PRECONDITION, non-negotiable: a, b, out are 16-byte aligned. Our caching
  // allocator rounds every allocation to 256 B, so this holds by construction.
  // If you ever bolt this framework onto foreign memory (numpy buffers, mmap,
  // a friend's allocator), check alignment here or eat an alignment fault.
  const float4* va = (const float4*)a;
  const float4* vb = (const float4*)b;
  float4*       vo = (float4*)out;

  const int64_t n4 = n >> 2;                       // whole vectors
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (; i < n4; i += stride) {
    float4 x = va[i], y = vb[i], r;                // one LDG.E.128 each
    r.x = x.x + y.x; r.y = x.y + y.y;
    r.z = x.z + y.z; r.w = x.w + y.w;              // 4 independent FMAs: ILP
    vo[i] = r;                                     // one STG.E.128
  }
  // Tail: n % 4 scalars. At most three. One thread cleans up; spawning a
  // second kernel for three floats would be a launch overhead joke.
  if (threadIdx.x == 0 && blockIdx.x == 0)
    for (int64_t j = n4 << 2; j < n; ++j) out[j] = a[j] + b[j];
}
