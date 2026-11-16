// src/kernels/softmax.cu
// One warp (32 lanes) per row; 4 rows per block via threadIdx.y.
// probs[r][c] = exp(x[r][c] - max_r) / sum_r exp(x - max_r)
extern "C" __global__ void k_softmax_row(
    const float* __restrict__ in, float* __restrict__ out, int C, int nrows) {
  extern __shared__ float row[];            // dynamic smem: 4 rows * C floats
  int row_id = blockIdx.x * blockDim.y + threadIdx.y;
  int lane   = threadIdx.x;
  const unsigned FULL = 0xffffffffu;
  if (row_id >= nrows) return;

  float* cache = row + threadIdx.y * C;     // this warp's scratch row
  // Pass 0: cooperatively pull the row into smem (strided across 32 lanes).
  for (int c = lane; c < C; c += 32) cache[c] = in[(int64_t)row_id * C + c];
  __syncwarp();

  // Pass 1: row max, warp tree. Lanes past C contribute -inf so they vanish.
  float mx = -INFINITY;
  for (int c = lane; c < C; c += 32) mx = fmaxf(mx, cache[c]);
  for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(FULL, mx, off));
  // (xor, not down: with xor EVERY lane ends up with the result, which saves
  //  a broadcast. down is for when only lane 0 cares, like in k_sum.)

  // Pass 2: sum of exp(x - mx), same tree shape.
  float s = 0.f;
  for (int c = lane; c < C; c += 32) s += __expf(cache[c] - mx);
  for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(FULL, s, off);

  // Pass 3: normalize from smem. __fdividef: fast, and softmax's denominator
  // is never subnormal-ish enough to matter here. Part 2 will re-litigate this.
  float inv = __fdividef(1.f, s);
  for (int c = lane; c < C; c += 32)
    out[(int64_t)row_id * C + c] = __expf(cache[c] - mx) * inv;
}
