// src/kernels/optim.cu
extern "C" __global__ void k_adam_step(
    float* __restrict__ p, float* __restrict__ g,
    float* __restrict__ m, float* __restrict__ v,
    float lr, float b1, float b2, float t, float eps, int64_t n) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = (int64_t)gridDim.x * blockDim.x;
  // Bias-correction denominators are scalars; compute once per thread, they
  // are two expf's away from free and hoist out of the loop anyway.
  float bc1 = 1.f - powf(b1, t), bc2 = 1.f - powf(b2, t);
  for (; i < n; i += stride) {
    float gi = g[i];
    float mi = b1 * m[i] + (1.f - b1) * gi;          // first moment  (mean of grads)
    float vi = b2 * v[i] + (1.f - b2) * gi * gi;     // second moment (mean of squares)
    float mh = mi / bc1, vh = vi / bc2;              // bias correction
    p[i] -= lr * mh / (sqrtf(vh) + eps);             // THE step
    g[i] = 0.f;                                      // zero_grad, for free, fused.
    m[i] = mi; v[i] = vi;                            // PyTorch needs a separate
  }                                                  // zero_grad() call; we are
}                                                    // not PyTorch. yet.
