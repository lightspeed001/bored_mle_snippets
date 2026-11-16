// src/kernels/matmul.cu
// out[M,N] = A[M,K] @ B[K,N], all row-major.
extern "C" __global__ void k_matmul_naive(
    const float* __restrict__ A, const float* __restrict__ B,
    float* __restrict__ C, int M, int N, int K) {
  int m = blockIdx.y * blockDim.y + threadIdx.y;
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (m >= M || n >= N) return;
  float acc = 0.f;
  for (int k = 0; k < K; ++k)
    acc += A[m * K + k] * B[k * N + n];   // B's column walk: stride-N loads.
  C[m * N + n] = acc;                     // Coalescing? Never met her.
}
