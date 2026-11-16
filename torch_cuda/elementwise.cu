

// src/kernels/elementwise.cu
// NOTE: no pyra headers in here. Device code stays standalone; the host-side
// launcher lives in kernels.cpp and talks to this through get_function().

#define PYRA_MAX_DIM 4

// Unravel a flat index into up to 4 coordinates, then walk each operand's
// strides. Broadcasting is just "this operand's stride here is 0", which the
// host arranged for us with expand_to().
__device__ __forceinline__ int64_t strided_offset(
    int64_t idx, const int64_t* strides, int ndim, const int64_t* sizes) {
  int64_t off = 0;
  for (int d = ndim - 1; d >= 0; --d) {        // innermost dim first
    int64_t c = idx % sizes[d];
    idx      /= sizes[d];
    off      += c * strides[d];               // stride 0 => broadcast: free
  }
  return off;
}

// Generic binary elementwise kernel. OP is a macro symbol like PYRA_OP_ADD.
#define PYRA_BINARY_KERNEL(NAME, EXPR)                                          \
extern "C" __global__ void k_##NAME(                                            \
    const float* __restrict__ a, const int64_t* __restrict__ sa,                \
    const float* __restrict__ b, const int64_t* __restrict__ sb,                \
    float* __restrict__ out,                                                    \
    const int64_t* __restrict__ osizes, int ndim, int64_t n) {                  \
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;                   \
  int64_t stride = (int64_t)gridDim.x * blockDim.x;                             \
  for (; i < n; i += stride) {                    /* grid-stride: one launch  */\
    int64_t oa = strided_offset(i, sa, ndim, osizes);  /* config fits any n,  */\
    int64_t ob = strided_offset(i, sb, ndim, osizes);  /* and Part 2 can tune  */\
    float x = a[oa], y = b[ob];                    /* occupancy independently */\
    out[i] = (EXPR);                                                           \
  }                                                                             \
}

#define PYRA_OP_ADD (x + y)
#define PYRA_OP_SUB (x - y)
#define PYRA_OP_MUL (x * y)
#define PYRA_OP_DIV (x / y)
PYRA_BINARY_KERNEL(add, PYRA_OP_ADD)
PYRA_BINARY_KERNEL(sub, PYRA_OP_SUB)
PYRA_BINARY_KERNEL(mul, PYRA_OP_MUL)
PYRA_BINARY_KERNEL(div, PYRA_OP_DIV)

// Unary twin for neg/exp/log/tanh/relu/gelu; gelu uses the tanh
// approximation because the exact one needs erff() and a therapy session.
#define PYRA_UNARY_KERNEL(NAME, EXPR)                                           \
extern "C" __global__ void k_##NAME(                                            \
    const float* __restrict__ a, float* __restrict__ out, int64_t n) {          \
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;                   \
  int64_t stride = (int64_t)gridDim.x * blockDim.x;                             \
  for (; i < n; i += stride) { float x = a[i]; out[i] = (EXPR); }               \
}
PYRA_UNARY_KERNEL(neg, -x)
PYRA_UNARY_KERNEL(exp, expf(x))
PYRA_UNARY_KERNEL(log, logf(x))
PYRA_UNARY_KERNEL(tanh, tanhf(x))
PYRA_UNARY_KERNEL(relu, x > 0.f ? x : 0.f)
PYRA_UNARY_KERNEL(gelu, 0.5f * x * (1.f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))))
