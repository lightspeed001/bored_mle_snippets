// ---- v0: naive. 32 lanes × 4 B = one 128-B window, four 32-B sectors. ----
__global__ void copy_v0(const float* __restrict__ in,
                        float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];          // LDG.E; lane k reads base + 4k
}

// ---- v1: vectorised. One 16-B load per lane. ----
// float4 is 16 B, so nvcc emits LDG.E.128 and the *base pointer* must be
// 16-B aligned. cudaMalloc returns generously aligned storage (256 B in
// practice), so a fresh allocation is fine; a sliced view may not be.
// One warp now covers 32 * 16 = 512 B = 4 cache lines = 16 useful sectors.
// Same sector physics, 4x fewer instructions and 4x fewer requests.
__global__ void copy_v1(const float4* __restrict__ in,
                        float4* __restrict__ out, int n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) out[i] = in[i];
}

// ---- v2: the diagnostic, NOT an optimisation step. ----
// Take v0's contiguous 128-B warp window and start it at base + 4 B.
// The window is now [4, 132), crossing the 128-B line boundary: 5 sectors.
// Useful bytes 128 / fetched 160 = 80%. With float4 the equivalent is
// 16 -> 17 sectors (94%) -- and a misaligned float4* is undefined behaviour,
// so the compiler may split it into narrower loads instead. Either way the
// hardware is charging you for the sector, not for the bytes you wanted.
//
// Stride version (each lane 128 B apart): 32 sectors, 12.5% efficiency.
// This is exactly what Nsight Compute reports as "sectors per request":
//     4  = perfect coalescing (32-bit loads)
//     5  = one-line-boundary misalignment
//     32 = fully scattered
__global__ void copy_strided(const float* __restrict__ in,
                             float* __restrict__ out, int n, int stride) {
    int lane = threadIdx.x;
    int idx  = (blockIdx.x * blockDim.x + lane) * stride;
    if (idx < n) out[idx] = in[idx];    // set stride = 32 and watch the sectors
}
