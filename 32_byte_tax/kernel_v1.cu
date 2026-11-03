// ---- v3: cp.async / LDGSTS, multi-stage. ----
// Legacy intrinsics shown for clarity; new code should prefer
// cuda::memcpy_async + cuda::barrier / cuda::pipeline from <cuda/barrier>.
//
// What changed vs v1: data goes GMEM -> SMEM WITHOUT visiting a register.
// What did NOT change: 32 lanes still compute 32 addresses and issue 32
// copy instructions per stage. The issue stream is still your problem.
#include <cuda_pipeline.h>

__shared__ alignas(128) float4 smem[STAGES][TILE_F4];  // 128-B aligned base:
                                                       // LDGSTS's happy path
for (int k = 0; k < K_TILES; ++k) {
    int stage = k % STAGES;
    const float4* src = in + k * TILE_F4 + threadIdx.x;
    __pipeline_memcpy_async(&smem[stage][threadIdx.x], src, sizeof(float4));
    __pipeline_commit();                  // close this stage's async group

    if (k >= STAGES) {
        // Wait for all groups PRIOR to the oldest still-needed one, which is
        // what keeps STAGES-1 tiles in flight. This is the "bytes in flight"
        // dial from §1, implemented as pipeline depth.
        __pipeline_wait_prior(STAGES - 1);
        __syncthreads();                  // SMEM writes visible to consumers
        // ... consume smem[(k - STAGES + 1) % STAGES] ...
    }
}
