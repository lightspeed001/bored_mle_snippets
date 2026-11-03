// ---- v4b, DEVICE SIDE. The whole "movement" part of the kernel. ----
#include <cuda/barrier>
#include <cuda/ptx>
using barrier = cuda::barrier<cuda::thread_scope_block>;

// elect.sync is better than `threadIdx.x == 0`: the compiler *knows* exactly
// one lane is predicated on, so it won't emit a peeling loop over all lanes
// (which would serialise the TMA issue and quietly wreck the pipeline).
__device__ inline bool elect_one() {
    const unsigned warp_id = threadIdx.x / 32;
    const unsigned uniform_warp_id = __shfl_sync(0xFFFFFFFF, warp_id, 0);
    return uniform_warp_id == 0 && cuda::ptx::elect_sync(0xFFFFFFFF);
}

__global__ void tma_kernel(const __grid_constant__ CUtensorMap tmap,
                           int tile_k, int tile_m) {
    // 128-B aligned: multi-dimensional TMA requires a 128-B-aligned SMEM
    // destination because the engine writes swizzled 16-B chunks. NVIDIA's
    // 128-B-swizzle example even uses alignas(1024), since that swizzle
    // pattern repeats every 1024 B.
    __shared__ alignas(128) __half tile[BLOCK_M][BLOCK_K];

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar;              // mbarrier is 8-B aligned by the type

    if (threadIdx.x == 0) {
        init(&bar, blockDim.x);
        // CRITICAL: the mbarrier lives in SMEM but the TMA engine operates in
        // the ASYNC proxy. Without this fence the engine may observe a stale
        // (uninitialised) barrier. Same fence is needed before a TMA store,
        // after threads have written to SMEM.
        cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
    }
    __syncthreads();

    barrier::arrival_token tok;
    if (elect_one()) {
        int32_t coords[2] = {tile_k, tile_m};   // innermost coordinate FIRST
        cuda::ptx::cp_async_bulk_tensor(
            cuda::ptx::space_shared, cuda::ptx::space_global,
            &tile, &tmap, coords,
            cuda::device::barrier_native_handle(bar));

        // Tell the barrier how many BYTES the hardware will deliver. The
        // barrier flips phase only when BOTH the thread arrivals AND the
        // transaction count reach zero. Register this in the same
        // single-threaded region as the issue; CUTLASS typically hoists it
        // ahead of the copy so the tx count is armed before completion.
        tok = cuda::device::barrier_arrive_tx(bar, 1, sizeof(tile));
    } else {
        tok = bar.arrive();
    }
    bar.wait(std::move(tok));             // no polling loop, no spin flag

    // tile is now in SMEM, already swizzled into the MMA's expected layout.
    // Not one lane computed an address. Not one register held a data byte.
}
