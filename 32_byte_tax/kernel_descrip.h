// ---- v4a, HOST SIDE. Build ONCE per tensor, not once per tile. ----
// cuTensorMapEncodeTiled lives in the driver API (<cuda.h> / cudaTypedefs.h).
CUtensorMap tmap{};
constexpr uint32_t rank = 2;

// IMPORTANT: TMA coordinates and dimensions are innermost-dimension FIRST.
// For a row-major M x K matrix, globalDim[0] = K (columns), [1] = M (rows).
cuuint64_t globalDim[rank] = {K, M};

// Row pitch IN BYTES. This is the field that used to be `lda` in your kernel.
// HARD RULE: every global stride must be a multiple of 16 bytes. If lda*2
// isn't, cuTensorMapEncodeTiled rejects the descriptor -- the padded pitch
// from §2 is no longer a performance tweak, it's a precondition.
cuuint64_t globalStrides[rank - 1] = {lda * sizeof(__half)};

cuuint32_t boxDim[rank]     = {BLOCK_K, BLOCK_M};  // the tile, in ELEMENTS
cuuint32_t elemStrides[rank] = {1, 1};             // sub-element stepping

CUresult r = cuTensorMapEncodeTiled(
    &tmap,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
    rank,
    (void*)A,                 // global base: 16-B aligned; 128-B for swizzle_128B
    globalDim, globalStrides, boxDim, elemStrides,
    CU_TENSOR_MAP_INTERLEAVE_NONE,      // NONE / 16B / 32B for narrow types
    CU_TENSOR_MAP_SWIZZLE_128B,         // hardware XORs SMEM addresses on write
                                        // so the MMA sees a bank-conflict-free
                                        // layout; granularity is 16 B
    CU_TENSOR_MAP_L2_PROMOTION_NONE,    // optional 64/128/256-B L2 window
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE); // OOB elements are ZERO-filled, so
                                        // edge tiles need no predication and
                                        // no divergent warp branches
assert(r == CUDA_SUCCESS);

// Pass by value, not by pointer-to-global: __grid_constant__ puts the
// descriptor in the kernel parameter/constant space, which is how CUTLASS
// avoids extra H2D copies and lets the engine prefetch the tensor map.
tma_kernel<<<grid, block>>>(tmap, ...);
