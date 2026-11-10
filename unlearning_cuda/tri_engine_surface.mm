// step7_tri_engine_surface.mm — One Physical Allocation Aliased by SME2, Metal 4 GPU, and ANE
#import <Metal/Metal.h>
#import <IOSurface/IOSurface.h>
#import <CoreVideo/CoreVideo.h>
#include <arm_neon.h>
#include <mach/vm_page_size.h>

struct TriEngineSurface {
    IOSurfaceRef     surface;    // Kernel-managed physical page backing (resident in Shared SLC)
    __fp16*          cpu_ptr;    // View 1: Direct pointer for CPU NEON / 512-bit ARM SME2
    id<MTLBuffer>    gpu_buf;    // View 2: Zero-copy DART GPUVA buffer for Metal 3 / Metal 4
    CVPixelBufferRef ane_pixbuf; // View 3: Zero-copy OneComponent16Half wrapper for Core ML ANE
    size_t           s_stride;   // Spatial stride in fp16 elements (multiple of 32 = 64 bytes!)
    size_t           alloc_bytes;
};

TriEngineSurface create_tri_engine_surface(id<MTLDevice> dev, size_t channels, size_t seq_len) {
    // Rule 1: Pad spatial width S to a multiple of 32 fp16s (64 bytes) for ANE / CVPixelBuffer
    const size_t s_stride       = (seq_len + 31) & ~size_t(31);
    const size_t bytes_per_row  = s_stride * sizeof(__fp16); // Guaranteed multiple of 64 bytes
    const size_t raw_bytes      = channels * bytes_per_row;

    // Rule 2 & 3: Round total size to 16 KB Darwin/DART page multiple, with a 64 KB (>49 KB) floor
    const size_t min_ane_bytes  = 65536;
    const size_t clamped_bytes  = raw_bytes < min_ane_bytes ? min_ane_bytes : raw_bytes;
    const size_t aligned_alloc  = (clamped_bytes + vm_page_size - 1) & ~(vm_page_size - 1);

    NSDictionary* props = @{
        (id)kIOSurfaceWidth:           @(s_stride),
        (id)kIOSurfaceHeight:          @(channels),
        (id)kIOSurfaceBytesPerElement: @(2),
        (id)kIOSurfaceBytesPerRow:     @(bytes_per_row),
        (id)kIOSurfaceAllocSize:       @(aligned_alloc),
        (id)kIOSurfacePixelFormat:     @(kCVPixelFormatType_OneComponent16Half)
    };
    IOSurfaceRef surf = IOSurfaceCreate((__bridge CFDictionaryRef)props);

    // View 1 (CPU / SME2): Base address is kernel VM-mapped and 16 KB page-aligned
    __fp16* base_ptr = static_cast<__fp16*>(IOSurfaceGetBaseAddress(surf));

    // View 2 (Metal GPU): Map exact IOSurface pages into GPUVA — 0 bytes copied!
    id<MTLBuffer> mtl_buf = [dev newBufferWithBytesNoCopy:base_ptr
                                                   length:aligned_alloc
                                                  options:MTLResourceStorageModeShared
                                              deallocator:nil]; // Owned by IOSurfaceRef

    // View 3 (16-Core ANE): Alias the exact same IOSurface as a CVPixelBuffer
    CVPixelBufferRef pix = nullptr;
    CVPixelBufferCreateWithIOSurface(kCFAllocatorDefault, surf, nullptr, &pix);

    return {surf, base_ptr, mtl_buf, pix, s_stride, aligned_alloc};
}

// ZERO-TRANSPOSE PRODUCER TRICK:
// Stage A (RMSNorm on P-Core NEON/SME) normalizes 4 token rows [s..s+3, 0..C-1] at a time and
// transposes 4x4 fp16 micro-tiles IN REGISTERS (`vtrn1q_f16` / `vtrn2q_f16`) before storing to
// `out_planar[c * s_stride + s]`. Cost: 0 extra memory passes, lands hot in Shared SLC!
void rmsnorm_fused_to_planar_slc(const __fp16* __restrict x_rowmajor, // [S, C]
                                 const __fp16* __restrict gamma,      // [C]
                                 __fp16* __restrict out_planar,       // [1, C, 1, s_stride] in IOSurface
                                 size_t S, size_t C, size_t s_stride) {
    // 1. Compute 4 row inverse-RMS scalars in FP32 registers (1 pass over L1/L2-hot rows)
    // 2. Walk c += 8, multiply by gamma[c..c+7], 4x8 in-register NEON transpose -> contiguous
    //    writes into out_planar[c * s_stride + s0 .. s0+3] (omitted loop boilerplate for brevity).
}
