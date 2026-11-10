// uma_allocator.mm — True Zero-Copy 16 KB DART-Mapped Shared Buffers + Private Scratch
#import <Metal/Metal.h>
#include <mach/vm_page_size.h> // vm_page_size == 16384 on all Apple Silicon Macs
#include <stdlib.h>

struct ZeroCopyBuffer {
    void*         cpu_ptr;   // Direct pointer for CPU NEON / ARM SME reads & writes
    id<MTLBuffer> mtl_buf;   // Mapped into GPUVA via DART — zero byte copies!
    size_t        alloc_len; // Rounded to 16 KB multiple
};

ZeroCopyBuffer alloc_zero_copy_shared(id<MTLDevice> dev, size_t raw_bytes) {
    // CRITICAL: Both base address AND byte length MUST be multiples of 16,384 bytes!
    // Using 4096 here compiles cleanly and silently returns nil at runtime.
    const size_t page_sz   = vm_page_size; // 16,384 bytes
    const size_t aligned_n = (raw_bytes + page_sz - 1) & ~(page_sz - 1);

    void* ptr = nullptr;
    if (posix_memalign(&ptr, page_sz, aligned_n) != 0 || !ptr) {
        abort();
    }

    // Pre-fault pages once at init so our first benchmark run doesn't measure Darwin VM zero-fill faults
    memset(ptr, 0, aligned_n);

    id<MTLBuffer> buf = [dev newBufferWithBytesNoCopy:ptr
                                               length:aligned_n
                                              options:MTLResourceStorageModeShared
                                          deallocator:^(void* p, NSUInteger) { free(p); }];
    return {ptr, buf, aligned_n};
}

// For GPU-only intermediate activations (e.g., Stage A -> Stage C normalized activations):
// Allocate once in MTLStorageModePrivate + HazardTrackingModeUntracked (we order via encoders/barriers).
id<MTLBuffer> alloc_gpu_private_scratch(id<MTLDevice> dev, size_t bytes) {
    MTLResourceOptions opts = MTLResourceStorageModePrivate | MTLResourceHazardTrackingModeUntracked;
    return [dev newBufferWithLength:bytes options:opts];
}
