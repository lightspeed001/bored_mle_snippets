// BAD: What 90% of quick CUDA-to-Metal ports accidentally write
std::vector<__fp16> host_x(B * D);
// Wait: we're on Unified Memory, so this just wraps host_x, right?
// NO! newBufferWithBytes: allocates a BRAND NEW Metal page pool and memcpy's host_x into it!
id<MTLBuffer> buf_x = [device newBufferWithBytes:host_x.data()
                                          length:bytes
                                         options:MTLResourceStorageModeShared];
