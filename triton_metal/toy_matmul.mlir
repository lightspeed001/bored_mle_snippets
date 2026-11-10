// Educational MLIR sketch: a tile is logical here; no NVIDIA lane layout is baked in.
func.func @tile_matmul(%A: memref<64x32xf16>,
                       %B: memref<32x64xf16>,
                       %C: memref<64x64xf32>) {
  %zero = arith.constant 0.0 : f32
  linalg.fill ins(%zero : f32) outs(%C : memref<64x64xf32>)
  // Real production code would use explicit loop/tile bounds and masks.
  linalg.matmul ins(%A, %B : memref<64x32xf16>, memref<32x64xf16>)
               outs(%C : memref<64x64xf32>)
  return
}

// A target pass can tile and distribute this op only after it knows the target:
// Metal-safe scratch estimate: (BM*BK + BK*BN) * sizeof(f16) + epilogue bytes
// Must be <= device.maxThreadgroupMemoryLength (commonly 32 KiB on current Apple GPUs).
// If not, shrink the tile, stream from device memory, or refuse this specialization.
