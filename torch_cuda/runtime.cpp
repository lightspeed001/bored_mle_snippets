// src/runtime.cpp
namespace pyra {

void init(int device) {
  // CONFESSION: we are a driver-API-only framework, but the driver needs a
  // CUDA context, and the cheapest legal way to create the *primary* context
  // is one runtime call. cudaFree(0) is the documented idiom for "initialize
  // the runtime and its primary context, harmlessly". After this, the driver
  // and runtime APIs share that context in this process, forever, peacefully.
  cudaFree(0);

  CU_CHECK(cuInit(0));
  CUdevice dev; CU_CHECK(cuDeviceGet(&dev, device));
  CUcontext ctx; CU_CHECK(cuCtxGetCurrent(&ctx));   // the primary context, now current
  if (!ctx) CU_CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));

  char name[256]; CU_CHECK(cuDeviceGetName(name, sizeof name, dev));
  int smem_per_block = 0, sms = 0, clock = 0;
  CU_CHECK(cuDeviceGetAttribute(&smem_per_block, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, dev));
  CU_CHECK(cuDeviceGetAttribute(&sms,  CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
  CU_CHECK(cuDeviceGetAttribute(&clock, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev));
  std::printf("[pyra] %s | %d SMs | %.0f MHz | %d KiB smem/block opt-in\n",
              name, sms, clock / 1e3, smem_per_block / 1024);
}

} // namespace pyra
