// include/pyra/runtime.h
#pragma once
#include <cuda.h>      // driver API: cu*
#include <cuda_runtime.h>  // ONLY for cudaFree(0) + cudaGetFuncBySymbol; see below
#include <cstdio>
#include <cstdlib>

// Every driver call returns CUresult. Ignoring it is how you get
// "my kernel did nothing" with a straight face. Check. Everything.
#define CU_CHECK(x) do {                                                     \
    CUresult _r = (x);                                                       \
    if (_r != CUDA_SUCCESS) {                                                \
      const char* s = nullptr; cuGetErrorString(_r, &s);                     \
      std::fprintf(stderr, "[pyra] CUDA driver error: %s\n  at %s:%d\n",     \
                   s ? s : "?", __FILE__, __LINE__);                         \
      std::abort();                                                          \
    }                                                                        \
  } while (0)

// After a launch, the error may only materialize when the GPU actually
// gets around to running your kernel. Sync to catch it. We keep this ON
// in debug builds and OFF in benchmarks, and we are honest about which.
#ifdef PYRA_SYNC_CHECKS
#  define CU_LAUNCH_CHECK(x) do { CU_CHECK(x); CU_CHECK(cuCtxSynchronize()); } while (0)
#else
#  define CU_LAUNCH_CHECK(x) CU_CHECK(x)
#endif
