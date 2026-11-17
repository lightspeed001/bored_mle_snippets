// src/ptx/load.cpp — your framework loading PTX it did not compile
static CUmodule load_ptx(const char* path) {
  std::string src = read_file(path);            // PTX is TEXT. that's the point.
  CUmodule m;
  // The driver runs its own ptxas here, JITing to the CURRENT gpu's SASS.
  // This call is where "forward-compatible virtual ISA" becomes real.
  CU_CHECK(cuModuleLoadDataEx(&m, src.c_str(), 0, nullptr, nullptr));
  return m;
}
static CUfunction get_fn(CUmodule m, const char* name) {
  CUfunction k; CU_CHECK(cuModuleGetFunction(&k, m, name)); return k;
}
void vadd_ptx(CUdeviceptr a, CUdeviceptr b, CUdeviceptr c, int64_t n) {
  static CUmodule   m = load_ptx("src/ptx/vadd.ptx");   // once per process
  static CUfunction f = get_fn(m, "vadd");
  void* args[] = { &a, &b, &c, &n };            // yes, the rite of passage again
  CU_LAUNCH_CHECK(cuLaunchKernel(f, (int)((n + 255) / 256), 1, 1, 256, 1, 1,
                                 0, nullptr, args, nullptr));
}
