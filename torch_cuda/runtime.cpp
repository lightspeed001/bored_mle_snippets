// src/runtime.cpp (allocator)
namespace pyra {

namespace {
constexpr std::size_t kAlign = 256;                 // float4 loads want 16 B; 256 B keeps
inline std::size_t round256(std::size_t n) {        // us clear of partition-camping jokes
  return (n + kAlign - 1) & ~std::size_t(kAlign - 1);
}
struct FreeLists { std::map<std::size_t, std::vector<CUdeviceptr>> by_size; };
FreeLists g_free;
std::size_t g_live = 0;                             // for the "how much VRAM am I?" debug print
} // namespace

CUdeviceptr alloc(std::size_t nbytes) {
  nbytes = round256(nbytes ? nbytes : kAlign);
  auto& bucket = g_free.by_size[nbytes];
  if (!bucket.empty()) {                            // hit: O(log n) map lookup, no ioctl
    CUdeviceptr p = bucket.back(); bucket.pop_back(); return p;
  }
  CUdeviceptr p = 0;
  CU_CHECK(cuMemAlloc(&p, nbytes));                 // miss: pay the municipality
  g_live += nbytes;
  return p;
}

void alloc_free(CUdeviceptr p, std::size_t nbytes) {
  g_free.by_size[round256(nbytes ? nbytes : kAlign)].push_back(p);
}

} // namespace pyra
