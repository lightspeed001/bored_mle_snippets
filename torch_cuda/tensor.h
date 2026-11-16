// include/pyra/tensor.h
#pragma once
#include "pyra/storage.h"
#include <cstdint>
#include <vector>

namespace pyra {

using Shape = std::vector<int64_t>;
struct GradNode;                     // defined in autograd.h; TensorImpl only
                                     // ever holds a pointer to one

struct TensorImpl {
  Handle<Storage> storage;          // who owns the bytes
  Shape sizes;                      // [3, 4]
  Shape strides;                    // [4, 1]  (in ELEMENTS, like PyTorch, not bytes)
  int64_t offset{0};                // first element, in elements
  bool requires_grad{false};
  // autograd hooks; both null for a fresh leaf:
  struct GradNode* node{nullptr};   // set when this tensor was produced by a tracked op
  Handle<TensorImpl> grad;          // the .grad buffer, allocated lazily on first accumulate
  mutable std::atomic<int> refcount{1};
  void retain() const { refcount.fetch_add(1, std::memory_order_relaxed); }
  void release() const;             // frees impl; Storage refcount handles the bytes
  int64_t numel() const { int64_t n = 1; for (auto s : sizes) n *= s; return n; }
};

using Tensor = Handle<TensorImpl>;

// the whole geometry of the framework, in one formula:
//   address(i0, i1, ..., ik) =
//       storage.ptr + (offset + i0*stride[0] + i1*stride[1] + ...) * sizeof(float)
//
// Every view, slice, transpose and broadcast in this article is just a
// different (sizes, strides, offset) triple over unchanged bytes.

Tensor from_host(const float* data, Shape sizes);       // upload
std::vector<float> to_host(const Tensor& t);            // download (syncs; it's fine)
Tensor zeros(Shape sizes);
Tensor full_like(const Tensor& t, float v);
Tensor view(const Tensor& t, Shape sizes);              // contiguous reshape only
Tensor transpose(const Tensor& t, int d0, int d1);      // <- the trick
Tensor slice(const Tensor& t, int dim, int64_t lo, int64_t hi);
Tensor expand_to(const Tensor& t, Shape sizes);         // stride-0 broadcast
void dump(const Tensor& t, const char* name);           // debug print, host-side

} // namespace pyra
