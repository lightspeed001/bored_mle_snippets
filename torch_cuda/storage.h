// include/pyra/storage.h
#pragma once
#include <cuda.h>
#include <atomic>
#include <utility>

namespace pyra {

void alloc_free(CUdeviceptr p, std::size_t nbytes);   // -> caching allocator, runtime.cpp

struct Storage {
  CUdeviceptr ptr{0};
  std::size_t nbytes{0};
  mutable std::atomic<int> refcount{1};   // intrusive refcount; no shared_ptr overhead,
                                          // no control block, no excuses

  void retain() const { refcount.fetch_add(1, std::memory_order_relaxed); }
  void release() const {
    // Last reference out the door frees the device memory. This is the ONLY
    // place in the framework where cuMemFree happens for tensor data.
    if (refcount.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      alloc_free(ptr, nbytes);            // -> the caching allocator, below
      delete this;
    }
  }
};

template <class T>
struct Handle {                          // the 12-line intrusive_ptr that carries
  T* p{nullptr};                         // Storage and TensorImpl around
  Handle() = default;
  explicit Handle(T* q, bool addref = true) : p(q) { if (p && addref) p->retain(); }
  Handle(const Handle& o) : p(o.p) { if (p) p->retain(); }
  Handle(Handle&& o) noexcept : p(o.p) { o.p = nullptr; }
  Handle& operator=(const Handle& o) { if (o.p) o.p->retain(); if (p) p->release(); p = o.p; return *this; }
  Handle& operator=(Handle&& o) noexcept { if (p) p->release(); p = o.p; o.p = nullptr; return *this; }
  ~Handle() { if (p) p->release(); }
  T* operator->() const { return p; }
  T& operator*() const { return *p; }
  explicit operator bool() const { return p != nullptr; }
};

} // namespace pyra
