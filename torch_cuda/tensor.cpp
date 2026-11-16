// src/tensor.cpp
Tensor transpose(const Tensor& t, int d0, int d1) {
  TensorImpl* out = new TensorImpl();
  out->storage = t->storage;          // SAME storage: refcount goes 1 -> 2.
  out->sizes   = t->sizes;            // No allocation. No copy. No kernel.
  out->strides = t->strides;          // We are about to swap six integers and
  out->offset  = t->offset;           // call it linear algebra.
  std::swap(out->sizes[d0],   out->sizes[d1]);
  std::swap(out->strides[d0], out->strides[d1]);
  return Tensor(out, false);
}

Tensor slice(const Tensor& t, int dim, int64_t lo, int64_t hi) {
  TensorImpl* out = new TensorImpl();
  out->storage = t->storage;
  out->sizes   = t->sizes;   out->strides = t->strides;
  out->sizes[dim] = hi - lo;
  out->offset = t->offset + lo * t->strides[dim];   // walk the pointer, keep the strides
  return Tensor(out, false);
}
