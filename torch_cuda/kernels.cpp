// src/kernels.cpp
void launch_add(const Tensor& a, const Tensor& b, const Tensor& out) {
  // a and b arrive PRE-EXPANDED: the op wrapper broadcast_shapes()'d them and
  // called expand_to() so both have out's rank, with stride 0 on any dim they
  // broadcast over. The kernel never thinks about shapes; strides think.
  CUfunction f = get_function((const void*)k_add);   // resolved once, cached

  // Stride vectors, left-padded to the output rank, uploaded once per launch.
  // (A real framework keeps these pinned; we allocate, we are young.)
  int64_t sa[PYRA_MAX_DIM], sb[PYRA_MAX_DIM], osz[PYRA_MAX_DIM];
  int ndim = (int)out->sizes.size();
  pad_strides(a, sa, ndim); pad_strides(b, sb, ndim); pad_sizes(out, osz, ndim);
  CUdeviceptr dsa = upload_i64(sa, ndim), dsb = upload_i64(sb, ndim),
              dsz = upload_i64(osz, ndim);

  const float* pa = (const float*)a->storage->ptr;
  const float* pb = (const float*)b->storage->ptr;
  float*       po = (float*)out->storage->ptr;
  int64_t      n  = out->numel();

  // *** THE LINE ***
  // args[] holds POINTERS TO THE ARGUMENT VALUES, not the values, in exactly
  // the order of the kernel signature: (a, sa, b, sb, out, osizes, ndim, n).
  // The driver copies sizeof(arg) bytes from each address into kernel
  // parameter space. Write {pa, pb, po, n} and you get no compile error, no
  // crash — just gorgeous garbage, because the pointer values themselves
  // become your arguments. Get the ORDER wrong and it is the same story,
  // one indirection subtler: a float* reinterpreted as an int64_t* stride.
  void* args[] = { &pa, &dsa, &pb, &dsb, &po, &dsz, &ndim, &n };

  const int threads = 256;
  // Cap the grid: with a grid-stride loop we don't need one block per element,
  // and 4096 blocks keeps tail effects and launch overhead both boring.
  int blocks = (int)std::min<int64_t>((n + threads - 1) / threads, 4096);
  CU_LAUNCH_CHECK(cuLaunchKernel(f, blocks, 1, 1, threads, 1, 1,
                                 0, /*stream*/ nullptr, args, nullptr));
}
