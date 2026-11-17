// examples/mnist.cpp — the only new code is a 30-line IDX parser, because
// MNIST's format predates your framework and will outlive it.
static Tensor load_idx_images(const char* path, int64_t* n_out) {
  std::vector<uint8_t> raw = read_file_bytes(path);
  // magic: 0x00000803 | count | rows | cols, all big-endian. yes, really.
  auto be32 = [&](size_t o) { return (int32_t)((raw[o] << 24) | (raw[o+1] << 16) |
                                               (raw[o+2] << 8) | raw[o+3]); };
  int64_t n = be32(4), r = be32(8), c = be32(12);
  std::vector<float> host(n * r * c);
  for (int64_t i = 0; i < n * r * c; ++i) host[i] = raw[16 + i] / 255.f;  // [0,1]
  *n_out = n;
  return from_host(host.data(), {n, r * c});      // one upload, never again
}
// labels: same parser, magic 0x00000801, kept as int32 on device.

int main() {
  init(0);
  Tensor x = load_idx_images("data/train-images-idx3-ubyte", &ntrain);
  Tensor y = load_idx_labels("data/train-labels-idx1-ubyte", &ntrain);
  Net net(rng);                                   // 784-128-128-10, kaiming-uniform
  Adam optim(net.parameters(), 1.2e-3f);
  for (int epoch = 1; epoch <= 6; ++epoch) {
    for (auto [xb, yb] : shuffle_batches(x, y, 128, rng)) {
      Tensor loss = cross_entropy(net(xb), yb);   // fused: softmax+nll+reduce
      backward(loss);                             // 10 kernels after §5
      optim.step();                               // 6 fused, grads zeroed inline
    }
    NoGradGuard ng;
    std::printf("epoch %d: test acc %.2f%%\n", epoch, 100.0 * accuracy(net, xte, yte));
  }
}
