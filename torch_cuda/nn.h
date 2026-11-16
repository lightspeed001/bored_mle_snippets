// include/pyra/nn.h
namespace pyra::nn {

struct Module {
  virtual ~Module() = default;
  virtual Tensor forward(const Tensor& x) = 0;
  Tensor operator()(const Tensor& x) { return forward(x); }

  std::vector<Tensor> params;                 // this module's own leaves
  std::vector<Module*> children;              // borrowed pointers; ownership is
                                              // yours (unique_ptr in your model)
  void register_param(Tensor t) { t->requires_grad = true; params.push_back(t); }
  void register_child(Module* m) { children.push_back(m); }

  std::vector<Tensor> parameters() {          // recursive walk, PyTorch-style
    std::vector<Tensor> out = params;
    for (auto* c : children)
      for (auto& p : c->parameters()) out.push_back(p);
    return out;
  }
};

struct Linear : Module {
  Tensor W, b;
  Linear(int64_t in, int64_t out, std::mt19937& rng) {
    // kaiming-uniform, a=sqrt(5): bound = 1/sqrt(fan_in). Host RNG, uploaded.
    // cuRAND in-kernel Philox is a Part 2 footnote; determinism across runs
    // is worth more here than GPU-side entropy.
    W = uniform_tensor({out, in}, -1.f / std::sqrt((float)in), rng);
    b = uniform_tensor({out},     -1.f / std::sqrt((float)in), rng);
    register_param(W); register_param(b);
  }
  Tensor forward(const Tensor& x) override {  // x:[B,in] -> [B,out]
    return add(matmul(x, transpose(W, 0, 1)), b);   // b broadcasts via stride 0
  }
};

struct ReLU : Module { Tensor forward(const Tensor& x) override { return relu(x); } };
struct GELU : Module { Tensor forward(const Tensor& x) override { return gelu(x); } };

} // namespace pyra::nn
