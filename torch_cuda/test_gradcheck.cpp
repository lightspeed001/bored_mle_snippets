// tests/test_gradcheck.cpp — central finite differences against our own kernels
bool gradcheck(const std::function<Tensor(Tensor)>& f, const Tensor& x,
               float eps = 1e-2f, float rtol = 1e-2f) {
  Tensor y = f(x); backward(y);
  auto g = to_host(x->grad);
  auto xh = to_host(x);
  for (size_t i = 0; i < xh.size(); i += std::max<size_t>(1, xh.size() / 32)) {
    auto fp = f(set_elem(x, i, xh[i] + eps));   // perturb ONE element, host-side,
    auto fm = f(set_elem(x, i, xh[i] - eps));   // re-upload, re-forward, item()
    float numeric = (item(fp) - item(fm)) / (2 * eps);
    if (std::abs(numeric - g[i]) > rtol * std::abs(numeric) + 1e-3f) return false;
  }
  return true;
}
