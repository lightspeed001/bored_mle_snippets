// src/autograd.cpp
namespace pyra {
thread_local Tape g_tape;
thread_local bool g_grad_mode = true;

// Called by every differentiable op right after launching its forward kernel.
Tensor track(Op op, std::vector<Tensor> inputs, std::vector<Tensor> saved,
             Tensor output) {
  if (!g_grad_mode) return output;
  bool any = false;
  for (auto& t : inputs) if (t->requires_grad) any = true;
  if (!any) return output;                       // inference-shaped subgraph: skip
  g_tape.nodes.push_back(GradNode{op, std::move(inputs), std::move(saved), output});
  output->requires_grad = true;
  output->node = &g_tape.nodes.back();
  return output;
}

void accumulate(const Tensor& leaf, const Tensor& g) {
  // +=, never =. If a tensor is used twice (think x*x), its gradient arrives
  // twice, and the multivariate chain rule says those contributions SUM.
  // Every scalar-autograd tutorial on the internet has this bug once. Ours is
  // in the comments, where it can hurt nobody.
  Tensor gg = g;
  if (!same_shape(gg->sizes, leaf->sizes)) gg = expand_to(gg, leaf->sizes);
  if (!leaf->grad) leaf->grad = gg;
  else launch_add_inplace(leaf->grad, gg);       // grad = grad + gg, fused
}

void backward(const Tensor& loss) {
  accumulate(loss, full_like(loss, 1.0f));       // seed: dL/dL = 1
  for (auto it = g_tape.nodes.rbegin(); it != g_tape.nodes.rend(); ++it) {
    // The gradient of this node's output, already accumulated by the nodes
    // that consumed it (we walk in reverse, so they ran first).
    Tensor go = it->output->grad;
    if (!go) continue;
    switch (it->op) {                            // the "engine": a switch statement
      case Op::ADD:
        accumulate(it->inputs[0], go);           // d(a+b)/da = 1
        accumulate(it->inputs[1], go);           // d(a+b)/db = 1
        break;
      case Op::MUL:
        accumulate(it->inputs[0], mul(go, it->inputs[1]));  // d(ab)/da = b
        accumulate(it->inputs[1], mul(go, it->inputs[0]));  // d(ab)/db = a
        break;
      case Op::TANH:
        // saved[0] is the OUTPUT y: d tanh/dx = 1 - y^2. Saving outputs
        // instead of inputs when the derivative is cheaper in y is a
        // micro-optimization frameworks actually make.
        accumulate(it->inputs[0], mul(go, sub(full_like(it->saved[0], 1.f),
                                              mul(it->saved[0], it->saved[0]))));
        break;
      case Op::MATMUL: {
        // d/dA (A@B) = G@B^T ;  d/dB (A@B) = A^T@G
        accumulate(it->inputs[0], matmul(go, transpose(it->inputs[1], 0, 1)));
        accumulate(it->inputs[1], matmul(transpose(it->inputs[0], 0, 1), go));
        break;
      }
      case Op::SUM:
        accumulate(it->inputs[0], expand_to(go, it->inputs[0]->sizes));
        break;                                   // broadcast the scalar grad back
      case Op::MEAN:
        accumulate(it->inputs[0],
                   mul(expand_to(go, it->inputs[0]->sizes),
                       full_like(go, 1.f / (float)it->inputs[0]->numel())));
        break;
      case Op::CROSS_ENTROPY:
        // saved[0] = softmax probs P. d/dlogits mean(-log P[target]) =
        // (P - onehot(target)) / batch. One fused kernel; see loss.cu.
        accumulate(it->inputs[0], ce_backward(it->saved[0], it->inputs[1],
                                              (int)it->inputs[0]->sizes[0]));
        break;
      /* ... SUB, DIV, NEG, EXP, LOG, RELU, GELU, SOFTMAX, LOG_SOFTMAX, MSE:
         same shape of code, one line of calculus each. Full table in
         Appendix A; full source at the v0.6-autograd tag. ... */
      default: break;
    }
  }
  g_tape.nodes.clear();      // the tape is per-backward. keeping it is how you
}                            // leak your entire forward pass "for debugging".
} // namespace pyra
