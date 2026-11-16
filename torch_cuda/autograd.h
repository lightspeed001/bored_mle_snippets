// include/pyra/autograd.h
namespace pyra {

enum class Op : uint8_t {
  ADD, SUB, MUL, DIV, NEG, EXP, LOG, TANH, RELU, GELU,
  SUM, MEAN, MATMUL, SOFTMAX, LOG_SOFTMAX, CROSS_ENTROPY, MSE
};

struct GradNode {
  Op op;
  std::vector<Tensor> inputs;    // saved for backward (or their sufficient stats:
  std::vector<Tensor> saved;     // e.g. tanh saves its OUTPUT, softmax saves probs)
  Tensor output;
};

struct Tape { std::vector<GradNode> nodes; };
extern thread_local Tape g_tape;          // one tape per thread; one thread, so: one tape
extern thread_local bool g_grad_mode;     // NoGradGuard flips this

struct NoGradGuard {                      // eval loops without this allocate a tape
  NoGradGuard() { g_grad_mode = false; }  // for every forward and then leak it into
  ~NoGradGuard() { g_grad_mode = true; }  // the void. ask me how I know.
};

void backward(const Tensor& loss);        // reverse-walk + seed
void accumulate(const Tensor& leaf, const Tensor& g);   // leaf.grad += g  (ALWAYS +=)

} // namespace pyra
