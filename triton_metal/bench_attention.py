# Pseudocode harness: adapt the callables to the chosen library versions.
# The same inputs and semantics must reach all three implementations.
import statistics
import time

def benchmark(fn, inputs, warmup=20, repeats=100):
    for _ in range(warmup):
        fn(*inputs)                       # Includes lazy compilation/cache warmup.
    synchronize_device()                  # Ensure warmup work is complete.
    samples_ms = []
    for _ in range(repeats):
        synchronize_device()
        t0 = time.perf_counter_ns()
        out = fn(*inputs)
        synchronize_device()              # Time completed GPU work, not enqueue speed.
        samples_ms.append((time.perf_counter_ns() - t0) / 1e6)
    return statistics.median(samples_ms), percentile(samples_ms, 90), out

cases = [(512, 64, "fp16", False), (2048, 64, "fp16", True),
         (2048, 128, "fp16", False)]
for seq, head_dim, dtype, causal in cases:
    inputs = make_fixed_inputs(seq, head_dim, dtype, causal, seed=1234)
    # Compile each candidate before measuring; report compile time separately.
    results = {name: benchmark(fn, inputs) for name, fn in candidates.items()}
    validate_allclose(results, reference_attention(inputs), atol=..., rtol=...)
    print(seq, head_dim, dtype, causal, summarize(results))

# Record chip, OS, Xcode/Metal SDK, compiler commits, power mode, and exact kernel configs.
# Also log scratch bytes, registers/spills if available, and whether a fallback was selected.
