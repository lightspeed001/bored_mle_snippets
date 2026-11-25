@triton.jit
def gemm_tile(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # One logical program owns one output tile. “Logical” is doing important work here.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)  # Accumulator precision is a real design knob.

    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        # These loads state demand. Hardware must still choose DMA bursts, SRAM banks, and reuse.
        a = tl.load(A + ... , mask=rows[:, None] < M, other=0.0)
        b = tl.load(B + ... , mask=cols[None, :] < N, other=0.0)
        acc += tl.dot(a, b)  # A reduction request, not a pre-installed systolic array.

    tl.store(C + ..., acc, mask=(rows[:, None] < M) & (cols[None, :] < N))
