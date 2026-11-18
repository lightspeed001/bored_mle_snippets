@triton.jit
def ssm_chunk_scan(a_ptr, b_ptr, h_ptr, D, T,
                   CHUNK: tl.constexpr, BLOCK_D: tl.constexpr):
    # Diagonal A (S4D, Mamba). This is why the operator collapses from a matrix
    # product to a scalar one: (a1,b1) x (a2,b2) = (a2*a1, a2*b1 + b2).
    c = tl.program_id(0)                     # which chunk
    d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)

    h = tl.load(h_ptr + c * D + d)          # state carried in from the last chunk
    for i in tl.static_range(CHUNK):         # the only sequential part
        t = c * CHUNK + i
        a = tl.load(a_ptr + t * D + d)      # Ā = exp(ΔA), already discretised
        u = tl.load(b_ptr + t * D + d)      # B̄ x
        h = a * h + u                       # <- the recurrence, one FMA per feature
        tl.store(h_ptr + (c + 1) * D + d, h)   # hand the boundary state on

# Between chunks: h_{c+1} = (prod Ā_i) h_c + sum(...) - a scan over a few hundred
# values, or a rank-n GEMM. Never write the (T, D, N) expanded state to HBM: it is
# the buffer that turns a fast model into a slow one.
