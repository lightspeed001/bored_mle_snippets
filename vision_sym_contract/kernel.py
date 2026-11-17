# One program computes a BLOCK_M x BLOCK_N tile of the *implicit* im2col matrix.
# The matrix is never built: rows are output pixels, columns are (kh, kw, c) taps,
# so we index the activation with three independent offsets instead of copying it.
@triton.jit
def conv2d_implicit_gemm(x_ptr, w_ptr, y_ptr, H, W_, C_IN, C_OUT,
                          sx_h, sx_w, sx_c,            # NHWC strides
                          KH: tl.constexpr, KW: tl.constexpr,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                          BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # flat output pixel
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # output channel
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K loop over the reduction dim = C_IN * KH * KW. This is where the
    # *banded* structure lives: consecutive taps read overlapping windows, so
    # the same cache lines get reused k^2 times. Nothing tells the compiler
    # that — you buy it back with BLOCK_K and whatever L1 you were born with.
    for k0 in range(0, C_IN * KH * KW, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        c  = offs_k // (KH * KW)
        dr = (offs_k % (KH * KW)) // KW          # kernel row
        dc = offs_k % KW                         # kernel column

        # A: activation tile (BLOCK_M, BLOCK_K). stride 1, pad 0, so the sketch
        # stays readable; the mask is where the contract lapses at the border.
        oh = offs_m // W_
        ow = offs_m % W_
        ih = oh[:, None] + dr[None, :]
        iw = ow[:, None] + dc[None, :]
        m  = (ih < H) & (iw < W_) & (offs_k[None, :] < C_IN * KH * KW)
        a  = tl.load(x_ptr + ih * sx_h + iw * sx_w
                           + c[None, :] * sx_c, mask=m, other=0.0)

        # B: weight tile (BLOCK_K, BLOCK_N), flat as (kh, kw, c, out).
        w = tl.load(w_ptr + offs_k[:, None] * C_OUT + offs_n[None, :],
                    mask=offs_k[:, None] < C_IN * KH * KW, other=0.0)

        acc = tl.dot(a, w, acc)   # tensor cores; fp32 accumulate by default

    out = offs_m[:, None] * C_OUT + offs_n[None, :]
    tl.store(y_ptr + out, acc, mask=offs_n[None, :] < C_OUT)
