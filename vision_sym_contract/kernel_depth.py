# One filter per channel means no reduction across channels, so there's nothing
# for tl.dot() to do. This is a stencil: 9 loads, 9 FMAs, 1 store.
@triton.jit
def depthwise3x3_nhwc(x_ptr, w_ptr, y_ptr, H, W_, C, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)   # flat (h, w, c)
    acc  = tl.zeros((BLOCK,), dtype=tl.float32)
    ch   = offs % C

    for tap in tl.static_range(9):                 # unrolled at compile time
        dr, dc = tap // 3 - 1, tap % 3 - 1
        # In NHWC a 1-pixel x-shift is +C elements, a 1-pixel y-shift is +W*C.
        # Shift the *flat* index; never recompute (h, w) per tap. This is the
        # whole reason depthwise is fine in NHWC and a crime in NCHW, where
        # neighbouring lanes sit W*C elements apart.
        base = offs + dr * W_ * C + dc * C
        m    = (base >= 0) & (base < H * W_ * C) & (ch >= dc)
        a    = tl.load(x_ptr + base, mask=m, other=0.0)
        wk   = tl.load(w_ptr + ch * 9 + tap)   # 9 weights per channel
        acc += a * wk

    tl.store(y_ptr + offs, acc)   # bandwidth-bound: ~1 FLOP per byte moved
