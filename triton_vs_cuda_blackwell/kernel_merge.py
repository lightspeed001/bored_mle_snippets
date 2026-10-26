@triton.jit
def merge_decode_triton(
    partial_o_ptr, partial_m_ptr, partial_l_ptr, out_ptr,
    num_chunks,
    D: tl.constexpr,        # 128
    BLOCK_R: tl.constexpr,  # 32 for the primary shape
):
    h = tl.program_id(0)
    r = tl.arange(0, BLOCK_R)
    valid_r = r < num_chunks

    m = tl.load(partial_m_ptr + h * num_chunks + r,
                mask=valid_r, other=float("-inf"))
    l = tl.load(partial_l_ptr + h * num_chunks + r,
                mask=valid_r, other=0.0)

    global_m = tl.max(m, axis=0)
    weight = tl.where(valid_r, tl.exp(m - global_m), 0.0)
    denom = tl.sum(weight * l, axis=0)

    d = tl.arange(0, D)
    partial = tl.load(
        partial_o_ptr + h * (num_chunks * D) + r[:, None] * D + d[None, :],
        mask=valid_r[:, None],
        other=0.0,
    )
    numerator = tl.sum(partial * weight[:, None], axis=0)
    tl.store(out_ptr + h * D + d, (numerator / denom).to(tl.bfloat16))
