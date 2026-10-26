import triton
import triton.language as tl


@triton.jit
def paged_decode_partial_triton(
    q_ptr, k_ptr, v_ptr, page_ids_ptr,
    partial_o_ptr, partial_m_ptr, partial_l_ptr,
    length, num_chunks, softmax_scale,
    HQ: tl.constexpr,       # 64 query heads
    HKV: tl.constexpr,      # 8 KV heads
    G: tl.constexpr,        # 8 query heads per KV head
    D: tl.constexpr,        # 128
    PAGE: tl.constexpr,     # 16 tokens
    CHUNK: tl.constexpr,    # 1,024 tokens
    BLOCK_H: tl.constexpr,  # 16: pads the live G=8 group for tl.dot
):
    chunk = tl.program_id(0)
    hkv = tl.program_id(1)

    h_in_group = tl.arange(0, BLOCK_H)
    heads = hkv * G + h_in_group
    head_ok = (h_in_group < G) & (heads < HQ)
    d = tl.arange(0, D)

    # [16, 128]: padded head rows load zero and are never stored.
    q = tl.load(
        q_ptr + heads[:, None] * D + d[None, :],
        mask=head_ok[:, None],
        other=0.0,
    )

    m = tl.full((BLOCK_H,), float("-inf"), dtype=tl.float32)
    l = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, D), dtype=tl.float32)

    token_begin = chunk * CHUNK
    token_end = tl.minimum(token_begin + CHUNK, length)
    first_page = token_begin // PAGE
    pages_here = tl.cdiv(token_end - token_begin, PAGE)

    for page_i in range(0, pages_here):
        logical_page = first_page + page_i
        physical_page = tl.load(page_ids_ptr + logical_page).to(tl.int64)
        t = tl.arange(0, PAGE)
        valid_t = logical_page * PAGE + t < length

        # One (physical page, KV head) slab is contiguous [PAGE, D].
        cache_base = (physical_page * HKV + hkv) * (PAGE * D)
        k = tl.load(
            k_ptr + cache_base + t[:, None] * D + d[None, :],
            mask=valid_t[:, None],
            other=0.0,
        )

        # BF16 inputs, FP32 dot accumulation: [16,128] x [128,16].
        scores = tl.dot(q, tl.trans(k)) * softmax_scale
        scores = tl.where(valid_t[None, :], scores, float("-inf"))

        new_m = tl.maximum(m, tl.max(scores, axis=1))
        alpha = tl.exp(m - new_m)
        p = tl.exp(scores - new_m[:, None])

        # Load V only after the QK phase; both tiles reuse the same page mapping.
        v = tl.load(
            v_ptr + cache_base + t[:, None] * D + d[None, :],
            mask=valid_t[:, None],
            other=0.0,
        )

        # BF16 p enables the tensor-core PV path; account for this in accuracy tests.
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        l = l * alpha + tl.sum(p, axis=1)
        m = new_m

    # partial_o: [HQ, R, D] FP32; partial_m/l: [HQ, R] FP32.
    tl.store(
        partial_o_ptr + heads[:, None] * (num_chunks * D)
                      + chunk * D + d[None, :],
        acc,
        mask=head_ok[:, None],
    )
    tl.store(partial_m_ptr + heads * num_chunks + chunk, m, mask=head_ok)
    tl.store(partial_l_ptr + heads * num_chunks + chunk, l, mask=head_ok)
