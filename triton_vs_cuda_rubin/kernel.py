import triton
import triton.language as tl


@triton.jit
def grouped_w4a16_triton(
    x_ptr, w_ptr, s_ptr,
    offsets_ptr, active_experts_ptr, token_ids_ptr,
    y_ptr,
    N,
    K: tl.constexpr,
    GROUP: tl.constexpr,    # 128
    BLOCK_T: tl.constexpr,  # 4 assignments in the primary fast path
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    active_i = tl.program_id(1)

    # Compact active list: inactive experts never create weight-reading programs.
    expert = tl.load(active_experts_ptr + active_i).to(tl.int64)
    start = tl.load(offsets_ptr + expert)
    count = tl.load(offsets_ptr + expert + 1) - start

    t = tl.arange(0, BLOCK_T)
    token_ok = t < count
    tokens = tl.load(token_ids_ptr + start + t, mask=token_ok, other=0)

    rows = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_ok = rows < N
    pairs = tl.arange(0, GROUP // 2)
    acc = tl.zeros((BLOCK_T, BLOCK_N), dtype=tl.float32)

    for g in range(0, K // GROUP):
        # [BLOCK_N, 64]: loaded once and shared by every live assignment.
        packed = tl.load(
            w_ptr + expert * (N * (K // 2))
                  + rows[:, None] * (K // 2)
                  + g * (GROUP // 2) + pairs[None, :],
            mask=row_ok[:, None],
            other=0,
        )
        lo = (packed & 0x0F).to(tl.int32)
        hi = ((packed >> 4) & 0x0F).to(tl.int32)
        lo = tl.where(lo >= 8, lo - 16, lo)
        hi = tl.where(hi >= 8, hi - 16, hi)

        # [BLOCK_T, 64]: each row belongs to a possibly different original token.
        x0 = tl.load(
            x_ptr + tokens[:, None] * K + g * GROUP + 2 * pairs[None, :],
            mask=token_ok[:, None],
            other=0.0,
        ).to(tl.float32)
        x1 = tl.load(
            x_ptr + tokens[:, None] * K + g * GROUP + 2 * pairs[None, :] + 1,
            mask=token_ok[:, None],
            other=0.0,
        ).to(tl.float32)

        scale = tl.load(
            s_ptr + expert * (N * (K // GROUP))
                  + rows * (K // GROUP) + g,
            mask=row_ok,
            other=0.0,
        ).to(tl.float32)

        # [BLOCK_T, BLOCK_N, 64]: weight values are reused along BLOCK_T.
        # Watch register pressure carefully; larger BLOCK_N is not always better.
        partial = (x0[:, None, :] * lo.to(tl.float32)[None, :, :]
                 + x1[:, None, :] * hi.to(tl.float32)[None, :, :])
        group_dot = tl.sum(partial, axis=2)
        acc += group_dot * scale[None, :]

    # Output remains sorted by expert; no atomic scatter is required here.
    tl.store(
        y_ptr + (start + t)[:, None] * N + rows[None, :],
        acc.to(tl.bfloat16),
        mask=token_ok[:, None] & row_ok[None, :],
    )
