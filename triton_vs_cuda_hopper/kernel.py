import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 4},  num_warps=2, num_stages=2),
        triton.Config({"BLOCK_N": 8},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 8},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 16}, num_warps=8, num_stages=2),
    ],
    key=["N"],
)
@triton.jit
def w4a16_gemv_triton(
    x_ptr, w_ptr, s_ptr, y_ptr,
    N,
    K: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    row_ok = rows < N

    # GROUP=128 means 64 packed bytes and 64 activation pairs per group.
    pairs = tl.arange(0, GROUP // 2)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for g in range(0, K // GROUP):
        # [BLOCK_N, 64]: rows are independent; bytes within a row are contiguous.
        packed = tl.load(
            w_ptr + rows[:, None] * (K // 2)
                  + g * (GROUP // 2) + pairs[None, :],
            mask=row_ok[:, None],
            other=0,
        )

        # Widen before subtracting 16: unsigned nibbles must not wrap around.
        lo = (packed & 0x0F).to(tl.int32)
        hi = ((packed >> 4) & 0x0F).to(tl.int32)
        lo = tl.where(lo >= 8, lo - 16, lo)
        hi = tl.where(hi >= 8, hi - 16, hi)

        # These are strided element streams, but together cover each cache line.
        # Inspect PTX/SASS rather than assuming the emitted loads are ideal.
        x0 = tl.load(x_ptr + g * GROUP + 2 * pairs).to(tl.float32)
        x1 = tl.load(x_ptr + g * GROUP + 2 * pairs + 1).to(tl.float32)

        scale = tl.load(
            s_ptr + rows * (K // GROUP) + g,
            mask=row_ok,
            other=0.0,
        ).to(tl.float32)

        # FP32 arithmetic avoids BF16 accumulation error across K=4096.
        partial = (lo.to(tl.float32) * x0[None, :]
                 + hi.to(tl.float32) * x1[None, :])
        acc += scale * tl.sum(partial, axis=1)

    tl.store(y_ptr + rows, acc.to(tl.bfloat16), mask=row_ok)


def launch_triton(x, packed, scales, y):
    n, k2 = packed.shape
    k = 2 * k2
    assert k % 128 == 0
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_N"]),)
    w4a16_gemv_triton[grid](x, packed, scales, y, n, K=k, GROUP=128)
