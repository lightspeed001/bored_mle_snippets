// Maps: T2 + T3 -> DAG nodes g2, g4 -> E2 (RMSNorm), E6/E7 (RoPE).
// In Part I these were standalone kernels. On real hardware they are FUSED
// into the epilogue of the preceding GEMM, because an unfused norm costs one
// full [N x d] HBM read + write per layer (2 x 61 round-trips per token).

template<int d, int rope_dim>
__device__ void epilogue_visitor(float4& acc,          // from tcgen05.ld
                                 const half* gamma,
                                 const float* cosT, const float* sinT,
                                 int pos, float* warp_sums) {
    // (a) E2: sum of squares. Warp shuffle reduction (no SMEM, no barrier),
    //     then rsqrt.approx on the SFU -- the SAME unit that softmax wants.
    float ss = 0.f;
    #pragma unroll
    for (int i = 0; i < d/8; ++i) ss = fmaf(acc[i], acc[i], ss);
    for (int off = 16; off; off >>= 1) ss += __shfl_down_sync(~0u, ss, off);
    float inv = rsqrtf(ss / d + 1e-6f);

    // (b) E2 gain + E6 rotation, in registers, in ONE pass. No HBM traffic.
    //     Only the DECOUPLED 64-D slice is rotated for MLA; the 128-D NoPE
    //     content part stays position-free. This is why the KV cache can be
    //     one 512-D latent plus one 64-D shared rotary key.
    #pragma unroll
    for (int i = 0; i < d/2; ++i) {
        float2 f = make_float2(acc[2*i] * inv * gamma[2*i],
                               acc[2*i+1] * inv * gamma[2*i+1]);
        if (i < rope_dim/2) {
            float c = cosT[pos*rope_dim/2 + i], s = sinT[pos*rope_dim/2 + i];
            f = make_float2(f.x*c - f.y*s, f.x*s + f.y*c);   // R(i*theta_m)
        }
        acc[2*i] = f.x; acc[2*i+1] = f.y;
    }
    // (c) BF16 cast + store. Three equations, zero extra memory moves.
}

// TRAP FOR THE READER: this fusion is legal ONLY because RMSNorm is applied
// to the GEMM output before any other consumer. If the engine chooses the
// decompressed MLA path, the fusion boundary moves. The DAG does not care;
// the epilogue visitor most certainly does.
