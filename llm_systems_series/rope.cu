// Maps: E6/E7 (RoPE) -> DAG node g4 -> SIMT FMA on CUDA cores.
// Runs inside the Q/K projection epilogue, so q and k never visit HBM raw.
// Kimi/Mistral MLA: rope_dim = 64 (DECOUPLED, one key shared across heads);
// GQA control Qwen3: rope_dim = d_h = 128, applied per head.
__device__ __forceinline__ void rope_rotate(half2& v, int pos, int m,
                                            const float* cosT, const float* sinT,
                                            int stride) {
    // v packs channels (2m, 2m+1). One lookup, four FMAs, one store.
    const float c = cosT[pos * stride + m];
    const float s = sinT[pos * stride + m];
    float2 f = __half22float2(v);
    // R(i*theta_m) * [f.x, f.y]  -- the entire mathematical content of E6.
    float nx = f.x * c - f.y * s;
    float ny = f.x * s + f.y * c;
    v = __float22half2_rn(make_float2(nx, ny));
}

// YaRN changes ONLY the table: theta_m -> interpolated frequency. The kernel
// above is identical at 4K and at 256K context. No new node in the DAG; a
// different constant in L2. That is a nice example of "architecture vs config".
//
// TRAP FOR THE READER: do NOT rotate the value vector v, and do NOT rotate the
// residual stream. RoPE is a Q/K-only operation; rotating V breaks E7's
// relative-position argument and silently destroys long-context behaviour.
