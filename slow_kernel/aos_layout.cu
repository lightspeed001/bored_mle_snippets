// BAD: AoS. Each Particle is 32 bytes, so lane i reading p[i].x touches
// a different 32B sector -> the warp fetches 32 sectors to use 128 bytes. 8x waste.
struct Particle { float x, y, z, w, vx, vy, vz, vw; };   // 32 bytes
__global__ void step_bad(Particle* p, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) p[i].x += p[i].vx;                        // stride-32B access across lanes
}

// GOOD: SoA — structure of arrays. Consecutive lanes read consecutive floats:
// one 128B line per 8 lanes... i.e. the whole warp in 4 lines. Fully coalesced.
struct Particles { float *x, *y, *z, *w, *vx, *vy, *vz, *vw; };
__global__ void step_good(Particles p, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) p.x[i] += p.vx[i];                        // contiguous: 4 sectors per warp
}

// EVEN BETTER: vectorized loads. One lane moves 16 bytes (float4),
// and the warp still coalesces perfectly — 2 lines per warp for 512B of data.
__global__ void step_vec(const float4* in, float4* out, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        float4 v = in[i];          // single 16B load per lane, requires 16B alignment!
        v.x += v.z;  v.y += v.w;   // (silly math stands in for your real work)
        out[i] = v;
    }
}
