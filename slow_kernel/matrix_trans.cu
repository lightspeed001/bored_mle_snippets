#define TILE 32

// BAD: tile[32][32] has row stride 32 == 32 banks.
// Reading a COLUMN (tile[x][y], fixed y... actually fixed threadIdx.y row read)
// makes all 32 lanes hit the SAME bank -> 32-way conflict -> 32 serialized cycles.
__global__ void transpose_bad(const float* in, float* out, int N) {
    __shared__ float tile[TILE][TILE];                 // stride 32: the trap
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    tile[threadIdx.y][threadIdx.x] = in[y * N + x];   // coalesced global read, fine
    __syncthreads();                                   // everyone sees the tile
    out[x * N + y] = tile[threadIdx.x][threadIdx.y];   // column read: 32-way bank conflict
}

// GOOD: pad one word per row. Stride 33 is coprime with 32 banks,
// so consecutive rows start on different banks -> conflict-free.
__global__ void transpose_good(const float* in, float* out, int N) {
    __shared__ float tile[TILE][TILE + 1];             // stride 33: costs 3% extra smem
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    tile[threadIdx.y][threadIdx.x] = in[y * N + x];
    __syncthreads();
    out[x * N + y] = tile[threadIdx.x][threadIdx.y];   // now one wavefront, 1 cycle
}
