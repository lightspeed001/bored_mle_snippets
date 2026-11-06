// Maps: T6 (router) + T7 (dispatch/combine) -> DAG nodes g10, g11 -> E11.
// One CTA produced top-8 (expert_id, weight) pairs per token. Now the token
// must physically travel to whichever GPU OWNS that expert. This is expert
// parallelism (EP): experts are owners, not shards.

// ---- stream setup: compute and communication must OVERLAP ----------------
cudaStream_t compute_stream, comm_stream;
cudaEvent_t dispatched;
NCCL_CHECK(ncclGroupStart());
// Dispatch: each rank sends its routed token-rows to the owner ranks.
// Message is [num_routed_tokens, 7168] BF16 = 14,336 B per token.
NCCL_CHECK(ncclSend(send_buf,  send_count*7168*sizeof(half), ncclHalf,
                    dst_rank, comm, comm_stream));
NCCL_CHECK(ncclRecv(recv_buf,  recv_count*7168*sizeof(half), ncclHalf,
                    src_rank, comm, comm_stream));
NCCL_CHECK(ncclGroupEnd());
cudaEventRecord(dispatched, comm_stream);

// ---- meanwhile, on the compute stream: local experts + shared expert -----
// Grouped GEMM: one tcgen05 kernel per (local expert, token bucket), or a
// single grouped-GEMM kernel with per-expert tile ranges. Kimi's experts are
// 7168x2048 SwiGLU triples; Mistral's are 7168x4096 ("fewer but fatter").
// The SHARED expert (g12) has no dependency on the all-to-all, so schedule
// it FIRST -- it hides dispatch latency for free.
grouped_gemm_kernel<<<grid, block, 0, compute_stream>>>(local_experts, recv_buf);
shared_expert_kernel<<<grid2, block2, 0, compute_stream>>>(u);

// ---- combine: send results back, apply router weights, add to residual ---
cudaStreamWaitEvent(compute_stream, dispatched, 0);   // GRAPH EDGE (T7)
NCCL_CHECK(ncclGroupStart());                         // combine all-to-all
/* ... ncclSend/ncclRecv of expert outputs ... */
NCCL_CHECK(ncclGroupEnd());
weighted_sum_and_add<<<..., compute_stream>>>(out, top_w, residual);

// Kimi's paper chose EP = 16 in TRAINING so all-to-all hides under attention
// during the 1F1B schedule [12](https://arxiv.org/html/2507.20534v1). That is
// a network decision expressed as a model-training hyperparameter. At
// inference the same question becomes: is EP <= NVLink domain size?
