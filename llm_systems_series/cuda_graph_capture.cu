// Maps: L4 runtime schedule -> ALL DAG nodes g1..g13 (x61) -> E1..E12.
// The mathematics is untouched. Only the SUBMISSION mechanism changes.

cudaGraph_t graph;
cudaStream_t stream;
cudaStreamCreate(&stream);

// Capture one full decode step (or one prefill chunk) with realistic shapes.
// In vLLM / SGLang / TRT-LLM this is done once per (shape, batch) bucket;
// dynamic shapes mean several captured graphs, which is why 'bucketed'
// batching exists. A different B can require a different graph.
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

embed_gather<<<..., stream>>>(token_ids, E);        // g1  T0
for (int l = 0; l < 61; ++l) {                      // 61 layer subgraphs
    rmsnorm_fused<<<..., stream>>>(x, gamma1[l]);   // g2  T2 (fused)
    mla_projections<<<..., stream>>>(h, W[l]);      // g3  T1 (tcgen05)
    // g4 RoPE is INSIDE the g3 epilogue -- no kernel here.
    paged_attention<<<..., stream>>>(q, block_table, k_cache, v_cache); // g6 T4/T5
    o_proj<<<..., stream>>>(attn_out, W_O[l]);      // g7  T1
    residual_add<<<..., stream>>>(x, attn_out);     // g8  T2
    rmsnorm_fused<<<..., stream>>>(x, gamma2[l]);   // g9  T2
    moe_gate<<<..., stream>>>(u, W_r[l]);           // g10 T6
    moe_dispatch_alltoall(comm_stream);             // g11 T7 (NCCL)
    grouped_experts<<<..., stream>>>(...);          // g11 T1
    shared_expert<<<..., stream>>>(...);            // g12 T1
    residual_add<<<..., stream>>>(x, expert_out);   // g13 T2
}
final_norm_and_lm_head<<<..., stream>>>(x, W_vocab); // T8
top_p_sample<<<..., stream>>>(logits, &next_token);  // T8

cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&exec, graph, 0);

// Per token, ONE submission instead of ~700. The CPU is now free to run the
// L5 serving scheduler instead of stealing time from the GPU. Note that NCCL
// collectives can be captured too -- but the all-to-all COUNT is data-dependent
// (top-k destinations), so real engines use graph-safe static buffers.
for (int step = 0; step < max_new_tokens; ++step) {
    cudaGraphLaunch(exec, stream);                  // replay, 61 layers
    cudaStreamSynchronize(stream);                  // + sampler, + EOS check
}

// DEBUGGING TIP: if a profile shows ~700 launch gaps, the graph is not being
// replayed. If it shows one giant node with poor overlap, fusion went too far
// and the SIMT warps are starving the MMA issuer. Both are L4/L1 problems,
// neither is a model problem.
