// Illustrative MSL-style pseudocode. Replace dotTile() with a validated target path:
// e.g. SIMD-group matrix ops or an available Metal 4 tensor operation.
kernel void flash_attention_tile(
    device const half* Q  [[buffer(0)]],
    device const half* K  [[buffer(1)]],
    device const half* V  [[buffer(2)]],
    device half* O        [[buffer(3)]],
    constant Params& p    [[buffer(4)]],
    uint tid              [[thread_index_in_threadgroup]]) {

  // Choose conservative tiles first; compile-time specialization may tune these.
  constexpr uint BQ = 16, BK = 32, D = 64;
  threadgroup float score[BQ * BK];      // 16*32*4 = 2 KiB
  threadgroup half  prob[BQ * BK];       // 16*32*2 = 1 KiB
  threadgroup float rowMax[BQ];
  threadgroup float rowSum[BQ];
  // Additional K/V staging and reduction scratch must be included in cost model!

  // Each output row owns online-softmax state; initialize once before KV loop.
  float m = -INFINITY;
  float l = 0.0f;
  float acc[D];
  for (uint d = 0; d < D; ++d) acc[d] = 0.0f;

  for (uint kb = 0; kb < p.sequenceLength; kb += BK) {
    // The backend maps these logical tiles to a legal SIMD-group/tensor primitive.
    // Mask partial tiles and causal positions before softmax.
    dotTileQK(Q, K, p, kb, score);              // scores = Q_tile * K_tile^T
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float blockMax = maskedRowMax(score, p, kb);
    float mNext = max(m, blockMax);
    float alpha = exp(m - mNext);
    float denomBlock = 0.0f;
    for (uint j = 0; j < BK; ++j) {
      float x = maskedScore(score, p, kb, j);
      float e = exp(x - mNext);
      probForThisRow[j] = half(e);
      denomBlock += e;
    }
    l = l * alpha + denomBlock;
    rescale(acc, alpha);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    dotTilePV(prob, V, p, kb, acc);             // acc += P_tile * V_tile
    m = mNext;
  }

  // Normalize once after the final KV tile; store only valid query rows.
  for (uint d = 0; d < D; ++d)
    storeIfInBounds(O, p, d, acc[d] / l);
}
