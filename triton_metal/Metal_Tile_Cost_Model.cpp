// Pseudocode: tile selection belongs in the target-aware compiler pipeline.
struct AttentionTile { int BQ, BK, D; int numSimdgroups; };

int64_t scratchBytes(const AttentionTile &t, const AttentionConfig &c) {
  // Use checked arithmetic in production; large static shapes can overflow.
  int64_t scores = int64_t(t.BQ) * t.BK * sizeof(float);
  int64_t probs  = c.keepProbabilities ? int64_t(t.BQ) * t.BK * sizeof(half) : 0;
  int64_t rows   = int64_t(t.BQ) * (2 * sizeof(float)); // running max + denominator
  int64_t stage  = c.stageK ? int64_t(t.BK) * t.D * sizeof(half) : 0;
  return alignUp(scores, 16) + alignUp(probs, 16)
       + alignUp(rows, 16) + alignUp(stage, 16) + c.reductionScratch;
}

FailureOr<AttentionTile> chooseTile(Shape s, DeviceCaps caps) {
  for (AttentionTile t : candidatesFor(s)) {
    if (!supportsDtypeAndDot(t, caps)) continue;
    if (scratchBytes(t, configFor(s)) > caps.maxThreadgroupBytes) continue;
    if (!hasLegalReductionScope(t, caps)) continue;
    // Prefer the biggest tile only after checking occupancy and register pressure.
    if (estimatedRegisters(t) < caps.registerBudgetPerThread) return t;
  }
  return failure(); // invoke smaller fallback or emit actionable diagnostic
}
