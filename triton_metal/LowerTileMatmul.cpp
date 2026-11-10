// Pseudocode: not a compilable pass; omits version-specific Triton/MLIR plumbing.
struct LowerTileMatmul : OpRewritePattern<linalg::MatmulOp> {
  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    // 1. Keep the operation logical while validating rank, dtype, masks, and strides.
    auto shape = inferStaticShape(op);
    if (!shape || !isSupportedType(shape->inputType, shape->accType))
      return rewriter.notifyMatchFailure(op, "unsupported shape or type");

    // 2. Pick a legal target plan, not an NVIDIA MMA layout by analogy.
    MetalTilePlan plan = chooseAppleTile(shape, /*threadgroup_budget=*/32_KiB);
    if (!plan.isSupported())
      return rewriter.notifyMatchFailure(op, plan.reason());

    // 3. Materialize explicit Metal-dialect concepts:
    //    tile coordinates, SIMD-group distribution, optional scratch, dot primitive.
    Value aTile = emitMetalTileLoad(rewriter, op.getInputs()[0], plan.a);
    Value bTile = emitMetalTileLoad(rewriter, op.getInputs()[1], plan.b);
    Value acc   = emitMetalAccumulator(rewriter, plan);
    acc = emitMetalDot(rewriter, aTile, bTile, acc, plan.dotKind);

    // 4. Preserve masks/strides and keep the epilogue fused when it is legal.
    emitMaskedStoreAndEpilogue(rewriter, op, acc, plan);
    rewriter.eraseOp(op);
    return success();
  }
};

// Design rule: a failed conversion is a diagnostic, never a silently wrong kernel.
// Keep a scalar/SIMD fallback for unsupported layouts until the fast path is proven.
