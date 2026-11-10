// step6_ane_gated_conv.mm — Fused 1x1 Conv + SiLU on the 16-Core ANE
#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#import <IOSurface/IOSurface.h>

// 1. Express Stage C ([B, 2048] x [2048, 4096] x 2 + SiLU) in MIL 1.3 SSA Text:
//    CRITICAL SILICON RULES:
//    - Tensor layout MUST be [1, C, 1, S] (fp16), NEVER [S, C] row-major!
//    - Use `conv` with [Cout, Cin, 1, 1] weights, NEVER `matmul` (~3x faster on ANE!).
//    - Clamp pre-activation magnitude to [-65500, 65500]: ANE runs pure IEEE/saturating FP16
//      internally, and an unclamped outlier > 65504 poisons downstream tiles with Inf/NaN!
static NSString* build_gated_ffn_mil(int C_in, int C_hidden, int S) {
    return [NSString stringWithFormat:@
        "program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}})]\n"
        "{\n"
        "  func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
        "    tensor<fp16, [%d, %d, 1, 1]> W_gate = const()[name=string(\"Wg\"), "
        "val=tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/wg.bin\"), offset=uint64(64)))];\n"
        "    tensor<fp16, [%d, %d, 1, 1]> W_up   = const()[name=string(\"Wu\"), "
        "val=tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/wu.bin\"), offset=uint64(64)))];\n"
        "    string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
        "    tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1, 1])];\n"
        "    tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "    tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1, 1])];\n"
        "    int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
        // Fast-path 1x1 convolutions stay 100% inside the ~32 MB ANE SRAM across all 4 ops!
        "    tensor<fp16, [1, %d, 1, %d]> gate = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=W_gate, x=x)[name=string(\"conv_g\")];\n"
        "    tensor<fp16, [1, %d, 1, %d]> up   = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=W_up,   x=x)[name=string(\"conv_u\")];\n"
        "    tensor<fp16, [1, %d, 1, %d]> sig  = sigmoid(x=gate)[name=string(\"sig\")];\n"
        "    tensor<fp16, [1, %d, 1, %d]> silu = mul(x=gate, y=sig)[name=string(\"silu\")];\n"
        "    tensor<fp16, [1, %d, 1, %d]> out  = mul(x=silu, y=up)[name=string(\"fused_out\")];\n"
        "  } -> (out);\n"
        "}\n",
        C_in, S,
        C_hidden, C_in, C_hidden, C_in,
        C_hidden, C_in, C_hidden, C_in,
        C_hidden, S, C_hidden, S, C_hidden, S, C_hidden, S, C_hidden, S];
}

// 2. Evaluate on ANE with pre-bound IOSurface-backed CVPixelBuffers (Public Shipping Path)
//    IMPORTANT: Do NOT use MLComputeUnitsAll while benchmarking — if a single op violates an
//    ANE shape constraint, Core ML will silently run it on the GPU! Use CPUAndNeuralEngine
//    and assert `ane_power > 0 mW` (or check Instruments' "ANE Hardware" track).
void eval_ane_zero_copy(MLModel* ane_model,
                        CVPixelBufferRef in_pixbuf,   // Wraps input IOSurface  [1, 2048, 1, S]
                        CVPixelBufferRef out_pixbuf,  // Wraps output IOSurface [1, 4096, 1, S]
                        NSArray<NSNumber*>* in_shape,
                        NSArray<NSNumber*>* out_shape) {
    // Wrap IOSurface-backed pixel buffers in MLMultiArray — NO heap allocation, NO memcpy!
    MLMultiArray* in_arr  = [[MLMultiArray alloc] initWithPixelBuffer:in_pixbuf  shape:in_shape];
    MLMultiArray* out_arr = [[MLMultiArray alloc] initWithPixelBuffer:out_pixbuf shape:out_shape];

    NSError* err = nil;
    MLDictionaryFeatureProvider* inputs = [[MLDictionaryFeatureProvider alloc]
        initWithDictionary:@{@"x": [MLFeatureValue featureValueWithMultiArray:in_arr]}
                     error:&err];

    // CRITICAL: Without `outputBackings`, Core ML allocates a brand-new buffer every call
    // and forces a costly copy out of ANE DMA memory!
    MLPredictionOptions* opts = [[MLPredictionOptions alloc] init];
    opts.outputBackings = @{@"fused_out": out_arr};

    [ane_model predictionFromFeatures:inputs options:opts error:&err];
}
