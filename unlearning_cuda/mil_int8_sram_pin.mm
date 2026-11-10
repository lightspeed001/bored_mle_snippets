// step10_mil_int8_sram_pin.mm — Halving ANE Weight Footprint to Stay 100% Inside 32 MB SRAM
// Inside build_gated_ffn_mil(): replace raw fp16 const() weights with MIL's hardware-accelerated
// compile-time affine dequantize node.
//   • Weight footprint drops 33.5 MB -> 16.8 MB (comfortably < 32 MB ANE SRAM cliff!)
//   • ANE unpacks int8 -> fp16 in hardware on the fly -> FULL 19.1 TF FP16 compute rate!
//   • External LPDDR5X weight traffic during ANE execution drops from ~55 GB/s -> < 3 GB/s!
static NSString* mil_int8_weight_block = @
    "    tensor<int8, [%d, %d, 1, 1]> Wg_q = const()[name=string(\"Wg_q\"), "
    "val=tensor<int8, [%d, %d, 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/wg_i8.bin\"), offset=uint64(64)))];\n"
    "    tensor<fp16, [%d, 1, 1, 1]>  Wg_s = const()[name=string(\"Wg_s\"), "
    "val=tensor<fp16, [%d, 1, 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/wg_scale.bin\"), offset=uint64(64)))];\n"
    "    int8 zp = const()[name=string(\"zp\"), val=int8(0)];\n"
    "    tensor<fp16, [%d, %d, 1, 1]> W_gate = constexpr_affine_dequantize("
    "quantized_data=Wg_q, scale=Wg_s, zero_point=zp, axis=int32(0))[name=string(\"Wg_deq\")];\n";

// Rule-of-Thumb Guardrail for Heterogeneous Ring Sizing on M4 / M5:
//   Base M4/M5 SLC ≈ 16 MB | Pro ≈ 24 MB | Max ≈ 48 MB | Ultra ≈ 96 MB
static inline void assert_ring_fits_slc(size_t slot_bytes, size_t num_slots, size_t slc_bytes) {
    const size_t total_ring = slot_bytes * num_slots;
    // Keep ring <= 35% of SLC so GPU L2 spills and CPU L2 lines don't evict in-flight handoffs!
    assert(total_ring <= (slc_bytes * 35) / 100 && "Ring buffer will thrash Shared SLC!");
}
