// step9_lockfree_heterogeneous_ring.mm — Concurrent SME2 + ANE + M5 GPU NAX Pipeline
#import <Metal/Metal.h>
#import <CoreML/CoreML.h>
#include <dispatch/dispatch.h>

constexpr size_t RING_SLOTS = 3; // Sized strictly for SLC residency (see Step 10!)

class WholeSoCPipeline {
    id<MTLDevice>              dev_;
    id<MTLCommandQueue>        gpu_queue_;
    id<MTLComputePipelineState> gpu_nax_pso_;
    MLModel*                   ane_ffn_model_;

    // Timeline 1: Monotonic stage handoff (2*t + 1 = SME done; 2*t + 2 = ANE done)
    id<MTLSharedEvent>         stage_evt_;
    // Timeline 2: Backpressure credit counter (signals when GPU frees Slot[t % 3])
    id<MTLSharedEvent>         free_evt_;
    MTLSharedEventListener*    ane_listener_;
    dispatch_queue_t           ane_gcd_q_;

    TriEngineSurface           in_ring_[RING_SLOTS];
    TriEngineSurface           ane_out_ring_[RING_SLOTS];

public:
    void init_queues_and_events() {
        stage_evt_ = [dev_ newSharedEvent];
        free_evt_  = [dev_ newSharedEvent];
        free_evt_.signaledValue = RING_SLOTS; // All 3 ring slots start free!

        // CRITICAL: ANE listener queue MUST carry QOS_CLASS_USER_INTERACTIVE!
        // Default QoS queues suffer 150–800 µs XNU wake jitter under heavy GPU load.
        dispatch_queue_attr_t attr = dispatch_queue_attr_make_with_qos_class(
            DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INTERACTIVE, 0);
        ane_gcd_q_    = dispatch_queue_create("soc.ane.worker", attr);
        ane_listener_ = [[MTLSharedEventListener alloc] initWithDispatchQueue:ane_gcd_q_];
    }

    // Submits microbatch `t` across all 3 engines without a single blocking waitUntilCompleted!
    void submit_microbatch_async(uint64_t t, const __fp16* raw_tokens, size_t S, size_t C) {
        const size_t slot = t % RING_SLOTS;

        // 0. Backpressure: wait only if CPU gets >3 microbatches ahead of GPU (protects ring slot)
        while (free_evt_.signaledValue <= t) {
            __builtin_arm_wfe(); // Ultra-low-power ARM64 Wait-For-Event hint
        }

        // 1. PRE-SUBMIT GPU COMMAND BUFFER TO AGX FIRMWARE *BEFORE* SME & ANE EVEN FINISH!
        //    AGX hardware waits on `stage_evt_ == 2*t + 2` on-chip — zero ~250 µs wakeup tax!
        id<MTLCommandBuffer> cmd = [gpu_queue_ commandBufferWithUnretainedReferences];
        [cmd encodeWaitForEvent:stage_evt_ value:(2 * t + 2)];

        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:gpu_nax_pso_];
        [enc setBuffer:in_ring_[slot].gpu_buf      offset:0 atIndex:0]; // Reads [1,C,1,S] directly!
        [enc setBuffer:ane_out_ring_[slot].gpu_buf offset:0 atIndex:1]; // Combines ANE Shared FFN!
        // ... bind routed expert weights & dispatch Metal 4 mpp::tensor_ops NAX threadgroups ...
        [enc endEncoding];

        // When GPU finishes microbatch `t`, AGX hardware bumps `free_evt_` to recycle `slot`
        [cmd encodeSignalEvent:free_evt_ value:(t + RING_SLOTS + 1)];
        [cmd commit];

        // 2. ARM ASYNC ANE LISTENER FOR MICROBATCH `t` (wakes the instant SME signals 2*t + 1)
        [stage_evt_ notifyListener:ane_listener_
                           atValue:(2 * t + 1)
                             block:^(id<MTLSharedEvent> ev, uint64_t) {
            eval_ane_zero_copy(ane_ffn_model_,
                               in_ring_[slot].ane_pixbuf,
                               ane_out_ring_[slot].ane_pixbuf,
                               @[@1, @(C), @1, @(in_ring_[slot].s_stride)],
                               @[@1, @(4096), @1, @(ane_out_ring_[slot].s_stride)]);
            // Unblock pre-parked GPU command buffer in AGX hardware!
            ev.signaledValue = 2 * t + 2;
        }];

        // 3. EXECUTE STAGE A (Fused Planar RMSNorm) + STAGE B (SME2 Router) INLINE ON P-CORE
        rmsnorm_fused_to_planar_slc(raw_tokens, /*gamma=*/nullptr,
                                    in_ring_[slot].cpu_ptr, S, C, in_ring_[slot].s_stride);
        // ... sme_gemm_32x32_fp32 router gate on P-core ...

        // Ensure CPU store buffers drain to Shared SLC before signaling ANE & GPU DMA engines!
        __asm__ volatile("dmb ishst" ::: "memory");
        stage_evt_.signaledValue = 2 * t + 1;
    }
};
