// ---- v5, SM100 only (sm_100a). Schematic. ----
// TMEM is allocated in columns (>= 32, power of two), by ONE WARP.
tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [tmem_slot], 128;
//                    └─ returns a 32-bit TMEM base address into an SMEM slot.
//                       TMEM addresses are their own address space.

// Producer (from §4) has already landed tiles A and B in swizzled SMEM.

// Optional: stage operand A SMEM -> TMEM without a register round-trip.
// shape/dtype qualifiers elided; a 64-bit SMEM descriptor encodes base,
// leading/stride byte offsets and swizzle.
tcgen05.cp.cta_group::1.128x256b [tmem_a], smem_desc_a;

// ONE thread issues the whole tile multiply. A may come from TMEM or an SMEM
// descriptor, B from an SMEM descriptor; D (the accumulator) lives in TMEM.
// `p` is the accumulate predicate (false on the first K-tile).
tcgen05.mma.cta_group::1.kind::f16 [tmem_acc], a_desc, b_desc, idesc, p;

// Commit a barrier, keep doing other work, then wait for the MMAs.
tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [mma_bar];
// ... overlap: issue TMA for the next K-tile here ...
tcgen05.wait.cta_group::1 ...               // schematic; exact mnemonic/qualifiers
                                            // depend on the commit mechanism

// Epilogue is the ONLY place registers re-enter the picture.
tcgen05.ld ...  [tmem_acc] → regs → GMEM

tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;
tcgen05.dealloc.cta_group::1.sync.aligned.b32 tmem_base, 128;
