

# The counters that answer "is this coalesced, and am I near the ceiling?"
# Run timing separately from profiling: NCU replays kernels and its wall
# clock is meaningless for throughput numbers.
ncu --metrics \
  gpu__time_duration.sum,\
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
  dram__throughput.avg.pct_of_peak_sustained_elapsed,\
  dram__bytes_read.sum,dram__bytes_write.sum,\
  l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
  l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,\
  lts__t_sectors_srcunit_tex_op_read.sum,\
  lts__t_sector_hit_rate.pct,\
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
  sm__warps_active.avg.pct_of_peak_sustained_active,\
  launch__registers_per_thread,\
  smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct \
  --csv -o results ./bench --experiment E2

# Reading the output:
#   sectors/requests = 4    perfect coalescing (32-bit loads)
#   sectors/requests = 32   fully uncoalesced
#   dram__throughput > 70%  bandwidth-bound
#   SM% and DRAM% < 40%     latency-bound -> add bytes in flight, not warps
#
# CAVEAT specific to this article: the l1tex__* sector counters describe the
# SIMT/LSU path. TMA bypasses L1, so use dram__* and lts__* for TMA traffic
# and say so in your post. Also note a reported NCU 2025.3.1 hang on
# InstructionStats/WarpStateStats for TMA+mbarrier kernels on B200; use
# --set subsets if you hit it:
# https://forums.developer.nvidia.com/t/nsight-compute-hangs-on-instructionstats-warpstatestats-when-profiling-tma-mbarrier-kernels-on-blackwell/35993
