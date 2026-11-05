# 1. Compile with source mapping, or the profiler is blind:
nvcc -arch=sm_89 -lineinfo demo.cu -o demo

# 2. Who eats the time? (whole app, CLI)
nsys profile -o app ./demo && nsys stats --report cuda_gpu_kern_sum app.nsys-rep

# 3. The microscope, targeted:
ncu --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis \
    --section WarpStateStats --launch-skip 10 --launch-count 3 ./demo

# 4. Part 1's metric table, one shot, CSV for before/after diffs:
ncu --csv --metrics sm__warps_active.avg.pct_of_peak_sustained_active,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
launch__registers_per_thread,local_ld_requests,local_st_requests,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio ./demo

# 5. Did the compiler spill?
cuobjdump -res-usage ./demo        # REG:xx SPILL:yy  — yy>0 is bad news

# 6. Is it even correct? (run before trusting any timing)
compute-sanitizer ./demo           # memcheck + racecheck + initcheck + synccheck

# 7. Memory bugs, individually:
compute-sanitizer --tool racecheck ./demo

# 8. Logic bugs, interactively:
cuda-gdb ./demo                    # break file.cu:N, condition 1 threadIdx.x==K, run, print ...

# 9. Error-check every launch in code:
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

# 10. Multi-GPU / MPI scale:
tau_exec -T cupti -cuda mpirun -np 32 ./demo
