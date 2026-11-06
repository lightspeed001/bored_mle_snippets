#!/usr/bin/env bash
# Replaces every EST in Part I/II with a MEAS. Nothing here is a benchmark
# result; it is the experiment you run. Pin these three things first:
#   (1) HF checkpoint revision, (2) B200 SKU + power limit, (3) driver/CUDA.
set -euo pipefail
MODEL=moonshotai/Kimi-K2-Instruct   # or mistralai/Mistral-Large-3-675B-Instruct-2512
PROMPT="Explain RoPE in one sentence."

# --- 1. One decode kernel, full counter set -------------------------------
# Look for: dram__throughput high + sm__throughput low  => memory-bound (T1).
ncu --set full --target-processes all \
    --kernel-name-base demangled \
    -o decode_profile \
    python bench_one_step.py --model "$MODEL" --prompt "$PROMPT" --batch 1

# --- 2. Batch sweep: locate the ridge ------------------------------------
# Theory says the crossover is B ~= 281 concurrent tokens. Plot TPOT vs B.
for B in 1 8 32 64 128 256 384 512; do
  python bench_one_step.py --model "$MODEL" --prompt "$PROMPT" --batch "$B" \
    --csv >> sweep.csv
done

# --- 3. Fabric: is EP really hiding under compute? -----------------------
# 13.76 MB/token x 256 = 3.52 GB aggregate; per-rank messages ~440 MB.
alltoall_perf -b 3M -e 512M -f 2 -g 8     # NVLink domain
NCCL_TOPO_DUMP_FILE=topo.xml nccl-tests/build/alltoall_perf -b 3M -e 512M -f 2 -g 8

# --- 4. Timeline: are we replaying a CUDA Graph? -------------------------
nsys profile -t cuda,nvtx,nccl --force-overwrite true -o timeline \
    vllm bench serve --model "$MODEL" --prompt "$PROMPT" --max-new-tokens 64

# --- 5. Honest metadata ---------------------------------------------------
nvidia-smi -q -d CLOCK,POWER,TEMPERATURE > hardware.txt
python -c "import json;print(json.load(open('config.json'))['torch_dtype'])" >> hardware.txt
# Then edit the receipt in section 13: EST -> MEAS, with SKU + date + clock.
