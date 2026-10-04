#!/usr/bin/env bash
# Local OpenAI-compatible server used for the Qwen3.5 runs (4 GPUs, one replica per GPU).
# Usage: serving/serve_vllm.sh <model path or HF id> <served name>, e.g.
#   serving/serve_vllm.sh Qwen/Qwen3.5-4B Qwen/Qwen3.5-4B
# vLLM 0.29.0; the experiments set enable_thinking=false per request (chat_template_kwargs).
set -euo pipefail
MODEL=${1:?model path or HF id}; NAME=${2:-$1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
exec vllm serve "$MODEL" \
  --served-model-name "$NAME" \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --data-parallel-size 4 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 65536 \
  --max-num-seqs 64 \
  --language-model-only \
  --reasoning-parser qwen3
