#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-}"
if [[ -z "${MODEL}" ]]; then
  echo "Usage: $0 <model_path_or_id> [port]"
  echo ""
  echo "Optional env overrides:"
  echo "  TP_SIZE=16 MAX_MODEL_LEN=100000 GPU_MEMORY_UTILIZATION=0.97 QUANTIZATION=ascend"
  echo "  ADDITIONAL_CONFIG_JSON={\"ascend_scheduler_config\":{\"enabled\":false},\"torchair_graph_config\":{\"enabled\":true}}"
  exit 1
fi
PORT="${2:-8000}"

# Trace envs
export VLLM_ASCEND_DSA_TRACE="${VLLM_ASCEND_DSA_TRACE:-1}"
export VLLM_ASCEND_TRACE_DIR="${VLLM_ASCEND_TRACE_DIR:-./trace_out}"
export VLLM_ASCEND_DSA_TRACE_DECODE_ONLY="${VLLM_ASCEND_DSA_TRACE_DECODE_ONLY:-1}"

# Optional trace labeling (useful for multi-process/multi-node)
export VLLM_ASCEND_TRACE_TAG="${VLLM_ASCEND_TRACE_TAG:-}"

# One run id per server start: all workers should share it.
# If you run multi-node, set this on node0 and export the same value on other nodes.
if [[ -z "${VLLM_ASCEND_TRACE_RUN_ID:-}" ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  if [[ -n "${VLLM_ASCEND_TRACE_TAG}" ]]; then
    export VLLM_ASCEND_TRACE_RUN_ID="dsa_${TS}_${VLLM_ASCEND_TRACE_TAG}"
  else
    export VLLM_ASCEND_TRACE_RUN_ID="dsa_${TS}"
  fi
fi

# Defaults aligned with Ascend A3 single-node (16 NPU) serving.
TP_SIZE="${TP_SIZE:-16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-100000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.97}"
QUANTIZATION="${QUANTIZATION:-ascend}"
ADDITIONAL_CONFIG_JSON="${ADDITIONAL_CONFIG_JSON:-{\"ascend_scheduler_config\":{\"enabled\":false},\"torchair_graph_config\":{\"enabled\":true}}}"

echo "Tracing enabled:"
echo "  VLLM_ASCEND_DSA_TRACE=${VLLM_ASCEND_DSA_TRACE}"
echo "  VLLM_ASCEND_TRACE_DIR=${VLLM_ASCEND_TRACE_DIR}"
echo "  VLLM_ASCEND_DSA_TRACE_DECODE_ONLY=${VLLM_ASCEND_DSA_TRACE_DECODE_ONLY}"
echo "  VLLM_ASCEND_TRACE_RUN_ID=${VLLM_ASCEND_TRACE_RUN_ID}"
echo "  VLLM_ASCEND_TRACE_TAG=${VLLM_ASCEND_TRACE_TAG}"

echo "Starting server on port ${PORT}..."
echo "  MODEL=${MODEL}"
echo "  TP_SIZE=${TP_SIZE}"
echo "  MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "  GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
echo "  QUANTIZATION=${QUANTIZATION}"

echo "  ADDITIONAL_CONFIG_JSON=${ADDITIONAL_CONFIG_JSON}"

exec vllm serve "${MODEL}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --tensor-parallel-size "${TP_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --quantization "${QUANTIZATION}" \
  --additional-config "${ADDITIONAL_CONFIG_JSON}"
