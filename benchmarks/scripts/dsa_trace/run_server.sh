#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-}"
if [[ -z "${MODEL}" ]]; then
  echo "Usage: $0 <model_path_or_id> [port]"
  exit 1
fi
PORT="${2:-8000}"

export VLLM_ASCEND_DSA_TRACE="${VLLM_ASCEND_DSA_TRACE:-1}"
export VLLM_ASCEND_TRACE_DIR="${VLLM_ASCEND_TRACE_DIR:-./trace_out}"
export VLLM_ASCEND_DSA_TRACE_DECODE_ONLY="${VLLM_ASCEND_DSA_TRACE_DECODE_ONLY:-1}"

echo "Tracing enabled:"
echo "  VLLM_ASCEND_DSA_TRACE=${VLLM_ASCEND_DSA_TRACE}"
echo "  VLLM_ASCEND_TRACE_DIR=${VLLM_ASCEND_TRACE_DIR}"
echo "  VLLM_ASCEND_DSA_TRACE_DECODE_ONLY=${VLLM_ASCEND_DSA_TRACE_DECODE_ONLY}"

echo "Starting server on port ${PORT}..."
exec vllm serve "${MODEL}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code

