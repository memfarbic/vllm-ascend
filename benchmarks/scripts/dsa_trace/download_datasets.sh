#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DATA_DIR="${DATA_DIR:-"${ROOT_DIR}/benchmarks/datasets/dsa_trace"}"

# curl behavior controls (safe by default)
# - VLLM_ASCEND_CURL_CA_BUNDLE: path to a CA certificate bundle (recommended in corp/self-signed env)
# - VLLM_ASCEND_CURL_INSECURE=1: pass -k/--insecure (NOT recommended; disables TLS verification)
# - VLLM_ASCEND_CURL_EXTRA_ARGS: extra args appended to curl
VLLM_ASCEND_CURL_CA_BUNDLE="${VLLM_ASCEND_CURL_CA_BUNDLE:-}"
VLLM_ASCEND_CURL_INSECURE="${VLLM_ASCEND_CURL_INSECURE:-1}"
VLLM_ASCEND_CURL_EXTRA_ARGS="${VLLM_ASCEND_CURL_EXTRA_ARGS:-}"

# Override download URLs if needed.
SHAREGPT_URL="${SHAREGPT_URL:-https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json}"
BURSTGPT_URL="${BURSTGPT_URL:-https://github.com/HPMLL/BurstGPT/releases/download/v1.1/BurstGPT_without_fails_2.csv}"

mkdir -p "${DATA_DIR}"

echo "Downloading datasets into: ${DATA_DIR}"

curl_fetch() {
  local url="$1"
  local out="$2"

  local -a args
  args=(
    -L
    --fail
    --retry 3
    --retry-delay 2
    --connect-timeout 20
    --max-time 0
    -o "$out"
  )

  if [[ -n "${VLLM_ASCEND_CURL_CA_BUNDLE}" ]]; then
    args+=( --cacert "${VLLM_ASCEND_CURL_CA_BUNDLE}" )
  fi
  if [[ "${VLLM_ASCEND_CURL_INSECURE}" == "1" ]]; then
    args+=( -k )
  fi
  if [[ -n "${VLLM_ASCEND_CURL_EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    args+=( ${VLLM_ASCEND_CURL_EXTRA_ARGS} )
  fi

  echo "curl ${url} -> ${out}"
  if ! curl "${args[@]}" "$url"; then
    echo "ERROR: download failed: ${url}" >&2
    echo "If you want to ENABLE TLS verification (recommended), set:" >&2
    echo "  export VLLM_ASCEND_CURL_INSECURE=0
  export VLLM_ASCEND_CURL_CA_BUNDLE=/path/to/ca-bundle.pem  # optional but recommended in corp env" >&2
    echo "Current default is INSECURE (VLLM_ASCEND_CURL_INSECURE=1)." >&2
    echo "You can keep it, or explicitly set it:
  export VLLM_ASCEND_CURL_INSECURE=1" >&2
    return 1
  fi
}

# ShareGPT
if [[ ! -f "${DATA_DIR}/ShareGPT_V3_unfiltered_cleaned_split.json" ]]; then
  echo "Downloading ShareGPT..."
  curl_fetch "${SHAREGPT_URL}" "${DATA_DIR}/ShareGPT_V3_unfiltered_cleaned_split.json"
else
  echo "ShareGPT already exists."
fi

# BurstGPT
if [[ ! -f "${DATA_DIR}/BurstGPT_without_fails_2.csv" ]]; then
  echo "Downloading BurstGPT..."
  curl_fetch "${BURSTGPT_URL}" "${DATA_DIR}/BurstGPT_without_fails_2.csv"
else
  echo "BurstGPT already exists."
fi

echo "Done."
