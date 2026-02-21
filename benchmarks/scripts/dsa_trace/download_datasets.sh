#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DATA_DIR="${DATA_DIR:-"${ROOT_DIR}/benchmarks/datasets/dsa_trace"}"

mkdir -p "${DATA_DIR}"

echo "Downloading datasets into: ${DATA_DIR}"

# ShareGPT (used by vLLM benchmarks docs)
if [[ ! -f "${DATA_DIR}/ShareGPT_V3_unfiltered_cleaned_split.json" ]]; then
  echo "Downloading ShareGPT..."
  curl -L \
    -o "${DATA_DIR}/ShareGPT_V3_unfiltered_cleaned_split.json" \
    "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
else
  echo "ShareGPT already exists."
fi

# BurstGPT (arrival trace)
if [[ ! -f "${DATA_DIR}/BurstGPT_without_fails_2.csv" ]]; then
  echo "Downloading BurstGPT..."
  curl -L \
    -o "${DATA_DIR}/BurstGPT_without_fails_2.csv" \
    "https://github.com/HPMLL/BurstGPT/releases/download/v1.1/BurstGPT_without_fails_2.csv"
else
  echo "BurstGPT already exists."
fi

echo "Done."

