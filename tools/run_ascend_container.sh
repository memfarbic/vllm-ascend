#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-vllm_ascend_x}"
IMAGE="${IMAGE:-quay.io/ascend/vllm-ascend:YOUR_TAG}"
SRC_MOUNT_WORKSPACE_HOST="${SRC_MOUNT_WORKSPACE_HOST:-${HOME}/work/vllm-ascend}"
DST_MOUNT_WORKSPACE_CONTAINER="${DST_MOUNT_WORKSPACE_CONTAINER:-/workspace}"

# Optional proxies (avoid hardcoding secrets)
HTTP_PROXY="${HTTP_PROXY:-}"
HTTPS_PROXY="${HTTPS_PROXY:-}"
NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,0.0.0.0}"

# Enumerate davinci devices if present.
DAVINCI_DEVS=()
for d in /dev/davinci*; do
  [[ -e "$d" ]] && DAVINCI_DEVS+=(--device="$d")
done

# Basic docker opts
DOCKER_OPTS=(
  docker run
  --privileged
  --name "$CONTAINER_NAME"
  --net=host
  --ipc=host
  -itd
)

# Device opts (always required)
DEVICE_OPTS=(
  --device=/dev/davinci_manager
  --device=/dev/devmm_svm
  --device=/dev/hisi_hdc
)

# Mount opts (Ascend driver/tools)
MOUNT_OPTS=(
  -v /usr/local/dcmi:/usr/local/dcmi
  -v /etc/hccn.conf:/etc/hccn.conf
  -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi
  -v /etc/ascend_install.info:/etc/ascend_install.info
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver
  -v /sys/fs/cgroup:/sys/fs/cgroup:ro
  -v /var/log/npu/:/usr/slog
  -v /etc/hosts:/etc/hosts
  -v /dev/shm:/dev/shm
)

# Workspace/data mounts
WORKSPACE_OPTS=(
  -v "${HOME}/.cache:/root/.cache"
  -v /etc/localtime:/etc/localtime
  -v "${SRC_MOUNT_WORKSPACE_HOST}:${DST_MOUNT_WORKSPACE_CONTAINER}"
  -v /mnt:/mnt
  -v /data:/data
)

# Environment variables
ENV_OPTS=(
  -e VLLM_USE_MODELSCOPE=True
  -e no_proxy="${NO_PROXY}"
)

if [[ -n "${HTTP_PROXY}" ]]; then
  ENV_OPTS+=( -e http_proxy="${HTTP_PROXY}" )
fi
if [[ -n "${HTTPS_PROXY}" ]]; then
  ENV_OPTS+=( -e https_proxy="${HTTPS_PROXY}" )
fi

# Final
"${DOCKER_OPTS[@]}" \
  "${DAVINCI_DEVS[@]}" \
  "${DEVICE_OPTS[@]}" \
  "${MOUNT_OPTS[@]}" \
  "${WORKSPACE_OPTS[@]}" \
  "${ENV_OPTS[@]}" \
  "${IMAGE}" bash
