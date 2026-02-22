# vLLM Ascend 容器开发与测试指南

本文档面向使用 `quay.io/ascend/vllm-ascend` 镜像的开发者和测试人员。

---

## 0. 约定：环境变量与默认值

本文档所有示例共享以下环境变量。**请在操作前按需设置**，后续所有命令直接引用它们：

```bash
# 镜像（默认 v0.13.0rc1-a3）
export VLLM_IMAGE="${VLLM_IMAGE:-quay.io/ascend/vllm-ascend:v0.13.0rc1-a3}"

# 分支（默认 v0.13.0rc1-dev，可持续 git pull 更新）
export VLLM_BRANCH="${VLLM_BRANCH:-v0.13.0rc1-dev}"

# 仓库 URL（内网请替换为 mirror）
export VLLM_REPO="${VLLM_REPO:-https://github.com/memfarbic/vllm-ascend.git}"

# 容器名
export CONTAINER_NAME="${CONTAINER_NAME:-vllm_ascend_x}"
```

> 如果你要切换版本，只需修改 `VLLM_BRANCH` 和 `VLLM_IMAGE`，其余命令不用改。

---

## 1. 什么是 editable install

镜像构建时执行了：

```bash
pip install -e /vllm-workspace/vllm
pip install -e /vllm-workspace/vllm-ascend
```

效果：Python import 时**直接读取源码目录**，而非 site-packages 里的副本。修改源码后无需重新安装（除非改了编译产物/扩展模块）。

验证方式：

```bash
python3 -c "import vllm_ascend, os; print(os.path.realpath(vllm_ascend.__file__))"
pip show vllm-ascend | grep -i editable
```

---

## 2. Quick Start：clone 源码 + 启动容器（推荐流程）

适用于：**跑服务 / 跑 benchmark / 跑 trace**，不在容器内提交代码。

### 2.1 宿主机准备源码

```bash
mkdir -p ~/work && cd ~/work

# 直接 clone 到可持续更新的分支（后续可 git pull）
git clone -b "$VLLM_BRANCH" --single-branch "$VLLM_REPO" vllm-ascend-"$VLLM_BRANCH"
cd vllm-ascend-"$VLLM_BRANCH"

# 记录版本信息（建议粘到实验日志）
git log -1 --oneline
git describe --tags --always
```

> **多版本并行**：建议每个版本一个目录（如 `vllm-ascend-v0.13.0rc1-dev/`、`vllm-ascend-main/`），可同时挂载到不同容器，互不覆盖。

### 2.2 启动容器

#### 方式 A：使用仓库提供的脚本（推荐）

脚本位于 `tools/run_ascend_container.sh`，特点：
- 自动枚举 `/dev/davinci*` 设备
- 默认挂载 Ascend 驱动/工具和常用 cache 目录
- 代理通过环境变量传入，不硬编码密码

```bash
cd ~/work/vllm-ascend-"$VLLM_BRANCH"

export IMAGE="$VLLM_IMAGE"
export SRC_MOUNT_WORKSPACE_HOST="$(pwd)"

# 可选：代理
# export HTTP_PROXY=http://user:pass@proxy:8080/
# export HTTPS_PROXY=http://user:pass@proxy:8080/

bash tools/run_ascend_container.sh
```

启动后进入容器：

```bash
docker exec -it "$CONTAINER_NAME" bash
```

#### 方式 B：单条 `docker run`（临时复制粘贴）

```bash
docker run --privileged --name "$CONTAINER_NAME" --net=host --ipc=host -itd \
  --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /etc/hccn.conf:/etc/hccn.conf \
  -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /sys/fs/cgroup:/sys/fs/cgroup:ro \
  -v /var/log/npu/:/usr/slog \
  -v /etc/hosts:/etc/hosts \
  -v /dev/shm:/dev/shm \
  -v "$HOME"/.cache:/root/.cache \
  -v /etc/localtime:/etc/localtime \
  -v "$(pwd)":/vllm-workspace/vllm-ascend \
  -v /mnt:/mnt -v /data:/data \
  -e VLLM_USE_MODELSCOPE=True \
  ${HTTP_PROXY:+-e http_proxy="$HTTP_PROXY"} \
  ${HTTPS_PROXY:+-e https_proxy="$HTTPS_PROXY"} \
  -e no_proxy="${NO_PROXY:-localhost,127.0.0.1,0.0.0.0}" \
  "$VLLM_IMAGE" bash
```

> 如果机器上有 `davinci0..davinci15`，需额外加 `--device=/dev/davinci0 ...`；方式 A 的脚本会自动处理。

### 2.3 容器内验证 editable install

首次进入容器建议执行一次，确保 import 指向挂载目录：

```bash
pip install -e /vllm-workspace/vllm-ascend

# 验证
python3 -c "import vllm_ascend, os; print(os.path.realpath(vllm_ascend.__file__))"
pip show vllm-ascend | grep -i editable
```

期望输出路径落在 `/vllm-workspace/vllm-ascend/...`。

### 2.4 更新代码与切换版本

| 场景 | 操作（在宿主机） |
|------|------------------|
| 拉取最新提交 | `git pull --ff-only` |
| 严格复现某个 tag | `git fetch --tags && git checkout tags/v0.13.0rc1` |
| 从 tag 切回可更新分支 | `git switch v0.13.0rc1-dev && git pull --ff-only` |

> **注意**：`git checkout tags/<tag>` 会进入 **detached HEAD**，此时不能 `git pull`。要恢复持续更新，请切回分支。避免使用模糊写法 `git checkout v0.13.0rc1`，请始终用 `tags/v0.13.0rc1`。

---

## 3. 开发工作流：修改代码并同步到 GitHub

### 3.1 推荐：宿主机开发 + bind mount（长期）

- 代码保存在宿主机（容器删了不丢）
- 用宿主机的 Git/SSH/PAT 凭据（更安全）
- 容器只提供 NPU/CANN/依赖环境

```bash
# 宿主机
cd ~/work/vllm-ascend-v0.13.0rc1-dev

# 正常开发、提交、推送
git add -A && git commit -m "your change" && git push
```

容器里的 editable install 会自动指向最新源码，无需重启服务（除非改了扩展模块）。

### 3.2 容器内直接推送（临时）

适用于临时验证、不长期使用的场景。

```bash
cd /vllm-workspace/vllm-ascend

git switch -c <your-branch>
git remote rename origin upstream
git remote add origin https://github.com/<yourname>/vllm-ascend.git

git add -A && git commit -m "your change"
git push -u origin <your-branch>
```

> 推送需要 GitHub 凭据（HTTPS PAT 或 SSH key）。容器销毁后改动可能丢失，建议优先用 3.1。

---

## 4. detached HEAD 与 tag/分支的关系

| 概念 | 说明 |
|------|------|
| **tag**（如 `v0.13.0rc1`） | 固定指针，永远指向同一个提交。适合严格复现。 |
| **分支**（如 `v0.13.0rc1-dev`） | 会随 push 前进。适合持续更新。 |
| **detached HEAD** | 切到 tag/commit 后的状态。不能 `git pull`，不能直接 push。 |

**本文档默认使用分支 `v0.13.0rc1-dev`**，所以你 clone 后直接就在分支上，可以 `git pull`。只有你显式 `git checkout tags/...` 才会进入 detached HEAD。

---

## 5. 特殊场景

### 5.1 内网环境（只能 clone，不能 push 到 GitHub）

把 remote 指向内网镜像（用于 fetch），修改通过 patch/bundle 导出：

```bash
# 容器内
cd /vllm-workspace/vllm-ascend
git remote add mirror <INTRANET_URL>
git fetch mirror --tags
```

导出修改（三选一）：

```bash
# A) format-patch（最常用）
git format-patch -N -o /tmp/patches

# B) bundle（含完整提交链）
git bundle create /tmp/vllm-ascend.bundle <base>..HEAD

# C) 宿主机持久化 + 在有网络的环境 push（推荐）
```

在有 push 权限的机器上应用：

```bash
git am /path/to/patches/*.patch
# 或
git fetch /tmp/vllm-ascend.bundle <branch>
```

### 5.2 什么时候需要自己 build 镜像

| 不需要 build | 需要 build |
|--------------|------------|
| 只改 Python 代码 | 要交付固化的可复现镜像 |
| 用挂载方式接入源码 | 改了依赖/系统库/编译选项/算子 |
| 复用官方镜像环境 | CI/CD 构建发布 |

需要 build 时的思路：

```dockerfile
FROM quay.io/ascend/vllm-ascend:v0.13.0rc1-a3
RUN git clone -b v0.13.0rc1-dev --single-branch https://github.com/memfarbic/vllm-ascend.git /workspace/vllm-ascend \
    && pip install -e /workspace/vllm-ascend
```

---

## 6. 快速排查清单

```bash
# editable 是否生效
pip show vllm-ascend | grep -i editable
python3 -c "import vllm_ascend, os; print(os.path.realpath(vllm_ascend.__file__))"

# 当前是否在分支上（非 detached HEAD）
git branch --show-current

# remote 指向
git remote -v
```

| 问题 | 原因 | 解决 |
|------|------|------|
| `git pull` 报 `not on a branch` | 你在 detached HEAD（切了 tag） | `git switch v0.13.0rc1-dev` |
| import 路径不在挂载目录 | 未执行 `pip install -e` | 容器内 `pip install -e /vllm-workspace/vllm-ascend` |
| push 失败 | 无凭据或 remote 错误 | 检查 `git remote -v`，配置 PAT/SSH |
