# vLLM Ascend 镜像内 `pip install -e` 机制与二次开发/同步 GitHub 指南

本文档面向使用 `quay.io/ascend/vllm-ascend:*`（或类似）镜像的开发者，说明：

- 镜像内为什么会出现 `pip install -e`（editable install），以及它实际做了什么
- 如何判断当前容器中的源码与 Python import 的对应关系
- 在容器里看到 `Not currently on any branch`（detached HEAD）意味着什么
- 你要修改 `vllm` / `vllm-ascend` 源码并同步到 GitHub 的两种推荐工作流
- 什么时候需要自己 build 镜像，什么时候不需要

> 说明：命令示例以容器内路径为主（例如 `/vllm-workspace/vllm`、`/vllm-workspace/vllm-ascend`）。你可按实际情况替换。

---

## 1. 现象：`pip show` 里出现 `Editable project location`

在容器内执行：

```bash
pip show vllm
pip show vllm-ascend
```

通常会看到类似字段：

- `Location`: Python 的 site-packages 路径（例如 `/usr/local/python3.11.13/lib/python3.11/site-packages`）
- `Editable project location`: editable 安装所指向的源码目录（例如 `/vllm-workspace/vllm`、`/vllm-workspace/vllm-ascend`）

这表示 **`vllm` 与/或 `vllm-ascend` 并不是以“拷贝源码进 site-packages”的方式安装**，而是 **Python import 时直接指向某个源码目录**。

---

## 2. 原理：`pip install -e`（editable install）到底做了什么

镜像构建阶段（Dockerfile 的 `RUN`）或容器启动脚本里，通常会做两步：

1) 准备源码到某个目录（clone 或 COPY）：

```bash
git clone <repo> /vllm-workspace/vllm
git clone https://github.com/memfarbic/vllm-ascend.git /vllm-workspace/vllm-ascend
```

2) 以 editable 方式安装：

```bash
pip install -e /vllm-workspace/vllm
pip install -e /vllm-workspace/vllm-ascend
```

editable 安装的核心效果是：在 `site-packages` 里放置一种“指向源码目录”的引用（例如 `.pth` 或 PEP 660 的 editable 机制），使得：

- 你修改源码目录（`/vllm-workspace/vllm`）里的代码
- Python 立刻 import 到修改后的版本（无需重新 `pip install`，除非你改了构建产物/扩展模块等）

什么时候需要重新安装？

- **只改 Python 源码**：通常不需要
- **改了构建产物/扩展模块**（例如需要重新编译的算子、C++ 扩展、wheel 生成逻辑、`pyproject.toml`/`setup.cfg` 中影响 build 的内容）：可能需要重新 build/安装，或重启服务进程

---

## 3. 如何验证：Python import 的代码到底来自哪里

在容器中运行：

```bash
python3 -c "import vllm,os; print(vllm.__file__); print(os.path.realpath(vllm.__file__))"
python3 -c "import vllm_ascend,os; print(vllm_ascend.__file__); print(os.path.realpath(vllm_ascend.__file__))"
```

如果输出路径位于：

- `/vllm-workspace/vllm/...`
- `/vllm-workspace/vllm-ascend/...`

就说明当前 Python import 的就是这些工作区目录里的源码（符合 `pip show` 的 `Editable project location`）。

---

## 4. 你当前容器内仓库状态解读（重点：detached HEAD）

在 `/vllm-workspace/vllm` 下看到：

```bash
git status
# Not currently on any branch.
```

这意味着你当前处于 **detached HEAD**（常见于 checkout 了某个 tag/commit，例如 `tag: v0.13.0rc1`），此时：

- 你依然可以修改代码并运行（editable 安装会生效）
- 但要 **规范地提交并推送到 GitHub**，建议先创建并切换到一个分支，否则后续管理很容易混乱

---

## 5. 目标：修改代码并同步到 GitHub（两种工作流）

下面分两种方式：**长期推荐方式（宿主机挂载）** 与 **直接在容器内推送（临时可用）**。

### 5.1 推荐（长期开发）：宿主机保存源码 + bind mount 到容器

适用场景：

- 你希望改动不丢失（容器删了也没关系）
- 你希望用宿主机的 Git/SSH/PAT 凭据（更顺手、更安全）
- 容器只负责提供 NPU/CANN/依赖环境

做法要点：

1) 在宿主机 fork 并 clone 你的仓库，例如：

- `vllm`: fork `vllm-project/vllm` -> `https://github.com/<yourname>/vllm.git`
- `vllm-ascend`: fork `vllm-project/vllm-ascend` -> `https://github.com/<yourname>/vllm-ascend.git`

2) 用挂载方式启动容器，把宿主机源码覆盖到容器工作区：

```bash
docker run --rm -it   -v /host/path/vllm:/vllm-workspace/vllm   -v /host/path/vllm-ascend:/vllm-workspace/vllm-ascend   <image> bash
```

3) 在容器内确保指向挂载目录的 editable 安装（必要时执行一次）：

```bash
pip install -e /vllm-workspace/vllm
pip install -e /vllm-workspace/vllm-ascend
```

4) 在宿主机正常开发并提交推送：

```bash
git add -A
git commit -m "Describe your change"
git push
```

优势总结：

- 容器环境随时可换；代码与 Git 历史永久保留在宿主机
- 更适合长期迭代与多人协作

---

### 5.2 可用（但不推荐长期）：直接在容器内改并推送到 GitHub

适用场景：

- 你只是临时改动验证，或容器本身就是你的“开发机”
- 你愿意在容器里配置 GitHub 凭据（HTTPS PAT 或 SSH key）
- 你接受容器层的改动可能随容器销毁而丢失（除非你做持久化或导出）

#### 5.2.1 `vllm`：从 tag/commit（detached HEAD）创建分支并推到你 fork

在容器中：

```bash
cd /vllm-workspace/vllm

# 1) 从当前 detached HEAD 创建分支
git switch -c <your-branch-name>

# 2) 建议保留 upstream，同时把 origin 指到你自己的 fork
git remote rename origin upstream
git remote add origin https://github.com/<yourname>/vllm.git
git remote -v

# 3) 修改代码后提交并推送
git add -A
git commit -m "Describe your change"
git push -u origin <your-branch-name>
```

#### 5.2.2 `vllm-ascend`：同样从当前状态创建分支并推到你 fork

```bash
cd /vllm-workspace/vllm-ascend

git switch -c <your-branch-name>
git remote rename origin upstream
git remote add origin https://github.com/<yourname>/vllm-ascend.git
git remote -v

git add -A
git commit -m "Describe your change"
git push -u origin <your-branch-name>
```

> 如果推送时提示认证失败，请改用 GitHub PAT（HTTPS）或配置 SSH key。建议在容器外（宿主机）完成凭据管理，再用挂载方式开发（见 5.1）。

---

## 5.3 内网环境（只能 clone）怎么做？

你问的“在内网只能 clone 的环境里，是否在 docker 里面直接把 `vllm-ascend` 的 remote url 设置成当前项目的就可以了？”需要拆成两件事来看：

### 5.3.1 结论（先说清楚）

- **如果你的内网 Git 只允许 clone/fetch（只读）**：
  - 把 remote URL 指到内网仓库（镜像仓库）**可以解决拉代码/同步 upstream 代码**的问题
  - 但它**不能解决 push**（提交同步回 GitHub/内网仓库）的问题——因为权限/网络策略不允许写
- **如果内网仓库对你是可写（允许 push）**：
  - 你可以在容器内把 `origin` 指向内网仓库并直接 `git push` 到内网
  - 后续再由“有出网权限/有 GitHub 凭据”的环境把内网的提交同步到 GitHub

所以“只改 remote URL”是否足够，取决于：**你能不能 push 到那个 remote**。

### 5.3.2 在容器里把 remote 指到内网仓库（只为 clone/fetch）

如果你的内网只能访问内网 Git（例如 `git.internal.local`），但不能访问 `github.com`，你可以保留两类 remote：

- `upstream`：指向内网的只读镜像（用于拉取）
- `origin`：指向你可写的目标（可能是内网 fork，也可能是离线导出后在外网机器推送）

示例（容器内）：

```bash
cd /vllm-workspace/vllm-ascend

# 1) 先看当前 remote
git remote -v

# 2) 把现有 origin 改名为 upstream（表示它是上游/镜像）
git remote rename origin upstream

# 3) 添加内网镜像 remote（只读也没问题，主要用于 fetch）
# 注意：URL 按你们内网实际地址替换
# git remote add mirror ssh://git@git.internal.local/vllm-ascend.git

git remote add mirror <INTRANET_READONLY_URL>

# 4) 拉取更新

git fetch mirror --tags
```

如果你想让默认的 `git pull` 使用内网镜像，可以把 `upstream`/`mirror` 作为你日常 fetch 的 remote。

### 5.3.3 只有 clone 权限时，怎么把修改“带出去/同步到 GitHub”？

当你无法 push（只读）时，推荐三种方式：

#### 方式 A：`git format-patch` 导出补丁（最常用）

在内网容器里把你做的提交导出成 patch 文件：

```bash
cd /vllm-workspace/vllm-ascend

# 假设你在分支上产生了若干提交
# 导出最近 N 个提交
mkdir -p /tmp/patches

git format-patch -N -o /tmp/patches
```

把 `/tmp/patches/*.patch` 拷贝到有 GitHub 推送权限的机器上，然后：

```bash
git am /path/to/patches/*.patch
```

#### 方式 B：`git bundle` 打包整个提交链（适合跨多分支/含 tag）

```bash
cd /vllm-workspace/vllm-ascend

# 打包当前分支的提交（示例：从某个 base 起到 HEAD）
git bundle create /tmp/vllm-ascend.bundle <base>..HEAD
```

在外网机器上：

```bash
git clone /tmp/vllm-ascend.bundle repo
# 或 git fetch /tmp/vllm-ascend.bundle <branch>
```

#### 方式 C：宿主机持久化源码 + 外部环境推送（长期推荐）

即使内网容器不能 push，你也可以把代码放在宿主机持久化目录，容器只负责运行环境（见 5.1）。
然后在“有推送权限”的环境（可能是另一台机或同一台机的另一个网络域）完成 `git push`。

### 5.3.4 什么时候“直接把 origin 指到内网仓库”是正确的？

当满足以下条件时，你可以直接在容器内把 `origin` 指到内网仓库并推送：

- 你对内网仓库有写权限（push）
- 内网仓库用于团队协作/代码审查（例如内网 GitLab/Gitea）

```bash
cd /vllm-workspace/vllm-ascend

git remote set-url origin <INTRANET_WRITE_URL>

git push -u origin <your-branch>
```

之后再在外网环境把内网仓库同步到 GitHub（这一步通常由 CI 或镜像同步任务完成）。




---

## 5.4 只做测试（不在容器内提交/推送）怎么做更好？

如果你明确 **不在 Docker 容器内修改代码并 push**，容器只用于“跑服务/跑 benchmark/跑 trace 采集”，最推荐把容器当成**纯运行环境**。

**重要（强烈建议）**：

- **代码放在容器外持久化**：把源码放在宿主机目录（或 NFS/共享盘），容器销毁也不会丢。
- **容器里不配置 Git 凭据**：不需要 SSH key / PAT，减少泄露风险。
- **用 bind mount 覆盖容器内工作区**：保证 `pip install -e` 指向你挂载进去的源码目录。
- **用 tag/commit 控制实验可复现**：在宿主机 `git checkout <tag/commit>`，容器内立刻生效。

### 5.4.1 推荐工作流：宿主机保存源码 + 挂载进容器

下面以“只测试 `vllm-ascend`”为主线给出**完整可复现步骤**。

#### Step 0：选择一个固定版本（tag/commit）

**重要**：只做测试也建议固定 `tag/commit`，这样你的实验结果可复现。

例如（仅示例）：

```bash
git -C ~/work/vllm-ascend fetch --tags
# git -C ~/work/vllm-ascend checkout v0.13.0rc1
```

**补充说明（重要：tag vs 分支）**：

- 你如果执行 `git checkout v0.13.0rc1`（切到 **tag**），会进入 **detached HEAD**。这时 `git pull` 会报 `You are not currently on a branch`，因为你不在任何分支上，Git 不知道要把远端哪个分支合并到哪里。
- **tag 是固定指针**：`v0.13.0rc1` 永远指向同一个提交，用于“可复现”；它不会随着我们后续提交而前进。
- 如果你想获取我们后续 push 到远端 `origin/v0.13.0rc1` 的新 commit，需要切到一个**跟踪分支**（推荐做法）：

```bash
cd ~/work/vllm-ascend

# 先确保拿到远端更新
git fetch origin

# 建一个本地分支跟踪远端分支（避免和 tag 同名造成歧义）
git switch -c v0.13.0rc1-dev --track origin/v0.13.0rc1

# 之后更新就用 pull（建议 fast-forward）
git pull --ff-only
```

如果你只想回到“严格固定的 tag 状态”做复现实验：

```bash
git switch --detach tags/v0.13.0rc1
```

#### Step 1：宿主机准备源码（持久化保存）

```bash
mkdir -p ~/work
cd ~/work

# 直接硬编码当前仓库 URL（如果你在内网，请把这个 URL 替换成内网 mirror）
git clone https://github.com/memfarbic/vllm-ascend.git

# 进入仓库并切到你要测试的版本
cd vllm-ascend
git fetch --tags
# 示例：切到某个 tag/commit（按需修改）
# git checkout v0.13.0rc1

# 记录版本信息（建议把输出粘到实验日志里）
git rev-parse HEAD
git describe --tags --always
```

#### Step 2：启动容器并挂载源码

```bash
docker run --rm -it   -v ~/work/vllm-ascend:/vllm-workspace/vllm-ascend   quay.io/ascend/vllm-ascend:<tag> bash
```

> 说明：如果你的镜像工作区路径不是 `/vllm-workspace/vllm-ascend`，请按实际路径调整挂载点。

#### Step 3：容器内执行/确认 editable 安装指向挂载目录

**重要**：首次进入容器建议执行一次 `pip install -e`，确保 import 指向挂载目录。

```bash
pip install -e /vllm-workspace/vllm-ascend

# 验证 import 来源路径
python3 -c "import vllm_ascend,os; print(os.path.realpath(vllm_ascend.__file__))"

# 验证 pip 看到的 editable 位置
pip show vllm-ascend | sed -n '1,160p'
```

你期望看到的效果：

- `python3 -c ...` 输出路径落在 `/vllm-workspace/vllm-ascend/...`
- `pip show vllm-ascend` 中包含 `Editable project location: /vllm-workspace/vllm-ascend`

#### Step 4：容器内只跑测试/跑服务（不提交/不推送）

此时你只需要在容器内执行你的测试流程，例如：

- 启动 `vllm serve ...`
- 运行 benchmark
- 运行我们提供的 trace 回放脚本

如果你要换版本：**退出容器 → 宿主机 `git checkout <tag/commit>` → 重新启动容器**。

### 5.4.2 备选工作流：容器内切换 tag（只读，适合临时对比）

如果你完全不改代码、只做临时对比，也可以直接在容器内 `git checkout <tag>` 然后运行。

**重要（风险提示）**：这种方式容器删掉就丢状态；若要可复现与可追溯，仍建议使用 5.4.1（宿主机持久化）。


---

## 5.5 在昇腾单机上启动容器做测试（推荐脚本 + 单命令）

本节适用于：你在一台昇腾机器上用官方/内部镜像启动容器，只做**跑服务/跑 benchmark/跑 trace**，不在容器内提交/推送代码。

### 5.5.1 版本目录组织建议（强烈建议）

如果你要同时测试多个 tag/分支，建议在宿主机用“目录名=版本名”的方式管理源码，例如：

- `~/work/vllm-ascend-v0.13.0rc1/`
- `~/work/vllm-ascend-v0.13.0rc2/`
- `~/work/vllm-ascend-main/`

这样你可以并行挂载不同目录进不同容器，互不覆盖。

示例：

```bash
mkdir -p ~/work
cd ~/work

git clone https://github.com/memfarbic/vllm-ascend.git vllm-ascend-v0.13.0rc1
cd vllm-ascend-v0.13.0rc1

git fetch --tags
git checkout v0.13.0rc1

git rev-parse HEAD
git describe --tags --always
```

**重要：为什么你在 tag 上不能 `git pull`？**

上面示例的 `git checkout v0.13.0rc1` 是切到 **tag**，因此会进入 **detached HEAD**。这对于“固定版本跑实验”是对的，但它的特性是：

- **不能直接 `git pull`**（你不在分支上）
- **tag 不会前进**（不会自动包含我们后续的提交）

如果你希望把我们后续新增的 commit（例如文档/脚本更新）拉下来，请改用跟踪远端分支：

```bash
cd ~/work/vllm-ascend-v0.13.0rc1

git fetch origin

git switch -c v0.13.0rc1-dev --track origin/v0.13.0rc1

git pull --ff-only
```

你依然可以在需要时切回 tag 做严格复现：

```bash
git switch --detach tags/v0.13.0rc1
```

> 内网环境请把 `git clone` URL 替换为内网 mirror；目录命名方式不变。

### 5.5.2 推荐方式：使用脚本启动（可维护、可复用）

仓库提供了一个推荐脚本（路径见下方），特点：

- 不硬编码代理账号密码（用环境变量传入）
- 自动枚举 `/dev/davinci*` 设备节点（按机器实际情况）
- 默认挂载常用 Ascend 驱动/工具与 cache 目录

示例（宿主机）：

```bash
export IMAGE=quay.io/ascend/vllm-ascend:YOUR_TAG
export CONTAINER_NAME=vllm_ascend_x

cd $HOME/work/vllm-ascend-v0.13.0rc1
export SRC_MOUNT_WORKSPACE_HOST="$(pwd)"
export DST_MOUNT_WORKSPACE_CONTAINER=/vllm-workspace/vllm-ascend

# 可选：代理（建议不要把账号密码写死到脚本里）
# export HTTP_PROXY=http://user:pass@proxy:8080/
# export HTTPS_PROXY=http://user:pass@proxy:8080/
# export NO_PROXY=localhost,127.0.0.1,0.0.0.0

bash tools/run_ascend_container.sh
```

容器启动后你可以进入：

```bash
docker exec -it $CONTAINER_NAME bash
```

### 5.5.3 等价的单条 `docker run`（便于临时复制粘贴）

当你只想快速启动一次，也可以用单命令（参数按需调整）：

```bash
docker run --privileged --name vllm_ascend_x --net=host --ipc=host -itd \
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
  -v $HOME/.cache:/root/.cache \
  -v /etc/localtime:/etc/localtime \
  -v $HOME/work/vllm-ascend-v0.13.0rc1:/vllm-workspace/vllm-ascend \
  -v /mnt:/mnt -v /data:/data \
  -e VLLM_USE_MODELSCOPE=True \
  ${HTTP_PROXY:+-e http_proxy=$HTTP_PROXY} \
  ${HTTPS_PROXY:+-e https_proxy=$HTTPS_PROXY} \
  -e no_proxy="${NO_PROXY:-localhost,127.0.0.1,0.0.0.0}" \
  quay.io/ascend/vllm-ascend:YOUR_TAG bash
```

> 提示：如果你机器上有 `davinci0..davinci15`，你可以额外加 `--device=/dev/davinci0 ...`；推荐脚本会自动枚举这些节点。

---

## 6. 什么时候需要自己 build 镜像？什么时候不需要？

### 6.1 不需要 build 的典型情况

- 你只是想开发/调试 `vllm` 或 `vllm-ascend` 的 Python 代码
- 你愿意用挂载方式把源码接入容器（推荐）
- 你希望快速复用官方镜像已配置好的 NPU/CANN/依赖环境

此时直接用现成镜像 + `pip install -e` 就够了。

### 6.2 需要自己 build（或在现有镜像上叠一层）的典型情况

- 你要交付一个“别人拉了镜像就能跑”的版本，需要把你的 commit 固化到镜像层
- 你修改了依赖版本、系统库、编译选项、扩展模块/算子等，需要在镜像里固定
- 你要在 CI/CD 里构建并发布镜像，确保可复现

思路通常是写一个新的 Dockerfile：

- `FROM quay.io/ascend/vllm-ascend:<tag>`
- `COPY`/`git clone` 你 fork 的指定 commit
- `pip install -e ...` 或构建 wheel 并安装

（具体怎么 build 与发布取决于你们的交付流程，这里不展开。）

---

## 7. 快速检查清单（遇到问题先看这里）

- **editable 是否生效**：

```bash
pip show vllm | sed -n '1,120p'
python3 -c "import vllm,os; print(os.path.realpath(vllm.__file__))"
```

- **是否在 detached HEAD**：

```bash
git -C /vllm-workspace/vllm status
```

- **是否已经有分支**：

```bash
git -C /vllm-workspace/vllm branch --show-current
```

- **remote 指向是否正确**（是否是你 fork）：

```bash
git -C /vllm-workspace/vllm remote -v
git -C /vllm-workspace/vllm-ascend remote -v
```

---

## 8. 推荐的实践总结

- **优先选 5.1（宿主机保存源码 + 挂载进容器）**：最稳、最不容易丢改动，GitHub 凭据也好管理。
- **容器内看到 `Not currently on any branch`** 时，先 `git switch -c ...` 再开始做需要提交/推送的开发。
- **是否需要 build 镜像**取决于你要不要“交付一个固化可复现环境”，开发调试通常不需要。
