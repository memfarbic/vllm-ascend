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
git clone <repo> /vllm-workspace/vllm-ascend
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

这意味着你当前处于 **detached HEAD**（常见于 checkout 了某个 tag/commit，例如 `tag: v0.11.0`），此时：

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
