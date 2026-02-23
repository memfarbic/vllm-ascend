# DSA 访问轨迹插桩与采集 —— 完整指南

## 目录

1. [背景与目标](#1-背景与目标)
2. [整体架构](#2-整体架构)
3. [环境变量参考](#3-环境变量参考)
4. [插桩点详解](#4-插桩点详解)
   - [4.0 step_idx 与 request_id 绑定（model_runner_v1）](#40-step_idx-与-request_id-绑定model_runner_v1)
   - [4.1 插桩点 A：DSA top-2048 选择输出](#41-插桩点-adsa-top-2048-选择输出)
   - [4.2 插桩点 B：token→KV block 映射](#42-插桩点-btoken→kv-block-映射)
   - [4.3 插桩点 C：外部 KV 取数与搬运](#43-插桩点-c外部-kv-取数与搬运)
   - [4.4 插桩点 D：prefix cache 交互](#44-插桩点-dprefix-cache-交互)
5. [适配不同 vLLM 版本（tag）的指南](#5-适配不同-vllm-版本tag的指南)
6. [产出物：JSONL 字段定义](#6-产出物jsonl-字段定义)
   - [6.1 dsa_access.jsonl](#61-dsa_accessjsonl)
   - [6.2 kv_io.jsonl](#62-kv_iojsonl)
   - [6.3 prefix_cache.jsonl](#63-prefix_cachejsonl)
7. [数据集准备](#7-数据集准备)
   - [7.1 D1: RULER](#71-d1-ruler)
   - [7.2 D2: LongBench v2](#72-d2-longbench-v2)
   - [7.3 D3: BurstGPT](#73-d3-burstgpt)
   - [7.4 D4: ShareGPT](#74-d4-sharegpt)
8. [端到端采集流程](#8-端到端采集流程)
9. [开销控制与常见问题](#9-开销控制与常见问题)
10. [改动文件清单](#10-改动文件清单)

---

## 1. 背景与目标

DeepSeek V3.2-Exp 使用 **DSA（Dynamic Sparse Attention）**：每个新 query token 经过 indexer 选出历史序列中最相关的 **top-2048** 个 token 位置，只对这些 token 做 attend。这意味着：

- 每个 decode step 实际只触达 KV cache 的一个子集（而非全部）
- 这些被选中 token 分散在不同的 KV block 中——block 的访问模式直接影响外部 KV（MemFabric/MemCache/Mooncake）的取数成本
- prefix cache 中哪些块被 DSA 频繁命中，决定了"把哪些块放近端/做多副本"的策略

**本工具的唯一目标**：在推理路径上以最低侵入方式导出可复现的 **token→block 访问轨迹 + 外部 KV 取数代价 + prefix cache 复用关系**，为后续设计 KV 放置/迁移策略提供数据依据。

**不做**：不改任何 KV 放置/迁移/副本/重算策略。

---

## 2. 整体架构

```
┌──────────────────────────────────────────────────────────────┐
│ vllm serve (DeepSeek V3.2-Exp)                               │
│                                                              │
│  model_runner_v1.execute_model()                             │
│   ├─ 设置 step_idx + batch_req_ids  ─────────────────┐      │
│   │                                                   │      │
│   │  ┌─ Transformer Layer (x N) ─────────────────┐   │      │
│   │  │  AscendSFAImpl.forward()                  │   │      │
│   │  │   ├─ indexer_select_post_process()         │   │      │
│   │  │   │   └─ npu_lightning_indexer → topk_indices │      │
│   │  │   │                                        │   │      │
│   │  │   ├─ ★ 插桩 A/B/D：记录 topk + block 映射  │  │      │
│   │  │   │   + prefix 交集                        │   │      │
│   │  │   │                                        │   │      │
│   │  │   └─ npu_sparse_flash_attention            │   │      │
│   │  └────────────────────────────────────────────┘   │      │
│   │                                                   │      │
│   │  ┌─ KV Transfer / Pool ──────────────────────┐   │      │
│   │  │  pool_worker.start_load_kv()              │   │      │
│   │  │   └─ m_store.get() → ★ 插桩 C             │   │      │
│   │  │  mooncake_connector._transfer_kv_cache()  │   │      │
│   │  │   └─ batch_transfer_sync_read → ★ 插桩 C  │   │      │
│   │  │  pool_scheduler.get_num_new_matched_tokens │   │      │
│   │  │   └─ ★ 插桩 C (lookup) + ★ 插桩 D         │   │      │
│   │  │  cpu_kv_cache_manager.get_matched_num...   │   │      │
│   │  │   └─ ★ 插桩 D                             │   │      │
│   │  └────────────────────────────────────────────┘   │      │
│   │                                                   │      │
│   └───────────────────────────────────────────────────┘      │
│                         │                                    │
│                         ▼                                    │
│               vllm_ascend/trace/                             │
│               ├─ dsa_tracer.py (DSATracer 单例)              │
│               └─ trace_writer.py (线程安全 JSONL 写入)       │
│                         │                                    │
│                         ▼                                    │
│         ${VLLM_ASCEND_TRACE_DIR}/<run_id>/                   │
│         ├─ dsa_access.jsonl   (A+B+D)                        │
│         ├─ kv_io.jsonl        (C)                            │
│         └─ prefix_cache.jsonl (D)                            │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. 环境变量参考

所有环境变量定义在 `vllm_ascend/envs.py` 中，**`VLLM_ASCEND_DSA_TRACE=0` 时全部插桩代码不执行、零开销**。

| 环境变量 | 类型 | 默认值 | 说明 |
|---------|------|--------|------|
| `VLLM_ASCEND_DSA_TRACE` | int | `0` | 总开关。`1` 启用插桩 |
| `VLLM_ASCEND_TRACE_DIR` | str | `./trace_out` | JSONL 输出根目录 |
| `VLLM_ASCEND_DSA_TRACE_DECODE_ONLY` | int | `1` | 是否只采 decode step（跳过 prefill） |
| `VLLM_ASCEND_DSA_TRACE_SAMPLE_RATE` | float | `1.0` | 采样率 `[0,1]`，`0.1` 表示只记录 10% 的 step |
| `VLLM_ASCEND_DSA_TRACE_MAX_STEPS` | int | `-1` | 最多记录多少步（负数不限） |
| `VLLM_ASCEND_DSA_TRACE_LAYER_FILTER` | str | `""` | 逗号分隔的 layer_name 白名单，空=全部 |
| `VLLM_ASCEND_DSA_TRACE_SEED` | int | `0` | 采样 RNG 种子（保证可复现） |

---

## 4. 插桩点详解

### 4.0 step_idx 与 request_id 绑定（model_runner_v1）

**改动文件**：`vllm_ascend/worker/model_runner_v1.py`

**原理**：attention 层（`AscendSFAImpl.forward`）执行时无法直接拿到"当前是第几步"和"batch 里有哪些 request"。我们在更上层的 `NPUModelRunner.execute_model()` 中：

1. 新增实例变量 `self._dsa_trace_step_idx: int = 0`（在 `__init__` 中）
2. 每次 `execute_model()` 被调时（且 tracer 已启用），调用：
   ```python
   tracer.set_step_context(self._dsa_trace_step_idx, list(req_ids))
   self._dsa_trace_step_idx += 1
   ```

这样 attention 层内就可以通过 `tracer.get_batch_req_ids()` 和 `tracer._step_ctx.step_idx` 关联到具体 request。

**为什么放在 `execute_model()` 而不是 attention forward**：因为 model 有 N 层 attention，每层都会被调用，但 step_idx 只应递增一次；且 `input_batch.req_ids` 只在 model_runner 层可见。

---

### 4.1 插桩点 A：DSA top-2048 选择输出

**改动文件**：`vllm_ascend/attention/sfa_v1.py`

**插桩位置**：`AscendSFAImpl.forward()` 中，在 `topk_indices = self.indexer_select_post_process(...)` 返回之后、`npu_sparse_flash_attention(...)` 调用之前。

**为什么在这里而不是 `indexer_select_post_process()` 内部**：在 `forward()` 中我们能同时拿到 `topk_indices`、`attn_metadata`（含 `block_table`、`seq_lens`、`cum_query_lens`）和 `forward_context`（含 profiling 状态），可以在一处完成 A+B+D 全部计算。

**代码逻辑**（伪代码）：

```python
# topk_indices: [num_tokens, num_heads, 2048]  (NPU tensor)
topk_np = topk_indices.detach().to("cpu").numpy()

for tok_i in range(num_tokens):
    req_idx = searchsorted(cum_query_lens, tok_i)  # 确定属于哪个 request
    request_id = batch_req_ids[req_idx]
    seq_len_current = seq_lens[req_idx]

    sel_pos_by_head = topk_np[tok_i]          # [num_heads, 2048]
    union_pos = np.unique(sel_pos_by_head)     # 所有 head 的 union
    unique_pos_count = len(union_pos)

    # offset 统计
    offsets = query_pos - union_pos
    offset_min, offset_median, offset_max = ...

    tracer.record_topk_and_blocks(
        selected_token_pos_by_head=sel_pos_by_head,  # per-head 保留
        unique_token_pos_count=unique_pos_count,
        offset_min=offset_min, ...
    )
```

**记录的字段**：`request_id, step_idx, layer_name, attn_state, seq_len_current, query_token_pos, selected_token_pos_by_head[num_heads][2048], unique_token_pos_count, offset_min/median/max`

**关于 scores**：当前 `npu_lightning_indexer` 算子内部计算了 Q*K 分数并排序，但**只返回 indices、不返回 scores**。若后续需要记录 scores，需修改 C++ 算子（`csrc/lightning_indexer_vllm/`）让它额外输出 topk scores 张量。

---

### 4.2 插桩点 B：token→KV block 映射

**改动文件**：`vllm_ascend/attention/sfa_v1.py`（与 A 在同一处）

**原理**：

```python
# 来自 A 点的 union_pos（去重后的 token 绝对位置）
logical_block_idx = union_pos // block_size
row = block_table_cpu[req_idx]                      # block_table 的第 req_idx 行
logical_block_idx = np.clip(logical_block_idx, 0, row.shape[0] - 1)
touched_blocks = row[logical_block_idx]              # 物理 block id
uniq_blocks, counts = np.unique(touched_blocks, return_counts=True)
```

**记录的字段**：`block_size, selected_block_ids（去重物理 block 列表）, unique_blocks, tokens_per_touched_block.mean/p50/p95`

**为什么不在 `block_table.py` 的 `compute_slot_mapping()` 里做**：因为 `compute_slot_mapping()` 映射的是"当前 step 要写入 KV cache 的 slot"，而我们需要的是"DSA 选出的历史 token 触达了哪些 block"——这是不同的集合。DSA 只出现在 `AscendSFAImpl`（SFA/sparse attention），非 DSA 的普通 attention 不走这个路径。

---

### 4.3 插桩点 C：外部 KV 取数与搬运

**改动文件**（三处）：

#### C-1：`pool_worker.py:start_load_kv()`

- **路径**：`vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py`
- **操作**：围绕 `self.m_store.get(key_list_c, addr_list_c, size_list_c)` 调用加了 `time.perf_counter()` 计时
- **记录**：`tier="remote_pool", op="batch_get", bytes_read=sum(size_list), read_ops=len(key_list), batch_size, latency_us, backend 类名`
- **安全性**：`tracer.enabled` 为 False 时走原路径（无任何计时开销）

#### C-2：`mooncake_connector.py:_transfer_kv_cache()`

- **路径**：`vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py`
- **操作**：在已有的 `req_start_time / req_end_time` 计时之后追加 tracer 调用
- **记录**：`tier="remote_pool", op="batch_transfer_sync_read", bytes_read=sum(length_list), read_ops=len(length_list), latency_us, session_id, num_blocks, num_groups`

#### C-3：`pool_scheduler.py:get_num_new_matched_tokens()`

- **路径**：`vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py`
- **操作**：在 lookup 结果确定后追加 tracer 调用
- **记录**：`tier="kvpool", op="lookup"` + 分层命中估算（HBM_hit_blocks / remote_pool_hit_blocks / kvpool_total_hit_blocks）

---

### 4.4 插桩点 D：prefix cache 交互

**改动文件**（三处）：

#### D-1：`cpu_kv_cache_manager.py:get_matched_num_and_touch()`

- **路径**：`vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload/cpu_kv_cache_manager.py`
- **记录**：`source="cpu_prefix", prefix_hit_tokens, prefix_cached_blocks, computed_blocks 数量`
- **同步维护**：tracer 内部 `_prefix_hit_tokens[request_id] = num_computed_tokens`，供 A/B 点查询

#### D-2：`pool_scheduler.py:get_num_new_matched_tokens()`

- **路径**：（同 C-3 文件）
- **记录**：`source="kvpool", prefix_hit_tokens=num_external_hit_tokens`

#### D-3：`sfa_v1.py`（在 A/B 同一处）

- 对每条 DSA 记录额外计算并附加：
  - `prefix_cached_blocks`：tracer 中缓存的该 request 的 prefix 命中块数
  - `dsa_prefix_intersection_ratio`：`(DSA touched logical blocks ∩ prefix blocks) / (DSA touched logical blocks)`
  - `dsa_prefix_hot_blocks`：交集中被选 token 最多的 Top-16 逻辑块，格式 `[{"logical_block": 3, "selected_tokens": 120}, ...]`

---

## 5. 适配不同 vLLM 版本（tag）的指南

本仓库通过 `vllm_ascend.utils.vllm_version_is()` 区分不同 vLLM tag（目前主要是 `v0.15.0` vs 非 `v0.15.0`）。DSA trace 插桩的兼容性分析如下：

### 5.1 本 trace 默认仅涉及 `AscendSFAImpl`

DSA（top-2048 选择 + sparse flash attention）只在 `AscendSFAImpl` 中实现（文件 `vllm_ascend/attention/sfa_v1.py`）。**其他 attention backend（`AscendAttentionBackendImpl`、`AscendMLAImpl`）不走 DSA 路径，因此不会触发 A/B 插桩。**

| Attention Backend | 类 | DSA? | 会产生 dsa_access? |
|---|---|---|---|
| `AscendSFABackend` | `AscendSFAImpl` | **是** | **是** |
| `AscendAttentionBackend` | `AscendAttentionBackendImpl` | 否 | 否 |
| `AscendMLABackend` | `AscendMLAImpl` | 否 | 否 |
| CP 变体 | `AscendAttentionCPImpl` / `AscendMlaCPImpl` | 否 | 否 |

### 5.2 version-specific 差异矩阵

| 插桩点 | 涉及文件 | vLLM v0.13.x (当前 main) | vLLM v0.15.0 | 适配说明 |
|--------|---------|--------------------------|--------------|---------|
| **step_idx 绑定** | `model_runner_v1.py` | `NPUModelRunner.execute_model()` | 同 | 两个版本的 `execute_model()` 入口结构一致 |
| **A/B/D（DSA）** | `sfa_v1.py` | `AscendSFAImpl.forward()` | 同 | `sfa_v1.py` 无 `vllm_version_is` 分支，DSA indexer 调用路径相同 |
| **C-1（pool_worker）** | `pool_worker.py` | `start_load_kv()` | 同 | 无版本分支 |
| **C-2（mooncake）** | `mooncake_connector.py` | `_transfer_kv_cache()` | 同 | 无版本分支 |
| **C-3 / D-2（pool_scheduler）** | `pool_scheduler.py` | `get_num_new_matched_tokens()` | 同 | 无版本分支 |
| **D-1（cpu prefix）** | `cpu_kv_cache_manager.py` | `get_matched_num_and_touch()` | 同 | 无版本分支 |
| **MLA attention** | `mla_v1.py`, `ops/mla.py` | `mla.py` 有 `vllm_version_is("v0.15.0")` 分支（影响 import path 和 weight 处理） | 见左列 | **但 MLA 不走 DSA**，因此不影响 trace |

### 5.3 如何在自己的 vLLM tag 上验证

1. 确认你的模型使用 `AscendSFABackend`（DeepSeek V3.2-Exp 的 DSA 默认走 SFA）
2. 设置 `VLLM_ASCEND_DSA_TRACE=1` 并跑一次 decode
3. 检查 `trace_out/<run_id>/dsa_access.jsonl` 是否有输出

如果你要在**非 SFA** 的 attention backend（普通 paged attention / MLA）上采集类似轨迹，需要在对应的 `AscendAttentionBackendImpl.forward()` 或 `AscendMLAImpl.forward()` 中自行添加相似逻辑——但那些 backend 没有 DSA top-k 选择环节，只能采集"全量 KV block 访问"的分布。

### 5.4 如果你要移植到全新 vLLM 版本

需要关注以下三个可能变化的 API：

| API | 当前使用方式 | 如果上游改了怎么办 |
|-----|------------|------------------|
| `attn_metadata.cum_query_lens` | 累计 query 长度，用于 `searchsorted` 确定 token→request 映射 | 检查新版本中对应字段名（可能改为 `query_start_loc`） |
| `attn_metadata.block_table` | GPU tensor，shape `[num_reqs, max_blocks]` | 检查新版本是否拆分为 multi-group（`MultiGroupBlockTable`） |
| `get_forward_context().in_profile_run` | 跳过 profiling 阶段 | 新版本可能改为不同的属性名 |


### 5.5 查看 tag / 获取可更新分支

查看所有 tag：

```bash
git fetch --tags
git tag -l "v0.13*"
```

**推荐（可持续 `git pull` 更新）**：直接 clone `v0.13.0rc1-dev` 分支，插桩改动已包含其中：

```bash
git clone -b v0.13.0rc1-dev --single-branch \
  https://github.com/memfarbic/vllm-ascend.git vllm-ascend-v0.13.0rc1
cd vllm-ascend-v0.13.0rc1

# 后续更新
git pull --ff-only
```

**严格复现（只读快照）**：切到 tag，会进入 detached HEAD，**不能 `git pull`**：

```bash
git fetch --tags
git checkout tags/v0.13.0rc1

# 要切回可更新分支：
# git switch v0.13.0rc1-dev
```

> 完整 git 工作流（多版本管理、内网环境、容器挂载等）请参阅
> `docs/source/developer_guide/contribution/editable_install_and_github_sync.md`。

### 5.6 向其他 vLLM tag 移植插桩的指南

#### 5.6.1 基本步骤

```bash
# 1) 从目标 tag 创建工作分支（命名加 -dev 避免与 tag 重名）
git fetch --tags
git checkout -b v<X.Y.Z>-dev refs/tags/v<X.Y.Z>

# 2) cherry-pick 插桩相关提交（在新 tag 上可能产生冲突）
git cherry-pick <instrumentation-commit>

# 3) 若有冲突：按"等价插入点"策略解决后继续
git add <resolved_files...>
git cherry-pick --continue

# 4) 推送
git push -u origin HEAD
```

> **`v0.13.0rc1` 的移植已完成**，直接使用远端分支 `v0.13.0rc1-dev` 即可，不需要重新执行上述步骤。

#### 5.6.2 冲突类型与处理建议

不同 vLLM tag 的目录结构和实现细节可能会变化，常见冲突：

| 差异类型 | 示例 |
|---------|------|
| attention metadata 字段 | `attn_state`、`block_table` vs `block_tables`、`cum_query_lens` vs `query_start_loc` |
| 外部 KV 目录结构 | `distributed/kvpool/*` vs `distributed/kv_transfer/*` |
| model runner 流程 | `execute_model()` 变量名差异 |

处理原则：

- 先保证 tag 版本原有逻辑语义正确（能跑），再按等价插入点把插桩 block 加回去：
  - **A/B/D**：`topk_indices` 产生后、`npu_sparse_flash_attention` 前
  - **step_idx 绑定**：每次 `execute_model()` 真实执行步只递增一次
  - **C**：`batch_get` / `transfer` / `lookup` 调用周围
- 遇到疑问先用 `git show refs/tags/<tag>:<path>` 对照原始实现
---

## 6. 产出物：JSONL 字段定义

所有输出落在 `${VLLM_ASCEND_TRACE_DIR}/<run_id>/`，`run_id` 格式为 `dsa_YYYYMMDD_HHMMSS_<pid>`。

### 6.1 `dsa_access.jsonl`

每行 = 一个 decode token（通常 = 一个 request 的一次 decode step）在某一层 attention 的 DSA 访问记录。

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | str | 固定 `"dsa_topk"` |
| `run_id` | str | 本次运行 id |
| `ts_us` | int | 记录时刻（微秒 Unix 时间戳） |
| `layer_name` | str | Transformer 层名（如 `model.layers.0.self_attn`） |
| `attn_state` | str | 当前 attention 状态（`DecodeOnly` / `SpecDecoding` / `ChunkedPrefill` 等） |
| `step_idx` | int | execute_model 调用次数（全局递增） |
| `request_id` | str | 请求 id |
| `req_idx` | int | 本 batch 内的请求索引 |
| `seq_len_current` | int | 当前序列长度 |
| `batch_req_ids` | list[str] | 本 batch 所有请求 id |
| `query_token_pos` | int 或 null | query token 在序列中的位置（通常 = seq_len - 1） |
| `block_size` | int | KV cache 块大小（如 128） |
| **A 点字段** | | |
| `selected_token_pos_by_head` | list[list[int]] | `[num_heads][2048]`，每个 head 选出的历史 token 绝对位置 |
| `unique_token_pos_count` | int | 所有 head union 后的去重 token 数 |
| `offset_min` | int 或 null | `query_pos - selected_pos` 的最小值 |
| `offset_median` | int 或 null | 中位数 |
| `offset_max` | int 或 null | 最大值 |
| **B 点字段** | | |
| `selected_block_ids` | list[int] 或 null | 去重后的物理 block id 列表 |
| `unique_blocks` | int 或 null | 去重后 block 数量 |
| `tokens_per_touched_block` | dict 或 null | `{"mean":…, "p50":…, "p95":…}` 每个 block 被选中的 token 数分布 |
| **D 点字段** | | |
| `prefix_cache_hit` | bool | 该 request 是否有 prefix cache 命中 |
| `prefix_cached_blocks` | int 或 null | prefix 缓存块数 |
| `dsa_prefix_intersection_ratio` | float 或 null | DSA touched blocks 与 prefix blocks 的块级交集占 touched 的比例 |
| `dsa_prefix_hot_blocks` | list[dict] 或 null | 交集中最热的 Top-16 块：`[{"logical_block": N, "selected_tokens": M}, ...]` |

### 6.2 `kv_io.jsonl`

每行 = 一次 KV I/O 操作。

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | str | 固定 `"kv_io"` |
| `run_id` | str | 本次运行 id |
| `ts_us` | int | 时间戳 |
| `step_idx` | int | 对应 step（-1 表示 step 上下文不可用） |
| `tier` | str | 存储层级：`"kvpool"` / `"remote_pool"` |
| `op` | str | 操作类型：`"lookup"` / `"batch_get"` / `"batch_transfer_sync_read"` |
| `request_id` | str | 请求 id |
| `bytes_read` | int | 本次读取的总字节数 |
| `read_ops` | int | 读取操作次数（如 key 数量） |
| `batch_size` | int 或 null | 批量操作的 batch 大小 |
| `latency_us` | int 或 null | 本次操作耗时（微秒） |
| `extra` | dict 或 null | 附加信息，因路径而异 |

`extra` 字段可能包含：

| 来源 | extra 内容 |
|------|-----------|
| pool_worker batch_get | `{"backend": "MemcacheBackend"}` |
| mooncake transfer | `{"session_id": "…", "num_blocks": N, "num_groups": M}` |
| pool_scheduler lookup | `{"HBM_hit_blocks": N, "remote_pool_hit_blocks": M, "kvpool_total_hit_blocks": K, …}` |

### 6.3 `prefix_cache.jsonl`

每行 = 一次 prefix cache 命中查询结果。

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | str | 固定 `"prefix_cache"` |
| `run_id` | str | 运行 id |
| `ts_us` | int | 时间戳 |
| `step_idx` | int | 对应 step |
| `request_id` | str | 请求 id |
| `prefix_cache_hit` | bool | 是否有命中 |
| `prefix_hit_tokens` | int | 命中的 token 数 |
| `block_size` | int | 块大小 |
| `prefix_cached_blocks` | int | 缓存块数 |
| `source` | str | 来源：`"cpu_prefix"` 或 `"kvpool"` |
| `extra` | dict 或 null | 附加信息 |

---

## 7. 数据集准备

### 7.1 D1: RULER

**论文**：RULER: What's the Real Context Size of Your Long-Context Language Models? (arXiv:2404.06654)

**用途**：合成长上下文基准，可控长度与复杂度。最大化/放大 DSA 的"非局部选择"，测 `unique_blocks`、`tokens_per_touched_block`、offset 分布。

**数据源**：HuggingFace `allenai/ruler_data`

> 注意：`allenai/ruler_data` 在 HuggingFace 上以 `*.tgz` 归档形式发布（例如 `data_100_samples.tgz`、`data_debug.tgz`），
> 不能用 `datasets.load_dataset()` 直接加载。本仓库的 `prepare_ruler.py` 会自动下载并解压归档，然后读取
> `data/ruler/<task>/validation_4096.jsonl`。

```bash
# 建议先用小包验证链路（更快）
python3 benchmarks/scripts/dsa_trace/prepare_ruler.py \
  --output benchmarks/datasets/dsa_trace/ruler.prompts.jsonl \
  --archive data_debug.tgz \
  --num-prompts 200 \
  --max-tokens 256 \
  --seed 0

# 需要更多样本时再用大包（下载/解压更慢）
# python3 benchmarks/scripts/dsa_trace/prepare_ruler.py \
#   --output benchmarks/datasets/dsa_trace/ruler.prompts.jsonl \
#   --archive data_100_samples.tgz \
#   --num-prompts 200 \
#   --max-tokens 256 \
#   --seed 0
```

可选：

- `--task-allowlist qa_1,qa_2,vt,niah_single_1`：只保留特定任务类型（task 名来自归档目录名）。
- `--cache-dir /path/to/cache`：指定 tgz 下载缓存目录（默认 `~/.cache/vllm-ascend/datasets`）。


### 7.2 D2: LongBench v2

**论文**：LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks (arXiv:2412.15204)

**用途**：真实长任务、多场景（文档 QA、多文档、代码仓库理解等），证明 RULER 上的观测不是合成数据特例。

**数据源**：HuggingFace `zai-org/LongBench-v2`（别名 `THUDM/LongBench-v2`）

```bash
python3 benchmarks/scripts/dsa_trace/prepare_longbenchv2.py \
  --output benchmarks/datasets/dsa_trace/longbenchv2.prompts.jsonl \
  --num-prompts 200 \
  --max-tokens 128 \
  --seed 0
```

### 7.3 D3: BurstGPT

**论文**：BurstGPT: A Real-world Workload Dataset to Optimize LLM Serving Systems (arXiv:2401.17644)

**用途**：真实到达过程/突发并发。记录跨层命中、远端 bytes/ops、p99 延迟随突发变化。

**数据源**：GitHub Release `HPMLL/BurstGPT` v1.1

```bash
# 下载
bash benchmarks/scripts/dsa_trace/download_datasets.sh

# 准备
python3 benchmarks/scripts/dsa_trace/prepare_burstgpt.py \
  --input benchmarks/datasets/dsa_trace/BurstGPT_without_fails_2.csv \
  --output benchmarks/datasets/dsa_trace/burstgpt.workload.jsonl \
  --num-requests 200
```

> **注意**：BurstGPT 只提供到达时间和 token 长度，**不包含 prompt 文本**。回放脚本使用简单的合成 prompt 来逼近目标输入长度（只用于触发 DSA 与并发行为，不做语义正确性评测）。

### 7.4 D4: ShareGPT

**论文**：可追溯 OpenChat 论文采用 ShareGPT 作为常用 SFT/对话数据集。

**用途**：典型对话请求下的 DSA 访问图 + prefix cache 交互（多轮/模板化带来共享前缀）。

**数据源**：HuggingFace `anon8231489123/ShareGPT_Vicuna_unfiltered`

```

> 如果你在内网/代理环境下载 BurstGPT 遇到 TLS 证书链问题（curl: (60)），本脚本默认使用 `-k`（不校验证书）。
>
> - **推荐开启校验**：`export VLLM_ASCEND_CURL_INSECURE=0`，并（可选但推荐）提供内网 CA：`export VLLM_ASCEND_CURL_CA_BUNDLE=/path/to/ca-bundle.pem`
> - **保持默认不校验**：`export VLLM_ASCEND_CURL_INSECURE=1`
bash
# 下载
bash benchmarks/scripts/dsa_trace/download_datasets.sh

# 准备
python3 benchmarks/scripts/dsa_trace/prepare_sharegpt.py \
  --input benchmarks/datasets/dsa_trace/ShareGPT_V3_unfiltered_cleaned_split.json \
  --output benchmarks/datasets/dsa_trace/sharegpt.prompts.jsonl \
  --num-prompts 200 \
  --max-tokens 256
```

---

## 8. 端到端采集流程

### Step 1：下载数据

```bash
bash benchmarks/scripts/dsa_trace/download_datasets.sh
```

### Step 2：准备 prompts / workload（按需执行上面 7.x 中的命令）

### Step 3：启动 server（开启 trace）

```bash
export VLLM_ASCEND_DSA_TRACE=1
export VLLM_ASCEND_TRACE_DIR=./trace_out
export VLLM_ASCEND_DSA_TRACE_DECODE_ONLY=1

bash benchmarks/scripts/dsa_trace/run_server.sh <model_path_or_id> [port]
```

`run_server.sh` 的核心操作就是 `vllm serve <model> --host 0.0.0.0 --port <port> --trust-remote-code`，同时把上面的环境变量透传给 worker 进程。

### Step 4：回放请求

**prompts 类（ShareGPT / RULER / LongBench v2）：**

```bash
python3 benchmarks/scripts/dsa_trace/replay_openai.py \
  --endpoint http://127.0.0.1:8000/v1/completions \
  --model <model_name> \
  --prompts-jsonl benchmarks/datasets/dsa_trace/sharegpt.prompts.jsonl \
  --concurrency 16
```

把 `--prompts-jsonl` 换成 `ruler.prompts.jsonl` / `longbenchv2.prompts.jsonl` 即可。

**workload 类（BurstGPT）：**

```bash
python3 benchmarks/scripts/dsa_trace/replay_openai.py \
  --endpoint http://127.0.0.1:8000/v1/completions \
  --model <model_name> \
  --workload-jsonl benchmarks/datasets/dsa_trace/burstgpt.workload.jsonl \
  --concurrency 64
```

BurstGPT 回放会**按到达时间间隔**发送请求（`arrival_offset_s`），模拟真实突发负载。

### Step 5：采集产出

```
trace_out/
└── dsa_20260222_141500_12345/
    ├── dsa_access.jsonl      # A+B+D
    ├── kv_io.jsonl           # C
    └── prefix_cache.jsonl    # D
```

---

## 9. 开销控制与常见问题

### 9.1 性能开销

| 场景 | 开销 |
|------|------|
| `VLLM_ASCEND_DSA_TRACE=0`（默认） | **零开销**（所有插桩被 `if tracer.enabled` 短路） |
| decode only + 全采样 | 主要开销在 `.to("cpu").numpy()` 搬运 topk_indices 和 block_table；decode 阶段 batch 通常很小，搬运量约 `num_heads * 2048 * 4B + block_table_row * 4B` |
| prefill 也采 | **不推荐**。prefill 阶段每个 request 可能有上千个 query token，会产生巨量 JSONL |

### 9.2 常见问题

**Q：为什么 `dsa_access.jsonl` 为空？**
- 确认 `VLLM_ASCEND_DSA_TRACE=1`
- 确认模型走的是 `AscendSFABackend`（DeepSeek V3.2-Exp 的 DSA 才走此 backend）
- 如果 `VLLM_ASCEND_DSA_TRACE_DECODE_ONLY=1`，prefill 阶段不会产生记录；确认有 decode step 发生

**Q：为什么没有 scores？**
- `npu_lightning_indexer` 内部做了 Q*K 计算和 top-k 排序，但**只返回 indices 张量**，不返回 scores
- 获取 scores 需要修改 `csrc/lightning_indexer_vllm/` 中的 C++ kernel，让算子额外输出一个 scores 张量

**Q：`kv_io.jsonl` 为空？**
- 只有启用了外部 KV 传输（Mooncake P2P / MemCache pool / AscendStore pool）时，C 点才会被触发
- 纯本地 HBM KV cache 场景不走 C 点

**Q：`prefix_cache.jsonl` 为空？**
- 只有启用了 prefix caching（CPU offload / KV pool）时，D 点才会被触发

**Q：如何只采某几层的数据？**
- `export VLLM_ASCEND_DSA_TRACE_LAYER_FILTER="model.layers.0.self_attn,model.layers.1.self_attn"`

---

## 10. 改动文件清单

### 新增文件

| 文件 | 说明 |
|------|------|
| `vllm_ascend/trace/__init__.py` | trace 包入口，导出 `get_dsa_tracer` |
| `vllm_ascend/trace/trace_writer.py` | 线程安全 JSONL writer（按 run_id 分目录） |
| `vllm_ascend/trace/dsa_tracer.py` | DSATracer 单例（维护 step 上下文、写 A/B/C/D 三类 JSONL） |
| `benchmarks/scripts/dsa_trace/README.md` | 本文档 |
| `benchmarks/scripts/dsa_trace/download_datasets.sh` | 下载 ShareGPT / BurstGPT 原始文件 |
| `benchmarks/scripts/dsa_trace/prepare_sharegpt.py` | ShareGPT -> prompts.jsonl |
| `benchmarks/scripts/dsa_trace/prepare_burstgpt.py` | BurstGPT CSV -> workload.jsonl |
| `benchmarks/scripts/dsa_trace/prepare_ruler.py` | RULER (HF datasets) -> prompts.jsonl |
| `benchmarks/scripts/dsa_trace/prepare_longbenchv2.py` | LongBench v2 (HF datasets) -> prompts.jsonl |
| `benchmarks/scripts/dsa_trace/run_server.sh` | 启动 vllm serve 并透传 trace 环境变量 |
| `benchmarks/scripts/dsa_trace/replay_openai.py` | 并发回放 prompts/workload 到 OpenAI API |

### 修改文件

| 文件 | 改动点 |
|------|--------|
| `vllm_ascend/envs.py` | 新增 7 个 `VLLM_ASCEND_DSA_TRACE_*` 环境变量 |
| `vllm_ascend/worker/model_runner_v1.py` | `__init__` 新增 `_dsa_trace_step_idx`；`execute_model()` 设置 step 上下文 |
| `vllm_ascend/attention/sfa_v1.py` | `AscendSFAImpl.forward()` 中插入 A/B/D 记录逻辑（约 120 行，在 `try/except` 中） |
| `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py` | `start_load_kv()` 中 `m_store.get()` 周围加 C 记录 |
| `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py` | `_transfer_kv_cache()` 中 `batch_transfer_sync_read` 后加 C 记录 |
| `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py` | `get_num_new_matched_tokens()` 中加 C+D 记录 |
| `vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload/cpu_kv_cache_manager.py` | `get_matched_num_and_touch()` 中加 D 记录 |
