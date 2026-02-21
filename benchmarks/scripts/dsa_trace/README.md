## DSA 访问轨迹采集（数据集脚本 + 插桩说明）

本目录的目标是：在 **vLLM 0.13.0 + DeepSeek V3.2-Exp（DSA top-2048）** 的推理路径上，采集以下三类可复现日志（JSONL 落盘）：  
- **A+B：DSA token→block 访问轨迹**：`dsa_access.jsonl`  
- **C：外部 KV 取数/搬运代价**：`kv_io.jsonl`  
- **D：prefix cache 命中与交互**：`prefix_cache.jsonl`

这些脚本只用于“采集轨迹”，不做任何 KV 放置/迁移/副本/重算策略。

---

## 一、这次改了什么（重要）

### 1) 新增 Trace 基础设施（可开关）
- **新增目录**：`vllm_ascend/trace/`  
  - `trace_writer.py`：线程安全 JSONL writer（按 run_id 输出到目录）  
  - `dsa_tracer.py`：统一 tracer（记录 step 上下文、写 `dsa_access/kv_io/prefix_cache` 三类日志）
- **新增环境变量**：`vllm_ascend/envs.py`  
  - `VLLM_ASCEND_DSA_TRACE`：总开关（默认 0）  
  - `VLLM_ASCEND_TRACE_DIR`：输出目录（默认 `./trace_out`）  
  - `VLLM_ASCEND_DSA_TRACE_DECODE_ONLY`：默认只采 decode（默认 1）  
  - `VLLM_ASCEND_DSA_TRACE_SAMPLE_RATE`：采样率 \([0,1]\)  
  - `VLLM_ASCEND_DSA_TRACE_MAX_STEPS`：最多采集多少个 step（<0 不限）  
  - `VLLM_ASCEND_DSA_TRACE_LAYER_FILTER`：可选的 layer allowlist（逗号分隔）  

### 2) 插桩点 A/B：DSA top-2048 输出 + token→block 映射
- **step_idx 与 request_id 绑定**：在 `vllm_ascend/worker/model_runner_v1.py` 的 `execute_model()`  
  - 每次 `execute_model()` 递增 `step_idx`，并将当步 batch 的 `req_ids` 写进 tracer（用于 attention 层按 `req_idx` 找回 `request_id`）。
- **A/B 记录位置**：在 `vllm_ascend/attention/sfa_v1.py` 的 `AscendSFAImpl.forward()`  
  - 在 DSA `topk_indices` 产出后、真正 `npu_sparse_flash_attention` 之前落盘。  
  - **per-head 保存**：每个 token 记录 `selected_token_pos_by_head[num_heads][2048]`。  
  - 同时把所有 head 的 token 位置 union 后映射到 block：得到 `selected_block_ids`、`unique_blocks`、`tokens_per_touched_block` 的统计。

> 关于 **scores[2048]**：当前 NPU lightning indexer 内部计算了分数但没有返回到 Python（只返回 indices）。如果你后续必须记录 scores，需要改 C++/kernel 让 op 额外输出 scores 或在 kernel 内做统计后回传。

### 3) 插桩点 C：外部 KV 取数/搬运代价
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py`  
  - 同步 `m_store.get(...)` 周围记录：`bytes_read/read_ops/batch_size/latency_us`
- `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py`  
  - `batch_transfer_sync_read(...)` 周围记录：`bytes_read/read_ops/batch_size/latency_us`
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py`  
  - kvpool lookup 记录分层命中估算字段（HBM/remote 等）到 `kv_io.jsonl`

### 4) 插桩点 D：prefix cache 命中与 DSA 交互
- `vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload/cpu_kv_cache_manager.py`  
  - 记录 CPU prefix cache 的 `prefix_hit_tokens/prefix_cached_blocks`
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py`  
  - 记录 kvpool 视角的 prefix 命中（可理解为外部 KV 命中 token 数）
- `vllm_ascend/attention/sfa_v1.py`  
  - 对每条 DSA 记录附加：  
    - `prefix_cached_blocks`  
    - `dsa_prefix_intersection_ratio`（逻辑块空间：touched_logical_blocks 与 prefix blocks 的交集比例）  
    - `dsa_prefix_hot_blocks`（交集块的热点 Top-N，按被选 token 次数排序）

---

## 二、输出文件与字段定义（JSONL）

所有输出都在：`${VLLM_ASCEND_TRACE_DIR}/<run_id>/`  

### 1) `dsa_access.jsonl`（A+B）
每行代表 **一个 decode token（通常对应一个 request 的一次 decode）** 的 DSA 访问情况（per-head top-2048 + block 映射）。

关键字段：  
- `request_id`：请求 id（来自 batch req_ids）  
- `step_idx`：本次 `execute_model()` 的 step 计数  
- `seq_len_current`：当前序列长度（来自 `attn_metadata.seq_lens[req_idx]`）  
- `query_token_pos`：当前 query token 在序列中的位置（这里用 `seq_len_current - 1`）  
- `selected_token_pos_by_head`：二维数组 `[num_heads][2048]`，历史 token 的绝对位置  
- **轻量统计**：`unique_token_pos_count`、`offset_min/offset_median/offset_max`  
- **block 映射**：  
  - `block_size`  
  - `selected_block_ids`（去重后的物理 block id 列表）  
  - `unique_blocks`  
  - `tokens_per_touched_block.mean/p50/p95`  
- **prefix 交互**：  
  - `prefix_cached_blocks` / `prefix_cache_hit`  
  - `dsa_prefix_intersection_ratio`  
  - `dsa_prefix_hot_blocks`（示例元素：`{"logical_block": 3, "selected_tokens": 120}`）

### 2) `kv_io.jsonl`（C）
每行代表一次 KV I/O 相关操作（lookup / batch_get / transfer）。

关键字段：  
- `tier`：如 `kvpool`、`remote_pool`  
- `op`：如 `lookup`、`batch_get`、`batch_transfer_sync_read`  
- `request_id`  
- `bytes_read`、`read_ops`、`batch_size`、`latency_us`  
- `extra`：不同路径补充信息（backend 类型、session_id、分层命中估算等）

### 3) `prefix_cache.jsonl`（D）
每行代表一次 prefix cache 命中结果记录（CPU prefix / kvpool 视角）。

关键字段：  
- `request_id`  
- `prefix_cache_hit`（bool）  
- `prefix_hit_tokens`  
- `block_size`、`prefix_cached_blocks`  
- `source`：`cpu_prefix` 或 `kvpool`

---

## 三、如何跑 4 个数据集（只为采集轨迹）

### 0) 前置条件
- 你需要能运行 `vllm serve ...`（本仓库安装/镜像环境、NPU 驱动/CANN 等按你现有流程即可）
- 脚本默认用 `python3`

### 1) 下载原始数据（ShareGPT / BurstGPT）

```bash
bash benchmarks/scripts/dsa_trace/download_datasets.sh
```

默认落到：`benchmarks/datasets/dsa_trace/`

### 2) 准备 prompts / workload 文件

#### D4) ShareGPT → prompts.jsonl

```bash
python3 benchmarks/scripts/dsa_trace/prepare_sharegpt.py \
  --input benchmarks/datasets/dsa_trace/ShareGPT_V3_unfiltered_cleaned_split.json \
  --output benchmarks/datasets/dsa_trace/sharegpt.prompts.jsonl \
  --num-prompts 200 \
  --max-tokens 256
```

#### D3) BurstGPT → workload.jsonl（到达过程回放）

```bash
python3 benchmarks/scripts/dsa_trace/prepare_burstgpt.py \
  --input benchmarks/datasets/dsa_trace/BurstGPT_without_fails_2.csv \
  --output benchmarks/datasets/dsa_trace/burstgpt.workload.jsonl \
  --num-requests 200
```

> 注意：BurstGPT 本身是到达/长度分布，不含 prompt 文本。回放脚本会生成一个简单的合成 prompt 来逼近输入 token 长度（只用于触发并发与尾延迟，不做语义正确性）。

#### D1) RULER（HF datasets 拉取）→ prompts.jsonl

```bash
python3 benchmarks/scripts/dsa_trace/prepare_ruler.py \
  --output benchmarks/datasets/dsa_trace/ruler.prompts.jsonl \
  --num-prompts 200 \
  --max-tokens 256
```

#### D2) LongBench v2（HF datasets 拉取）→ prompts.jsonl

```bash
python3 benchmarks/scripts/dsa_trace/prepare_longbenchv2.py \
  --output benchmarks/datasets/dsa_trace/longbenchv2.prompts.jsonl \
  --num-prompts 200 \
  --max-tokens 128
```

### 3) 启动 server（打开 trace）

```bash
export VLLM_ASCEND_DSA_TRACE=1
export VLLM_ASCEND_TRACE_DIR=./trace_out
export VLLM_ASCEND_DSA_TRACE_DECODE_ONLY=1

bash benchmarks/scripts/dsa_trace/run_server.sh deepseek_model_path_or_id 8000
```

### 4) 回放请求（采集轨迹）

#### 回放 prompts（ShareGPT / RULER / LongBench v2）

```bash
python3 benchmarks/scripts/dsa_trace/replay_openai.py \
  --endpoint http://127.0.0.1:8000/v1/completions \
  --model deepseek_model_name \
  --prompts-jsonl benchmarks/datasets/dsa_trace/sharegpt.prompts.jsonl \
  --concurrency 16
```

把 `--prompts-jsonl` 换成 `ruler.prompts.jsonl` / `longbenchv2.prompts.jsonl` 即可。

#### 回放 workload（BurstGPT）

```bash
python3 benchmarks/scripts/dsa_trace/replay_openai.py \
  --endpoint http://127.0.0.1:8000/v1/completions \
  --model deepseek_model_name \
  --workload-jsonl benchmarks/datasets/dsa_trace/burstgpt.workload.jsonl \
  --concurrency 64
```

---

## 四、开销控制与常见坑

### 1) 默认只采 decode
`VLLM_ASCEND_DSA_TRACE_DECODE_ONLY=1` 能避免 prefill 大量 token 带来的日志爆炸，也更贴近你要的 decode step 访问轨迹。

### 2) 限制采集量
- `VLLM_ASCEND_DSA_TRACE_SAMPLE_RATE`：例如 `0.1`  
- `VLLM_ASCEND_DSA_TRACE_MAX_STEPS`：例如 `200`

### 3) 为什么没有 scores
当前 indexer op 只返回 indices，不返回 scores；要拿到 scores 需要改自定义算子/内核接口。

---

## 五、快速验证
跑完一次回放后，在 `${VLLM_ASCEND_TRACE_DIR}/<run_id>/` 应该至少看到：
- `dsa_access.jsonl`
- `kv_io.jsonl`（如果启用了外部 KV / transfer 路径）
- `prefix_cache.jsonl`（如果 prefix caching 路径被走到）


