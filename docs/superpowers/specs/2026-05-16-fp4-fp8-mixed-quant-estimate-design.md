# FP4/FP8 混合量化精度建模 — `--estimate-config` 路径

> **状态**：Draft（待 user review）
> **日期**：2026-05-16
> **作者**：jiashaokun + Claude
> **影响范围**：`python/zrt/training/{spec,io,models,compose,search}/`，`python/zrt/hardware/configs/`，`tests/training/`
> **不在范围**：graph-capture 路径（已通过 `_bpe(op)` 正确读 op 级 dtype，无需改动）；Muon optimizer master/AG buffer dtype 拆分（master 始终 FP32，留 TODO）

---

## 1. 目标

让 `--estimate-config` 这条 spec-based 建模路径**真正建模** DeepSeek-V4 / V3 等使用的 FP4/FP8 混合精度量化方案。当前路径下：

- 所有 GEMM 一律按 BF16 peak TFLOPS 计时 → FP8/FP4 加速完全无法体现
- FP4 routed-expert 权重显存以字符串 hack 特判 → 不可扩展，不可读
- 激活显存不区分 region → MoE 内部 FP8 激活省下的显存不被反映
- MFU 计算分母硬编码 `flops_bf16` → 但这是**特性而非 bug**（与 V3 paper 对齐）

设计完成后，下列 YAML 应给出与论文 / 实测吻合的 step_time + peak memory：

```yaml
model:
  arch: deepseek_v4_pro
  quant_preset: "deepseek_v4_fp8_fp4"      # 一行展开 attn/moe/expert dtype
strategy:
  ep_overlap: megamoe
system:
  hw: nvidia_b300                          # 本次新增 spec
```

---

## 2. DeepSeek-V4 混合精度方案（来源对齐）

| 组件 | 权重 dtype | 计算 dtype | 来源 |
|---|---|---|---|
| Routed expert GEMM_up / GEMM_down | **FP4**（block=32 + BF16 scale） | **FP8 E4M3**(fwd) / **FP8 E5M2**(bwd grad) | V4 paper + DeepGEMM PR#316 + `docs/megamoe_modeling_zh.md` |
| Shared expert | BF16（V3 默认）/ FP8（可选） | 同左 | V4 paper 暧昧，做 knob |
| Attention QKV/Output projection + softmax | BF16 | BF16 | V3 paper §5.4，敏感路径保留高精度 |
| Embedding / lm_head | BF16 | BF16 | 同上 |
| Master weights | FP32 | — | ZeRO 内部累加 |
| Optimizer state (Adam moments / Muon state) | FP32 | — | 不变 |
| Gradient (聚合后) | FP32 | — | 默认；本设计**新增** `routed_expert_grad_dtype` 用于精细化 |

**实测校准点**（`megamoe_modeling_zh.md:58`）：大 batch 实测 TFLOPS / Blackwell FP8 peak ≈ 0.49 → 现有 `achieved_flops_efficiency` 返回 0.50 已吻合，FP8/FP4 复用同一曲线即可。

---

## 3. 数据流：从 YAML 到 step_time

```
YAML config
    ↓ io/config_loader.py::load_specs
    ├─ _expand_quant_preset(d)           ← 新增：展开 "deepseek_v4_fp8_fp4"
    └─ _parse_dtype + ModelSpec 字段填充
        ↓
ModelSpec(
    param_dtype, grad_dtype, master_dtype, act_dtype,           ← 原有
    attn_compute_dtype, attn_act_dtype,                         ← 新增
    moe_act_dtype,                                              ← 新增
    routed_expert_compute_dtype, routed_expert_weight_dtype,    ← 新增 / 收编
    shared_expert_compute_dtype,                                ← 新增
    routed_expert_grad_dtype,                                   ← 新增
)
    ↓
compose/stage.py::stage_time
    ↓ 对每个 op 调用：
_resolve_compute_dtype(op, model) → Dtype                       ← 新增 helper
    ↓
_cost_phase_time(cost, phase, system, gpu_name, dtype)          ← 加 dtype 形参
    ↓
op_to_time_hetero(..., dtype)
    ↓
peak_tflops_for(gpu, dtype) → float                             ← 新增 helper
    ├─ Dtype.BF16/FP16/FP32 → gpu.flops_bf16
    ├─ Dtype.FP8_*          → gpu.flops_fp8
    └─ Dtype.FP4            → gpu.flops_fp4 (新增 GPU 字段)
```

并行地：
- `memory.py` 按 region 拆 dtype 计算 weight / activation / EP 缓冲
- `comm.py:104` 把 DP all-reduce 拆成 expert + non-expert 两段，各用对应 grad dtype
- `compute_mfu / compute_hfu` 保持 `flops_bf16` 分母（默认 MFU 不变）；新增 `mfu_native` 字段，按 op-mix 加权有效 peak

---

## 4. 详细改动

### 4.1 `python/zrt/training/spec/dtype.py` — 扩展枚举

当前 enum 用 int value 兼任 byte size（FP32=4, BF16=2, ..., FP8=1）。无法承载 FP4(0.5B) 或区分 E4M3/E5M2。改为字符串 value + 独立 byte 属性：

```python
class Dtype(Enum):
    FP32 = "fp32"
    BF16 = "bf16"
    FP16 = "fp16"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    FP4 = "fp4"             # MXFP4-style block=32 + BF16 scale
    FP8 = "fp8"             # alias → FP8_E4M3，向后兼容

    @property
    def bytes(self) -> float:
        return {
            Dtype.FP32: 4.0, Dtype.BF16: 2.0, Dtype.FP16: 2.0,
            Dtype.FP8: 1.0, Dtype.FP8_E4M3: 1.0, Dtype.FP8_E5M2: 1.0,
            Dtype.FP4: 0.5,
        }[self]

    @property
    def block_overhead_bytes_per_elem(self) -> float:
        # FP4: block=32 个元素共享 1 个 BF16 (2B) scale → 2/32 = 0.0625
        return 2.0 / 32.0 if self is Dtype.FP4 else 0.0

    @property
    def stored_bytes(self) -> float:
        return self.bytes + self.block_overhead_bytes_per_elem
```

**向后兼容**：所有现有 `.bytes` 调用站点（`memory.py`, `comm.py`）继续 work，只是值变成 float（注意要 cast 到 int 的地方做对齐）。所有 YAML `"fp8"` 字符串经 `_parse_dtype` 映射到 `Dtype.FP8`（alias）。

---

### 4.2 `python/zrt/training/spec/model.py` — 新字段

```python
@dataclass
class ModelSpec:
    # === 现有字段 ===
    param_dtype: Dtype = Dtype.BF16
    grad_dtype: Dtype = Dtype.FP32
    master_dtype: Dtype = Dtype.FP32
    act_dtype: Dtype = Dtype.BF16           # 残差流默认精度
    routed_expert_dtype: str = "bf16"       # 保留作向后兼容 alias

    # === 新增：per-component compute dtype ===
    attn_compute_dtype: Dtype = Dtype.BF16
    shared_expert_compute_dtype: Dtype = Dtype.BF16
    routed_expert_compute_dtype: Dtype = Dtype.BF16     # V4 用 FP8_E4M3

    # === 新增：per-component weight dtype ===
    routed_expert_weight_dtype: Dtype = Dtype.BF16      # V4 用 FP4

    # === 新增：per-region activation dtype（None → fallback act_dtype）===
    attn_act_dtype: Dtype | None = None
    moe_act_dtype: Dtype | None = None

    # === 新增：per-component grad dtype ===
    routed_expert_grad_dtype: Dtype = Dtype.FP32        # V4 默认仍 FP32 (master)

    def __post_init__(self):
        # 向后兼容：routed_expert_dtype: str → routed_expert_weight_dtype
        if isinstance(self.routed_expert_dtype, str) and self.routed_expert_dtype != "bf16":
            mapped = _parse_dtype(self.routed_expert_dtype)
            if self.routed_expert_weight_dtype is Dtype.BF16:  # 仅在未显式设置时同步
                self.routed_expert_weight_dtype = mapped
```

**Fallback 规则**：
- `attn_act_dtype = attn_act_dtype or act_dtype`
- `moe_act_dtype = moe_act_dtype or routed_expert_compute_dtype if MoE else act_dtype`

由 `ModelSpec` 加 helper 属性 `effective_attn_act_dtype()` / `effective_moe_act_dtype()` 集中处理。

---

### 4.3 `python/zrt/hardware/configs/*.yaml` + spec — `fp4_tops` 字段

```yaml
# python/zrt/hardware/configs/nvidia_h100_sxm.yaml
compute:
  bf16_tflops: 989
  fp8_tops: 3958
  fp4_tops: 0          # 新增：H100 无原生 FP4

# python/zrt/hardware/configs/nvidia_b300.yaml （本次新建）
name: nvidia_b300
compute:
  bf16_tflops: 5000    # B300 dense BF16 ~5 PFLOPS
  fp8_tops: 15000      # B300 dense FP8 ~15 PFLOPS
  fp4_tops: 30000      # B300 dense FP4 ~30 PFLOPS
  # source: NVIDIA Blackwell Ultra GB300 datasheet
memory:
  hbm_capacity_gb: 288
  hbm_bw_gbps: 8000    # 8 TB/s HBM3e
interconnect:
  nvlink_bw_gbps: 1800   # NVLink 5 ~1.8 TB/s
  ...
```

`HardwareSpec.compute.fp4_tops: float = 0.0` 默认 0。`SystemSpec.gpu.flops_fp4` 经 `_parse_system` 加载。

**B300 数据源**：以 NVIDIA Blackwell Ultra GB300 公开 datasheet 为准，YAML 顶部加 `# source:` 注释指向 NVIDIA 官网链接。具体数值在实现 PR 中以最新公开数据校准。

---

### 4.4 `python/zrt/training/io/perf_tables.py` — 新增 peak 路由

```python
def peak_tflops_for(gpu, dtype: Dtype) -> float:
    """Return the hardware peak TFLOPS for given compute dtype.

    Falls back gracefully when hardware lacks native support:
      - FP4 on non-Blackwell → FP8 peak (one-time warning)
      - FP8 on non-FP8 hardware → BF16 peak
    """
    if dtype is Dtype.FP4:
        if getattr(gpu, "flops_fp4", 0) > 0:
            return gpu.flops_fp4 * 1e12
        _warn_once("fp4_fallback", f"GPU {gpu.name} lacks fp4_tops; falling back to fp8")
        dtype = Dtype.FP8
    if dtype in (Dtype.FP8, Dtype.FP8_E4M3, Dtype.FP8_E5M2):
        if getattr(gpu, "flops_fp8", 0) > 0:
            return gpu.flops_fp8 * 1e12
        _warn_once("fp8_fallback", f"GPU {gpu.name} lacks fp8_tops; falling back to bf16")
        dtype = Dtype.BF16
    return gpu.flops_bf16 * 1e12  # BF16/FP16/FP32 统一走 bf16 peak
```

`achieved_flops_efficiency(gpu_name, dtype, flops)` 函数签名不变；FP8/FP4 暂时复用 BF16 efficiency 曲线（megamoe doc:58 已校准吻合）。未来可加 dtype 分支。

---

### 4.5 `python/zrt/training/compose/stage.py` — 真正用 dtype 选 peak

**改动 1**：`op_to_time` / `op_to_time_hetero` 用 `peak_tflops_for` 替代硬编码。

```python
# 当前 line 62:
peak = gpu.flops_bf16 * 1e12

# 改为：
from zrt.training.io.perf_tables import peak_tflops_for
peak = peak_tflops_for(gpu, dtype)
```

对 `op_to_time_hetero` lines 103-107（heterogeneous Ascend），同样改为按 dtype 路由。注意 `cube_tflops` / `vector_tflops` 在 Ascend YAML 里是按 BF16 标注的；FP8/FP4 路径暂时复用同样的 cube/vector 拆分比例 + dtype peak ratio（建议加 helper `peak_cube_tflops_for` / `peak_vector_tflops_for`，初版直接以 `flops_bf16 : flops_fp8` 比例 scale）。

**改动 2**：`_cost_phase_time` 加 `dtype` 形参，向下传递。

```python
def _cost_phase_time(
    cost: OpCost, phase: str, system: SystemSpec,
    gpu_name: str, overlap: float = 0.0,
    dtype: Dtype = Dtype.BF16,      # 新增
) -> float:
    cube = getattr(cost, f"{phase}_cube_flops")
    vector = getattr(cost, f"{phase}_vector_flops")
    bytes_ = getattr(cost, f"{phase}_bytes")
    return op_to_time_hetero(cube, vector, bytes_, system, gpu_name,
                             overlap_ratio=overlap, dtype=dtype)
```

**改动 3**：新增 op→component 标签 + dtype 解析。

**Component 标签**：当前 `Op` 有 `kind`（operator 类型 matmul/rope/softmax/swiglu/...）和 `layer_kind`（layer-level enum DENSE/MOE/MTP），**没有**"哪个组件"的字段。具体 component 通过 op 名字前缀编码（`.qkv_proj`, `.routed_expert_ffn`, `.shared_up_proj`, `.embed`, `.lm_head`）。

**选择**：在 `Op` 上新增一个可选字段 `component: str | None = None`，在 `python/zrt/training/ir/builders.py` 各构建点显式标注。这是 single source of truth，比 name-based 模式匹配更稳。

```python
# python/zrt/training/ir/graph.py 在 Op dataclass 加：
@dataclass
class Op:
    ...
    component: str | None = None   # "attention" | "routed_expert" | "shared_expert" | "embedding" | "norm" | None
```

**builders.py 标注（实施时按 op 名规律批量加）**：

| 名字 pattern | component |
|---|---|
| `.qkv_proj`, `.q_a_proj`, `.q_b_proj`, `.kv_a_proj`, `.kv_b_proj`, `.wq_a`, `.wq_b`, `.wkv`, `.wo_a`, `.wo_b`, `.o_proj`, `.attn_core`, `.rope`, `.softmax` | `"attention"` |
| `.routed_expert_ffn`, `.expert_agg`, `.gate_proj` (路由 gate), `.hash_route`, `.compressor_pool`, `.indexer_topk` | `"routed_expert"` |
| `.shared_up_proj`, `.shared_gate_proj`, `.shared_down_proj`, `.shared_swiGLU` | `"shared_expert"` |
| `.embed`, `.lm_head` | `"embedding"` |
| `.mhc_pre`, `.mhc_post`, `.mhc_head` | `"norm"`（multi-head compression 视为 norm-like，BF16） |
| 其余 | `None`（fallback `act_dtype`）|

```python
def _resolve_compute_dtype(op: Op, model: ModelSpec) -> Dtype:
    """Map op.component to the compute dtype for its component.

    Falls back to model.act_dtype when component is unset.
    """
    comp = getattr(op, "component", None)
    if comp == "attention":
        return model.attn_compute_dtype
    if comp == "routed_expert":
        return model.routed_expert_compute_dtype
    if comp == "shared_expert":
        return model.shared_expert_compute_dtype
    if comp in ("embedding", "norm"):
        return Dtype.BF16    # 永远 BF16，与 V4 实践一致
    return model.act_dtype   # default fallback
```

**回归保护**：所有现有 op 创建点在添加 `component` 字段后，未标注的 op `component=None` → 走 `act_dtype` → 与现有行为完全一致。

**改动 4**：`stage_time` 主循环里，对每个 op 先解析 dtype，再传入 `_cost_phase_time`。

```python
for op in stage_ops:
    op_dtype = _resolve_compute_dtype(op, model)
    t_fwd  += _cost_phase_time(op.cost, "fwd", system, gpu_name, overlap, op_dtype)
    t_dx   += _cost_phase_time(op.cost, "dx",  system, gpu_name, overlap, op_dtype)
    t_dw   += _cost_phase_time(op.cost, "dw",  system, gpu_name, overlap, op_dtype)
```

---

### 4.6 `python/zrt/training/compose/schedules.py` — MFU 双轨

保留 `compute_mfu` / `compute_hfu` 现有行为（分母 = `flops_bf16`）→ 默认显示与 V3 paper 对齐。

新增 `compute_mfu_native(model, strategy, system, step_time, graph) -> float`，分母用 op-mix 加权：

```python
# 伪码
total_flops_by_dtype: dict[Dtype, float] = aggregate_flops_by_compute_dtype(graph)
effective_peak = sum(total_flops_by_dtype[d] for d in total_flops_by_dtype) / \
                 sum(total_flops_by_dtype[d] / peak_tflops_for(gpu, d) for d in total_flops_by_dtype)
mfu_native = total_flops / (effective_peak * step_time * num_gpus)
```

`StepResult` 加字段 `mfu_native: float | None = None`。Excel exporter 加一列。

---

### 4.7 `python/zrt/training/models/memory.py` — 按 region 拆 dtype

#### 4.7.1 权重（lines 108-150）

把现有 FP4 字符串特判替换为通用的"按组件 dtype 求和"：

```python
def _weight_bytes_for(p_count: int, dtype: Dtype) -> float:
    return p_count * dtype.stored_bytes   # FP4 自动包含 block scale

P_routed_expert = ...   # 路由专家参数量
P_shared_expert = ...
P_attention = ...
P_embedding = ...
P_other = P - (P_routed_expert + P_shared_expert + P_attention + P_embedding)

weights = (
    _weight_bytes_for(P_routed_expert, model.routed_expert_weight_dtype)
    + _weight_bytes_for(P_shared_expert, model.param_dtype)   # 暂用全局 param_dtype
    + _weight_bytes_for(P_attention, model.param_dtype)
    + _weight_bytes_for(P_embedding, model.param_dtype)
    + _weight_bytes_for(P_other, model.param_dtype)
)
```

**参数量拆分**：依赖 `ModelSpec` 现有 layer/expert 信息（`num_layers`, `num_experts`, `n_routed_experts`, `n_shared_experts`, `moe_inter_dim`, `hidden`, etc.）。新增 helper `model.param_count_by_component() -> dict[str, int]`，对 dense 模型 routed_expert 部分 = 0。

**P_routed_expert 计算公式**（per MoE layer）：`P_routed = 3 * hidden * moe_inter_dim * num_routed_experts`（SwiGLU 三矩阵 up/gate/down），累加 MoE 层数。
**P_shared_expert**：`P_shared = 3 * hidden * shared_inter_dim * num_shared_experts` per MoE layer。
**P_attention**：现有 attention 参数计算（MLA / 标准 / V4 三种 attention 拓扑各自有公式，复用现有 `_attention_params()` helper）。
**P_embedding**：`vocab_size * hidden * 2`（embedding + lm_head；若 weight tying 则 ×1）。
**P_other**：dense layer 中的 FFN 参数（V4 dense layer 仍是 SwiGLU），norm 参数等。

#### 4.7.2 梯度（line 137）

```python
grads = (
    P_routed_expert * model.routed_expert_grad_dtype.bytes
    + (P - P_routed_expert) * model.grad_dtype.bytes
)
```

#### 4.7.3 激活（lines 338-500）— region-aware

| 当前 line | 当前 dtype | 改为 |
|---|---|---|
| 338, 404 (`layer_act = s*h*act_bytes*coeff`) | `act_dtype` | **不变**（残差流 + 通用激活） |
| 405 (`hc_layer = (hc_mult-1)*s*h*act_bytes*coeff`) | `act_dtype` | **不变** |
| 414 (`5*num_heads*s*s*act_bytes` — QK^T score) | `act_dtype` | → `model.effective_attn_act_dtype.bytes` |
| 470 (MoE intermediate activation) | `act_dtype` | → `model.effective_moe_act_dtype.bytes` |
| 482 (TP AG/RS) | `act_dtype` | **不变**（残差流过 TP） |
| 491 (CP A2A) | `act_dtype` | → `model.effective_attn_act_dtype.bytes`（CP 仅 attn） |
| 500 (EP A2A) | `act_dtype` | → `model.routed_expert_compute_dtype.bytes` |

#### 4.7.4 Muon optimizer state（lines 89-91, 159; `optimizer.py:159,312`）

**本设计不动**。`master_dtype` 始终 FP32（user 已确认），hardcoded 4 与现状一致。代码注释加一行 `# TODO: scale by master_dtype when supporting non-FP32 master`。

---

### 4.8 `python/zrt/training/models/comm.py` — 梯度 + EP A2A dtype 拆分

**改动 1**：DP gradient all-reduce（line 104）。

```python
# 当前：
grad_bytes = P * model.grad_dtype.bytes

# 改为：拆 expert / non-expert 两段，因为 expert 不走 DP AR
P_expert = model.param_count_by_component().get("routed_expert", 0)
P_dp = P - P_expert    # 只有非 expert 走 DP AR
grad_bytes = P_dp * model.grad_dtype.bytes
```

> 注：routed expert 梯度在 EP-group 内部通过 reduce-scatter 处理，**不参与 DP AR**。若 EP < num_experts 导致 expert 在 EP-group 内还要 DP-style AR，需要在 `_ep_grad_comm()` 单独建模（**未来工作**，初版假设 EP shard 边界 = expert 边界）。

**改动 2**：EP A2A activation 字节数。`comm.py` 中的 EP dispatch/combine 字节数计算（具体行号实施时定位，搜索 `ep` + `dispatch` 关键字 + 现有 `act_dtype.bytes` 引用）替换为：

```python
# EP dispatch/combine 是路由专家激活，使用其 compute dtype
act_bytes_ep = model.routed_expert_compute_dtype.bytes
```

**改动 3**：PP P2P activation（line 210）**保持** `act_dtype`。残差流是 BF16，跨 PP stage 传递的是残差流不是 MoE 内部 FP8 值。

---

### 4.9 `python/zrt/training/io/config_loader.py` — YAML quant_preset

新增预设字典 + 展开函数：

```python
_QUANT_PRESETS = {
    "bf16_baseline": {
        "attn_compute_dtype": "bf16",
        "routed_expert_compute_dtype": "bf16",
        "routed_expert_weight_dtype": "bf16",
        "shared_expert_compute_dtype": "bf16",
        "routed_expert_grad_dtype": "fp32",
        "act_dtype": "bf16",
    },
    "fp8_mixed": {   # V3 风格
        "attn_compute_dtype": "bf16",
        "routed_expert_compute_dtype": "fp8_e4m3",
        "routed_expert_weight_dtype": "bf16",
        "shared_expert_compute_dtype": "bf16",
        "routed_expert_grad_dtype": "fp32",
        "act_dtype": "bf16",
        "moe_act_dtype": "fp8_e4m3",
    },
    "deepseek_v4_fp8_fp4": {   # V4 主推
        "attn_compute_dtype": "bf16",
        "routed_expert_compute_dtype": "fp8_e4m3",
        "routed_expert_weight_dtype": "fp4",
        "shared_expert_compute_dtype": "bf16",
        "routed_expert_grad_dtype": "fp32",
        "act_dtype": "bf16",
        "moe_act_dtype": "fp8_e4m3",
    },
    "deepseek_v4_full_fp8": {   # 假设场景：含 shared expert
        "attn_compute_dtype": "bf16",
        "routed_expert_compute_dtype": "fp8_e4m3",
        "routed_expert_weight_dtype": "fp4",
        "shared_expert_compute_dtype": "fp8_e4m3",
        "routed_expert_grad_dtype": "fp32",
        "act_dtype": "bf16",
        "moe_act_dtype": "fp8_e4m3",
        "attn_act_dtype": "bf16",
    },
}

def _expand_quant_preset(d: dict) -> dict:
    """Expand model.quant_preset into explicit dtype fields. Explicit
    fields in d override preset values."""
    preset_name = d.pop("quant_preset", None)
    if preset_name is None:
        return d
    preset = _QUANT_PRESETS[preset_name]
    for key, val in preset.items():
        d.setdefault(key, val)
    return d
```

`_parse_dtype` 扩展接受 `"fp8_e4m3"`, `"fp8_e5m2"`, `"fp4"`，并维护 `"fp8"` 别名 → `FP8_E4M3`。

---

### 4.10 Excel exporter

`python/zrt/training/io/excel_exporter.py` 增加列：
- "MFU (native)"（来自 `step.mfu_native`）
- "Routed Expert Weight Dtype"
- "Routed Expert Compute Dtype"
- "MoE Act Dtype"

不破坏现有列顺序，append 到末尾。

---

## 5. 测试矩阵

### 5.1 单元测试（新增）

`tests/training/test_mixed_quant_dtype.py`：
- `Dtype.FP4.bytes == 0.5` 与 `stored_bytes == 0.5625`
- `Dtype.FP8` alias 等于 `Dtype.FP8_E4M3`
- `_parse_dtype("fp8_e4m3")` 正确解析
- `Dtype.FP4.block_overhead_bytes_per_elem == 0.0625`

`tests/training/test_mixed_quant_peak_selection.py`：
- `peak_tflops_for(H100, FP8_E4M3) == 3958e12`
- `peak_tflops_for(H100, FP4) == 3958e12 + warning`（fallback to FP8）
- `peak_tflops_for(B300, FP4) == 30000e12`
- `peak_tflops_for(A100, FP8) == bf16_peak + warning`

`tests/training/test_mixed_quant_memory.py`：
- V4-Pro 配置下 FP4 routed expert vs BF16 routed expert：权重内存比例 ≈ 0.5625/2 = **0.28**
- FP8 moe_act vs BF16 moe_act：EP A2A 缓冲减半
- 现有 `routed_expert_dtype: "fp4"` 字符串 path 与新 `routed_expert_weight_dtype: Dtype.FP4` 结果**完全一致**（回归保护）

`tests/training/test_mixed_quant_op_dispatch.py`：
- 构造合成 op 列表（attn + routed_expert + shared_expert），各分别拿到正确 compute dtype
- `_cost_phase_time` 在不同 dtype 下时间正确缩放（FP8 ≈ BF16 / 2 on H100）

`tests/training/test_mixed_quant_preset.py`：
- `quant_preset: "deepseek_v4_fp8_fp4"` 展开后字段值正确
- 显式字段优先于 preset（preset 字典里 `attn_compute_dtype: bf16`，user 显式写 `attn_compute_dtype: fp8` → 用 `fp8`）

### 5.2 Anchor 回归（保护现有）

所有现有 anchor（`gpt3_175b_megatron`, `llama3_70b_meta`, `deepseek_v3_*`, `deepseek_v4_pro_*`）：
- YAML 不修改
- 行为 = 默认 dtype = BF16 = 现有 BF16 peak 路径
- **MFU / step_time 数值零变动**

CI gate：`test_anchor_mfu_strict` + `test_anchor_step_time_strict` 必须全绿。

### 5.3 新增 anchor

`tests/training/anchors/deepseek_v4_pro_fp8_fp4_h100.yaml`：
- 同 `deepseek_v4_pro_megamoe` 配置 + `quant_preset: deepseek_v4_fp8_fp4`
- 断言：
  - step_time_ms 比 megamoe BF16 baseline 低 **30–50%**
  - mfu (bf16 peak) ≈ 0.5–0.7 区间
  - peak memory 比 BF16 baseline 低 **30–40%**（权重占大头）
- `strict_mfu_check: false`（无公开校准点，进入 calibration mode）

`tests/training/anchors/deepseek_v4_pro_fp8_fp4_b300.yaml`：
- 同上配置 + `system.hw: nvidia_b300`
- 断言：FP4 路径走起，step_time 显著低于 H100
- `strict_mfu_check: false`

---

## 6. 实施顺序（写给 implementation 阶段）

按依赖顺序，每一步都应能独立 commit + 单元测过：

1. **Dtype 枚举扩展** + 单元测试（不影响任何业务代码）
2. **`peak_tflops_for` helper** + 单元测试
3. **`ModelSpec` 新字段 + `__post_init__` 兼容** + 单元测试
4. **`_parse_dtype` 扩展 + `quant_preset` 展开** + 单元测试
5. **`hardware/configs/nvidia_b300.yaml`** 新建 + `fp4_tops` 字段加到所有 spec
6. **`op_to_time` / `op_to_time_hetero` 用 `peak_tflops_for`** — 此时 dtype 形参终于生效
7. **`_cost_phase_time` 加 dtype 形参 + `_resolve_compute_dtype`** + 主循环改造
8. **`memory.py` weight 拆分** — 把字符串 FP4 path 替换为新接口；保留兼容
9. **`memory.py` 激活按 region 路由** — lines 414/470/491/500
10. **`memory.py` grad 拆 expert/non-expert**
11. **`comm.py` DP AR 排除 expert grad**
12. **`comm.py` EP A2A 用 routed_expert_compute_dtype**
13. **`compute_mfu_native` 新函数 + `StepResult.mfu_native`** + Excel 列
14. **新 anchor YAML × 2**（H100 + B300） + calibration mode 跑通
15. **回归测**：现有 anchor 全绿 + 新单测全绿

每一步可单独 PR，CI 应一直绿。

---

## 7. 风险与已知局限

| 风险 | 影响 | 应对 |
|---|---|---|
| `Dtype.value` 从 int 变 string 可能破坏依赖 `Dtype.FP32.value == 4` 的隐式代码 | 中（已知 `_optimizer_state_bytes` 等有 hardcoded 4） | 全仓 grep `.value` 用法 + 改成显式 `.bytes` |
| FP4 efficiency 曲线复用 BF16 数值，在 Blackwell B300 上可能严重偏差 | 中（B300 anchor 暂处于 calibration mode） | perf-table override path 已有（`megamoe_lookup`），未来 B300 实测可填表 |
| `param_count_by_component()` 需要从 `ModelSpec` 推导，dense 模型 routed_expert=0 | 低 | dense 模型自动走 BF16 default，行为不变 |
| Muon master FP32 hardcoded 与 `master_dtype` field 解耦 | 低 | 当前 user 确认 master 始终 FP32，加 TODO 注释即可 |
| Ascend cube/vector peak 缺 FP8 / FP4 子项 | 中 | 初版用比例 scale；具体硬件实测后填表 |
| Heterogeneous (Ascend) FP8 / FP4 路径 | 中 | 与 NVIDIA 路径同步加 fallback 警告 |

---

## 8. 显存收益量级估算（DeepSeek-V4-Pro 1.6T，H100 集群）

| 配置 | 路由专家权重 / GPU | 激活峰值 / GPU | step_time（预期）|
|---|---|---|---|
| Full BF16 (baseline) | ~5.7 GB | ~5.6 GB | T_0 |
| `fp8_mixed`（V3）| ~5.7 GB（weight 仍 BF16）| ~2.8 GB | ~0.65 T_0 |
| `deepseek_v4_fp8_fp4` | ~**1.6 GB** | ~2.8 GB | ~0.55 T_0 |
| `deepseek_v4_fp8_fp4` + B300 | ~1.6 GB | ~2.8 GB | ~**0.30 T_0**（FP4 peak ×4 + 互联升级） |

**总显存收益 V4-Pro per-GPU**：5.7+5.6 = 11.3 GB → 1.6+2.8 = 4.4 GB，**约 60% 显存节省**（仅权重 + 激活两项，未含 EP buffer / recompute 收益）。

---

## 9. 不在本设计范围

- Inference 路径的 KV cache 量化（V3.2 / V4 inference 用不同方案）
- Graph-capture 路径的 dtype 路由（`_bpe(op)` 已正确读 op 级 dtype，仅 peak 选择需要同步改 — 与本设计共用 `peak_tflops_for`）
- Optimizer master / Muon AG dtype 拆分（master 始终 FP32）
- 各 dtype 独立 efficiency 曲线 — 初版统一复用 BF16 曲线
- 通信带宽利用率与 dtype 关系（FP8 packed 通信可能利用率不同）— 暂用现有 `achieved_bandwidth_efficiency`
- B200 / H200 / Ascend 910C FP4 spec — 与 B300 同时按公开数据填，但优先级低

---

## 10. 验收标准

- [ ] `python -m python.zrt --estimate-config python/zrt/training/configs/deepseek_v4_pro_3d_h800.yaml` 使用 `quant_preset: deepseek_v4_fp8_fp4` 跑通
- [ ] 现有 anchor 全绿，MFU / step_time 数值零回归
- [ ] 新 anchor `deepseek_v4_pro_fp8_fp4_h100` + `_b300` 在 calibration mode 跑通
- [ ] 单元测试覆盖率覆盖到所有新增 dtype 路径
- [ ] Excel 报告显示新增 dtype 字段 + `mfu_native`
- [ ] B300 hardware spec 加入并被 `--hw nvidia_b300` 识别
