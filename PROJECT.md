# MiniQ-VL 项目说明

本文档描述当前代码实现的目标、架构、训练流程和主要约束。所有行为以仓库中的实际训练入口为准。

## 目标与边界

MiniQ-VL 提供一套可在单机单卡或多卡环境运行的视觉语言模型训练流程，重点覆盖：

- 基于图文样本的 SFT。
- 基于 chosen/rejected 样本的 DPO。
- 基于组内生成与奖励的 GRPO。
- Base、SFT、DPO 模型的统一生成评估。

项目不自行定义底层视觉编码器或语言模型结构；`model/qwen_vl.py` 负责加载现有多模态检查点并提供统一包装。

## 总体架构

```text
图像与文本数据
      │
      ├── SFT parquet ───────────────► train_sft.py
      │                                  │
      │                                  └── LoRA / 合并后的 SFT 模型
      │
      ├── DPO JSON ──────────────────► train_dpo.py
      │                                  │
      │                                  └── DPO 检查点
      │
      └── GRPO parquet ──────────────► train_grpo.py
                                         │
                                         ├── K 次独立生成
                                         ├── CLIP / Judge / 规则奖励
                                         └── GRPO 检查点

Base / SFT / DPO ────────────────────► eval_all.py ─► JSONL + Markdown 报告
```

## 核心模块

### 模型包装

`model/qwen_vl.py` 定义：

- `QwenVLMConfig`：模型路径、冻结策略等配置。
- `QwenVLM`：加载 `AutoProcessor` 与多模态生成模型。
- 加载时优先使用 `AutoModelForImageTextToText`，不可用时回退到 `AutoModelForCausalLM`。
- 视觉模块冻结通过参数名中的 `visual`、`vision`、`vit` 等标识完成。
- `forward` 与 `generate` 直接委托给内部 Hugging Face 模型。

### SFT

入口：`trainer/train_sft.py`

主要流程：

1. 从 parquet 读取图像与对话数据。
2. 使用 processor 构造多模态输入与标签。
3. 可选冻结视觉或语言模块。
4. 可选向内部模型注入 LoRA。
5. 使用 DDP、混合精度、梯度累积和梯度裁剪训练。
6. 定期保存检查点，恢复时跳过已消费批次并保留累积梯度状态。
7. 可选合并 LoRA 权重并导出可直接加载的模型目录。

`dataset/sft_dataset.py` 提供通用 `VLMDataset`，而训练入口还包含针对当前数据格式的 `MiniQVLMDataset`。运行行为以 `trainer/train_sft.py` 中实际实例化的数据集为准。

### DPO

入口：`trainer/train_dpo.py`

DPO 样本需要包含同一提示下的偏好回答和非偏好回答。训练器分别构造 chosen 与 rejected 完整多模态消息，并创建回答区域 mask。

序列得分仅累加助手回答 token 的对数概率：

```text
log π(y | x) = Σ response_mask[t] · log π(y_t | x, y_<t)
```

启用参考模型时，优化目标使用策略模型与冻结参考模型之间的偏好差异。默认 `--use_ref_model 1`，其代价是额外模型显存。

若 `--sft_checkpoint` 指向包含 `config.json` 的合并模型目录，训练器会直接将其作为初始策略模型加载。

### GRPO

入口：`trainer/train_grpo.py`

主要流程：

1. 为每个提示独立调用 `generate` 共 `group_size` 次。
2. 对每条生成结果计算奖励。
3. 在同一提示的回答组内标准化奖励，得到相对优势。
4. 使用裁剪目标更新策略模型。
5. 可选加入与参考模型之间的 KL 惩罚。

当前奖励由 `trainer/grpo_utils.py` 实现：

```text
R = w1·CLIPScore
  + w2·LLMJudge
  + w3·LengthReward
  + w4·AttributeCoverage
  + w5·Diversity
  + w6·HallucinationPenalty
```

其中属性覆盖、多样性和幻觉惩罚默认权重为 0。LLM Judge 返回 1 至 10 分，代码会归一化到 0 至 1。

多卡运行时，GRPO 使用 `DistributedSampler` 对样本索引分片，避免不同进程重复消费同一批数据。

### API 客户端

`utils/api_client.py` 提供兼容 OpenAI SDK 的 Judge 请求客户端：

- API Key：优先读取 `API_KEY`，同时支持 `OPENAI_API_KEY`。
- Base URL：默认 `https://api.minimaxi.com/v1`，支持 `OPENAI_BASE_URL`。
- 默认模型：`MiniMax-M2.7-highspeed`。
- 内置线程安全的滑动窗口限流与重试。
- 当前限制常量为每分钟 200 次请求、每分钟 10,000,000 tokens。

### 统一评估

入口：`eval_all.py`

支持 `base`、`sft`、`dpo`、`all` 四种模式，并可从 parquet 或图像目录读取评估数据。输出包括：

- 每阶段逐样本 JSONL。
- 汇总 `results.json`。
- 单模型 `report.md`。
- 多模型 `comparison_report.md`。

具体指标与命令见 [EVALUATION.md](EVALUATION.md)。

## 数据契约

### SFT / GRPO parquet

训练器预期每条记录能够解析出图像和对话文本。数据字段应与现有 `minimind-v_dataset` 保持一致；更换数据源时，应先通过少量样本验证 processor 输出、图像解码和标签 mask。

`dataset/prepare_grpo_dataset.py` 从 SFT parquet 中筛选适合 GRPO 的样本，并生成：

```text
grpo_i2t.parquet
grpo_i2t_meta.json
```

### DPO JSON

DPO 数据需要能够构造：

- 图像与用户提示。
- `chosen`：偏好回答。
- `rejected`：非偏好回答。

`dataset/prepare_dpo_dataset.py` 可借助 LLM 生成偏好数据，但当前路径、采样数量等配置写在脚本中，尚未完全参数化。

## 输出约定

默认输出根目录为 `./out`：

```text
out/
├── sft_vlm_merged/
├── dpo_<timestamp>/
│   └── dpo_final.pt
└── grpo_<timestamp>/
```

评估默认输出到：

```text
eval_output/comprehensive_<timestamp>/
```

模型、数据、训练输出与评估输出均已通过 `.gitignore` 排除，避免大文件进入版本库。

## 默认配置差异

Shell 启动脚本是面向常用训练环境的预设，其参数可能不同于 Python CLI 默认值。例如：

- `scripts/train_sft.sh` 默认按 4 卡配置，并使用每卡 batch size 1、梯度累积 8。
- `scripts/train_grpo.sh` 默认按 4 卡配置，启用离线模型加载，并在目标数据不存在时自动准备 GRPO 数据。
- `scripts/train_dpo.sh` 默认单卡，并限制样本数量以便先完成小规模训练。

调整实验时应检查实际启动命令，而不是只依赖 Python 参数默认值。

## 当前技术约束

- 参考模型会显著增加 DPO 与 GRPO 显存占用。
- GRPO 的成本与 `group_size`、生成长度、Judge 调用次数近似线性增长。
- CLIP 和 LLM Judge 属于代理指标，不能替代人工质量检查。
- API 失败、CLIP 模型不可用或输入异常时，需要关注日志中的奖励退化与失败计数。
- 真实模型兼容性、显存峰值和吞吐量应在目标 GPU 环境中验证。
- 预训练入口已存在，但没有与其他训练阶段同等完善的启动和数据准备流程。
