# MiniQ-VL 项目全流程

> 本文档面向面试场景，完整覆盖项目目标、数据、训练、效果等所有关键细节。

---

## 一、项目目标

**训练一个 2B 参数的多模态视觉语言模型（VLM），使其具备高质量的图像描述能力。**

具体来说：
- 基座模型：Qwen3-VL-2B-Instruct
- 通过 **SFT（监督微调）→ DPO（偏好对齐）** 两阶段训练，提升模型的图文描述质量
- 最终产出三个可对比的模型：Base（基座）、SFT（指令微调）、DPO（偏好对齐），并用统一的评估体系衡量各阶段提升

---

## 二、基座模型

| 项目 | 内容 |
|---|---|
| 基座模型 | **Qwen3-VL-2B-Instruct**（通义千问官方） |
| 模型大小 | **~1.72B 参数**（17.2 亿） |
| 视觉编码器 | **SigLIP-2**（Google，patch size 16×16） |
| 特征融合 | **DeepStack** 多层次视觉特征融合 |
| 位置编码 | **Interleaved-MRoPE** 3D 位置编码 |
| 视觉压缩比 | 32（将图像压缩为视觉 token） |
| 模型加载方式 | `AutoModelForImageTextToText`（HuggingFace transformers） |
| 精度 | **bfloat16** 混合精度 |

---

## 三、数据集

### 3.1 训练数据来源

| 阶段 | 数据集 | 来源 | 原始规模 | 实际使用 |
|---|---|---|---|---|
| **SFT** | `sft_i2t.parquet` | [minimind-v_dataset](https://www.modelscope.cn/datasets/gongjy/minimind-v_dataset)（魔搭） | ~290 万条 | **20 万条** |
| **DPO** | `dpo_i2t.json` | `prepare_dpo_dataset.py` 构造 | — | 约数千条 |

### 3.2 数据格式

SFT 数据为 **Parquet 格式**，每条样本包含：
```
image_bytes: 图像二进制数据
conversations: [
    {"role": "user", "content": "请详细描述这张图片。"},
    {"role": "assistant", "content": "这是一张展示城市天际线的照片..."}
]
```

### 3.3 DPO 数据构造

DPO 数据通过 `dataset/prepare_dpo_dataset.py` 自动构造：
- 从 SFT 数据采样图文对
- 调用 MiniMax API 生成低质量的 rejected 回答（减少细节、模糊描述）
- 原 SFT answer 作为 chosen，API 生成的低质量回答作为 rejected
- 保存为 JSON 格式的偏好对

### 3.4 评估数据

`./dataset/eval_images/` 下的自定义图像集（约 30 张，覆盖室内外、人物、物品、自然风光等场景）。

---

## 四、训练流程

```
基座模型 (Qwen3-VL-2B-Instruct)
    │
    ├─→ SFT 训练 (train_sft.py)
    │     使用 LoRA 微调
    │     冻结 Vision Encoder
    │     保存 merged 完整模型
    │
    ├─→ DPO 训练 (train_dpo.py)
    │     基于 SFT merged 模型
    │     偏好优化，减少冗余
    │     保存 .pt checkpoint
    │
    └─→ 评估 (eval_all.py)
          Base / SFT / DPO 三模型对比
```

---

## 五、SFT 训练细节

### 5.1 训练参数

| 参数 | 值 | 说明 |
|---|---|---|
| **GPU** | **4 × NVIDIA L20 (48GB)** | DDP 分布式训练 |
| **LoRA** | ✅ **开启** | rank=16, alpha=32, dropout=0.05 |
| LoRA 目标模块 | `q_proj, k_proj, v_proj, o_proj` | 只微调 Attention 层 |
| 冻结策略 | 冻结 Vision Encoder（SigLIP-2） | 不训练 visual 模块 |
| 学习率 | **2e-5** | AdamW optimizer |
| Batch Size | **1 / GPU** | 梯度累积 8 步 → 等效 batch = 32 |
| Epochs | **2** | |
| Max Seq Length | **1024** | 含图像 token |
| 精度 | **bfloat16** | 混合精度训练 |
| 梯度检查点 | ✅ 开启 | 节省显存 |
| torch.compile | ❌ 关闭 | DDP 下不稳定 |
| 训练样本数 | **200,000** | 从 290 万中采样 |

### 5.2 为什么用 LoRA？

- 2B 模型全量微调显存需求巨大
- LoRA rank=16 训练参数量约为全量的 **0.5%**（~860 万参数）
- 训练结束后 `lora_merge_and_save` 自动合并为完整模型，推理时无需额外加载 LoRA

### 5.3 显存占用

| 项目 | 占用 |
|---|---|
| 模型权重 (bfloat16) | ~3.4 GB |
| 优化器状态 (AdamW) | ~7 GB |
| 图像编码 + 激活值 | ~12 GB |
| 梯度 | ~1.5 GB |
| **单卡总占用** | **~22-26 GB** |
| **L20 显存 (48GB)** | 充足，有余量 |

### 5.4 训练时间

| 项目 | 时间 |
|---|---|
| **SFT**（20 万样本，2 epochs） | **约 6-7 小时**（4 × L20） |
| 训练步数 | 约 12,500 步 |
| 等效总 batch size | 32 |

### 5.5 SFT 启动命令

```bash
bash scripts/train_sft.sh
```

等效于：
```bash
torchrun --standalone --nproc_per_node=4 trainer/train_sft.py \
    --use_lora 1 --lora_r 16 --lora_alpha 32 \
    --freeze_vision 1 \
    --batch_size 1 --accumulation_steps 8 \
    --learning_rate 2e-5 --epochs 2 \
    --max_seq_len 1024 --max_samples 200000
```

---

## 六、DPO 训练细节

### 6.1 训练参数

| 参数 | 值 | 说明 |
|---|---|---|
| **GPU** | **4 × NVIDIA L20 (48GB)** | 与 SFT 相同 |
| **LoRA** | ❌ **未使用** | 基于 SFT merged 模型全量微调 |
| β (beta) | **0.1** | DPO 温度参数 |
| 学习率 | **5e-6** | 比 SFT 更小，防止灾难性遗忘 |
| Batch Size | **4 / GPU** | 无需梯度累积 |
| Epochs | **1** | DPO 收敛快，1 epoch 足够 |
| Max Seq Length | **512** | 仅处理文本，无需完整图像编码 |
| 冻结 Vision | ✅ | 冻结视觉编码器 |

### 6.2 训练时间

| 项目 | 时间 |
|---|---|
| **DPO**（数千条数据，1 epoch） | **约 20-30 分钟**（4 × L20） |

### 6.3 DPO 损失函数

```
loss = -logσ(β * (log_π(chosen|x) - log_π_ref(chosen|x)) 
              - β * (log_π(rejected|x) - log_π_ref(rejected|x)))
```

核心思想：让模型更倾向于生成 chosen（高质量）描述，远离 rejected（低质量）描述。

### 6.4 DPO 启动命令

```bash
python trainer/train_dpo.py \
    --sft_checkpoint ./out/sft_vlm_merged \
    --data_path ./dataset/minimind-v_dataset/dpo_i2t.json \
    --batch_size 4 --learning_rate 5e-6 --beta 0.1
```

---

## 七、评估系统

### 7.1 评估脚本

`eval_all.py` - 核心评估脚本，支持 **Base / SFT / DPO** 三模型在同一批图像上对比。

### 7.2 评估指标

| 指标 | 满分 | 测量方式 | 说明 |
|---|---|---|---|
| **CLIPScore** | — | CLIP 余弦相似度 | 描述与图像的语义对齐度 |
| **LLM-Judge 总分** | 25 | API 结构化评分 | 5 维度各 1-5 分 |
| ┣ 物体识别 | 5 | MiniMax 打分 | 图像中物体识别准确否 |
| ┣ 属性描述 | 5 | MiniMax 打分 | 颜色/形状/材质等 |
| ┣ 空间关系 | 5 | MiniMax 打分 | 物体间的空间位置描述 |
| ┣ 场景氛围 | 5 | MiniMax 打分 | 整体场景和氛围 |
| ┗ 语言流畅度 | 5 | MiniMax 打分 | 语言是否自然流畅 |
| **Self-BLEU** | — | BLEU 自相关 | 生成多样性（需 K>1） |
| **描述长度** | — | 字符数 | 生成文本平均长度 |

### 7.3 评估 API

使用 **MiniMax-M2.7-highspeed** 作为 LLM Judge：
- 内置滑动窗口速率限制：500 RPM / 20M TPM
- thinking 模式已禁用，确保返回结构化 JSON
- 快速评估可用 `--no_judge` 跳过 API 调用

### 7.4 按阶段预期的训练效果

| 阶段 | CLIPScore | 描述长度 | 预期效果 |
|---|---|---|---|
| **Base** | 基础值 | 基座输出 | 格式正确，能描述基本内容 |
| **SFT** | 提升 | 明显增长 | 描述更详细，覆盖更多属性 |
| **DPO** | 持平或略升 | 缩短 | 精简冗余，描述更精准高效 |

---

## 八、项目技术亮点

### 8.1 为什么选择 Qwen3-VL-2B-Instruct？

- **2B 参数**：显存友好，可在消费级 GPU 训练
- **原生多模态架构**：SigLIP-2 + DeepStack 融合，视觉编码能力领先同尺寸模型
- **中文优化**：Qwen 系列对中文支持优秀，适配图文描述任务

### 8.2 为什么 SFT→DPO 两阶段？

- **SFT**：教会模型"如何描述"（格式 + 基本能力）
- **DPO**：教会模型"什么描述更好"（质量对齐、去除冗余）
- 两阶段配合比单独 SFT 更有效，DPO 在保持 CLIPScore 的同时明显精简了输出

### 8.3 工程优化

| 优化手段 | 效果 |
|---|---|
| **LoRA 微调** | 训练参数仅 0.5%，大幅降低显存 |
| **梯度检查点** | 以时间换空间，节省 30%+ 显存 |
| **bfloat16 混合精度** | 几乎无损的速度提升和显存节省 |
| **梯度累积** | 小 batch size 模拟大 batch 效果 |
| **checkpoint 自动清理** | 只保留最近 2 个 checkpoint，节省磁盘 |
| **API 速率限制器** | 滑动窗口算法，避免被 API 限流 |

### 8.4 评估体系亮点

- **自动化对比**：base/sft/dpo 三个模型在同一批图像上对比
- **多维度评分**：5 维度结构化 LLM-Judge 评分
- **智能分析**：自动诊断薄弱维度，给出训练建议
- **逐样本输出**：`per_sample_*.jsonl` 支持人工抽检
- **加速选项**：`--no_judge` 跳过 API 可大幅加速

---

## 九、面试高频问题准备

### Q1: 为什么训练 2B 而不是 7B/14B？

A: 2B 模型可在 48GB L20 单卡完成推理，消费级硬件即可部署。同时 Qwen3-VL-2B 的视觉编码器与 7B 型号一致，图文理解能力有基础保障。

### Q2: LoRA 的 rank 怎么选的？

A: rank=16 是经过广泛验证的经验值。对于图文描述任务，16 提供了足够的表达能力，rank 再大对效果提升有限但增加训练成本。目标模块选择 `q_proj, k_proj, v_proj, o_proj` 覆盖 Attention 的全流程。

### Q3: 冻结 Vision Encoder 的考虑？

A: SigLIP-2 已经在大规模图文数据上预训练过，视觉特征提取能力足够强。微调视觉编码器容易导致过拟合且显存开销大，冻结是性价比最高的选择。

### Q4: DPO 的 beta 参数为什么选 0.1？

A: beta 控制与参考模型的距离，beta=0.1 是比较标准的设置。太大会限制 DPO 优化空间（模型不敢偏离 ref 太多），太小会导致策略漂移（忘记 SFT 学到的能力）。

### Q5: 怎么验证训练是否有效？

A: 三个维度验证：
1. **CLIPScore**：SFT/DPO 后应提升
2. **LLM-Judge 5 维度评分**：SFT 总分应提升，DPO 可能持平但描述更精炼
3. **人工抽检** `per_sample_*.jsonl`：直接对比三阶段生成的描述质量

### Q6: 如何处理训练过程中的问题？

A:
- **OOM**：降低 batch_size，增加 gradient checkpointing
- **Loss 不下降**：检查数据质量、学习率、LoRA rank
- **生成质量差**：检查 prompt 模板、max_new_tokens 设置
- **DPO 退化**：降低 beta 或增加训练 epoch

---

## 十、项目文件导航

| 文件 | 用途 |
|---|---|
| `model/qwen_vl.py` | 模型封装（加载 Qwen3-VL、冻结策略） |
| `dataset/sft_dataset.py` | SFT/Pretrain 数据集处理 |
| `dataset/prepare_dpo_dataset.py` | DPO 训练数据构造 |
| `trainer/train_sft.py` | SFT 训练入口 |
| `trainer/train_dpo.py` | DPO 训练入口 |
| `trainer/dpo_utils.py` | DPO 损失函数 |
| `trainer/grpo_utils.py` | CLIP 评分器 + 其他工具 |
| `utils/api_client.py` | 统一 LLM API 调用（含速率限制） |
| `eval_all.py` | 综合评估脚本（三模型对比） |
| `scripts/train_sft.sh` | SFT 启动脚本（含完整参数） |
| `scripts/train_grpo.sh` | GRPO 启动脚本 |

---

## 十一、快速复现

```bash
# 1. 环境
pip install -r requirements.txt

# 2. 下载基座模型
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3-VL-2B-Instruct', local_dir='./model/Qwen3-VL-2B-Instruct')"

# 3. 下载训练数据
python -c "from modelscope import snapshot_download; snapshot_download('gongjy/minimind-v_dataset', local_dir='./dataset/minimind-v_dataset')"

# 4. SFT 训练
bash scripts/train_sft.sh

# 5. 构造 DPO 数据
python dataset/prepare_dpo_dataset.py

# 6. DPO 训练
python trainer/train_dpo.py

# 7. 三模型对比评估
python eval_all.py --mode all --max_samples 30
```
