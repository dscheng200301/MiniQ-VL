# MiniQ-VL

基于 **Qwen3-VL-2B-Instruct** 的多模态视觉语言模型训练框架，支持 SFT/DPO 等训练方式及多模型对比评估。

---

## 项目结构

```
MiniQ-VL/
├── model/
│   ├── __init__.py
│   └── qwen_vl.py              # Qwen3-VL 模型封装
├── dataset/
│   ├── __init__.py
│   ├── sft_dataset.py          # SFT/Pretrain 数据集处理
│   ├── grpo_dataset.py         # GRPO 数据集
│   ├── prepare_grpo_dataset.py # GRPO 数据预处理
│   └── prepare_dpo_dataset.py  # DPO 数据构建
├── trainer/
│   ├── __init__.py
│   ├── trainer_utils.py        # 训练工具
│   ├── grpo_utils.py           # GRPO 工具 (Reward/CLIP)
│   ├── dpo_utils.py            # DPO 损失函数
│   ├── train_pretrain.py       # 预训练脚本
│   ├── train_sft.py            # SFT 训练脚本
│   ├── train_dpo.py            # DPO 训练脚本
│   └── train_grpo.py           # GRPO 训练脚本
├── utils/
│   ├── __init__.py
│   └── api_client.py           # 统一 LLM API 调用 (内置速率限制)
├── scripts/
│   ├── download_eval_data.py   # 下载评估数据集
│   ├── train_sft.sh            # SFT 启动脚本
│   └── train_grpo.sh           # GRPO 启动脚本
├── eval_all.py                 # ★ 综合评估脚本 (base/sft/dpo 三模型对比)
├── out/                        # 模型权重输出
├── eval_output/                # 评估报告输出
├── dataset/
│   ├── eval_images/            # 评估图像
│   ├── sft_i2t.parquet         # SFT 训练数据
│   └── pretrain_i2t.parquet    # Pretrain 数据
├── requirements.txt
├── CHANGELOG.md                # 改动日志
└── EVALUATION.md               # 评估指南
```

---

## 模型架构

| 组件 | 配置 |
|---|---|
| **基座模型** | Qwen3-VL-2B-Instruct |
| **Vision Encoder** | SigLIP-2 (patch size 16×16) |
| **特征融合** | DeepStack 多层次融合 |
| **位置编码** | Interleaved-MRoPE (3D) |
| **压缩比** | 32 (视觉 token 压缩) |
| **模型加载** | `AutoModelForImageTextToText` |

---

## 环境要求

| 训练阶段 | 最低显存 | 推荐显存 | 预估时间 |
|---|---|---|---|
| Pretrain | 16GB | 24GB+ | 12-24h |
| SFT | 16GB | 24GB+ | 24-48h |
| DPO | 16GB | 24GB+ | ~1h |
| GRPO | 24GB | 40GB+ | 8-16h |

| 依赖 | 版本 |
|---|---|
| Python | ≥ 3.9 |
| PyTorch | ≥ 2.5.0 |
| transformers | ≥ 4.50.0 |
| CUDA | ≥ 11.8 |

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# MiniMax API Key（用于 LLM Judge 和数据筛选）
export API_KEY="your_api_key_here"

# Wandb（可选，用于训练可视化）
pip install wandb
wandb login
export WANDB_API_KEY="your_wandb_key"
```

> API Key 可在 [MiniMax 开放平台](https://platform.minimaxi.com/) 注册获取。

### 3. 下载模型和数据

```bash
# 基座模型
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3-VL-2B-Instruct', local_dir='./model/Qwen3-VL-2B-Instruct')"

# 训练数据
python -c "from modelscope import snapshot_download; snapshot_download('gongjy/minimind-v_dataset', local_dir='./dataset/minimind-v_dataset')"
```

### 4. 训练

```bash
python trainer/train_sft.py              # SFT 训练
python trainer/train_dpo.py              # DPO 训练（需要 SFT merged 模型）
```

### 5. 评估

评估前准备评估图像：

```bash
# 下载 COCO Val2017 评估图像（推荐，约 5000 张）
python scripts/download_eval_data.py

# 或准备自定义图像，放入 ./dataset/eval_images/
mkdir -p ./dataset/eval_images/
# 将图像放入此目录
```

评估：

```bash
# 三模型对比评估 (base / sft / dpo)
python eval_all.py --mode all --max_samples 30

# 快速评估 (跳过 LLM Judge API 调用)
python eval_all.py --mode all --max_samples 30 --no_judge

# 仅评估单个模型
python eval_all.py --mode base --max_samples 30
```

---

## 评估系统

`eval_all.py` 是核心评估脚本，支持对基座/SFT/DPO 三种模型在同一套图像上对比。

### 评估指标

| 指标 | 说明 |
|---|---|
| **CLIPScore** | 生成描述与图像的语义对齐度 |
| **LLM-Judge** | 5 维度结构化评分 (物体识别/属性描述/空间关系/场景氛围/语言流畅度) |
| **Self-BLEU** | 生成多样性 (需 `--num_samples > 1`) |
| **描述长度** | 平均生成长度 |
| **失败类型统计** | 未识别/幻觉/过短/属性缺失等 |

### 常用参数

```
python eval_all.py --mode all \
    --max_samples 30 \           # 评估图像数量
    --temperature 0.7 \          # 生成温度 (0=贪婪解码, 更快)
    --no_judge \                 # 跳过 LLM Judge (大幅加速)
    --no_clip                    # 跳过 CLIPScore (略加速)
```

### 输出文件

```
eval_output/comprehensive_YYYYMMDD_HHMMSS/
├── report.md                   # ★ 完整可读报告 (含结果分析)
├── comparison_report.md        # 维度对比表
├── results.json                # 汇总指标
├── per_sample_base.jsonl       # base 每张图的描述
├── per_sample_sft.jsonl        # sft 每张图的描述
└── per_sample_dpo.jsonl        # dpo 每张图的描述
```

> `report.md` 包含自动结果分析：CLIPScore 对比、描述长度变化趋势、薄弱维度诊断、训练阶段有效性判断、后续优化建议。

---

## API 配置

项目使用 `utils/api_client.py` 统一管理所有 LLM API 调用，默认使用 **MiniMax** 接口。

| 配置项 | 默认值 | 说明 |
|---|---|---|
| 模型 | `MiniMax-M2.7-highspeed` | Judge / 数据筛选共用 |
| Endpoint | `https://api.minimaxi.com/v1` | API Base URL |
| RPM | 500 | 每分钟请求数限制 |
| TPM | 20,000,000 | 每分钟 Token 数限制 |
| thinking | 已禁用 | 确保 JSON 格式输出 |

如需切换 API 服务商：

```bash
python eval_all.py --mode all \
    --judge_model gpt-4o \
    --judge_api_key sk-xxx \
    --judge_base_url https://api.openai.com/v1
```

---

## 模型路径

| 模型类型 | 示例路径 | 加载方式 |
|---|---|---|
| Base (基座) | `./model/Qwen3-VL-2B-Instruct` | 直接加载完整目录 |
| SFT (Merged) | `./out/sft_vlm_merged` | 直接加载完整目录 |
| DPO (.pt) | `./out/dpo_xxx/dpo_final.pt` | 先加载 base 再载入 .pt 权重 |

`eval_all.py` 自动处理三种路径格式：目录按 merged 模型加载，`.pt` 文件走 checkpoint 加载。

---

## 训练参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model_path` | ./model/Qwen3-VL-2B-Instruct | 模型路径 |
| `--data_path` | ./dataset/sft_i2t.parquet | 数据路径 |
| `--epochs` | 2 | 训练轮数 |
| `--batch_size` | 1 | batch size |
| `--learning_rate` | 1e-5 | 学习率 |
| `--freeze_vision` | 1 | 冻结视觉编码器 |
| `--use_wandb` | 1 | 使用 wandb |

### GRPO/DPO 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--group_size` | 4 | 每个 prompt 采样数 |
| `--temperature` | 0.8 | 采样温度 |
| `--w1` | 0.3 | CLIPScore 权重 |
| `--w2` | 0.5 | LLM-Judge 权重 |
| `--w3` | 0.2 | Length Penalty 权重 |
| `--clip_eps` | 0.2 | GRPO clip 范围 |
| `--clip_model` | ./model/clip-vit-base-patch32 | CLIP 本地路径 |

---

## 参考项目

- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)
- [MiniMind-V](https://github.com/jingyaogong/minimind-v)
- [transformers](https://github.com/huggingface/transformers)
