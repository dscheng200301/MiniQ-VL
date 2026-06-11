# MiniQ-VL 评估指南

`eval_all.py` 用于统一评估和比较 Base、SFT、DPO 模型的图像描述能力。它支持从图像目录或 parquet 数据读取样本，并输出机器可读结果与 Markdown 报告。

## 支持的评估模式

```text
base    仅评估基础模型
sft     仅评估 SFT 模型
dpo     仅评估 DPO 模型
all     依次评估 Base、SFT、DPO
```

默认模式为 `all`。

## 准备评估数据

### 使用 COCO Val2017

```bash
python scripts/download_eval_data.py
```

脚本会下载并解压 COCO Val2017 图像到 `dataset/eval_images`，下载量约 600 MB。

### 使用自定义图像目录

```bash
python eval_all.py \
  --mode base \
  --image_dir /path/to/images \
  --max_samples 100
```

### 使用 parquet

```bash
python eval_all.py \
  --mode base \
  --data_path ./dataset/minimind-v_dataset/sft_i2t.parquet \
  --max_samples 100
```

## 常用命令

### 仅评估基础模型

```bash
python eval_all.py \
  --mode base \
  --base_model_path ./model/Qwen3-VL-2B-Instruct \
  --image_dir ./dataset/eval_images \
  --max_samples 100
```

### 比较 Base、SFT 和 DPO

```bash
python eval_all.py \
  --mode all \
  --base_model_path ./model/Qwen3-VL-2B-Instruct \
  --sft_checkpoint ./out/sft_vlm_merged \
  --dpo_checkpoint ./out/dpo_<timestamp>/dpo_final.pt \
  --image_dir ./dataset/eval_images \
  --max_samples 100
```

脚本中的 DPO 默认路径包含历史时间戳，因此实际评估时应显式传入当前检查点。

### 生成多条回答并计算 Self-BLEU

```bash
python eval_all.py \
  --mode sft \
  --sft_checkpoint ./out/sft_vlm_merged \
  --image_dir ./dataset/eval_images \
  --num_samples 4 \
  --temperature 0.8 \
  --top_p 0.9
```

只有 `--num_samples` 大于 1 时才会计算 Self-BLEU。

### 关闭耗时指标

```bash
python eval_all.py \
  --mode all \
  --image_dir ./dataset/eval_images \
  --no_clip \
  --no_judge
```

## 主要参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--mode` | `all` | 评估阶段 |
| `--max_samples` | `100` | 最大评估样本数 |
| `--max_new_tokens` | `512` | 单次生成最大新 token 数 |
| `--temperature` | `0.7` | 采样温度 |
| `--top_p` | `0.9` | nucleus sampling 阈值 |
| `--num_samples` | `1` | 每张图像生成回答数 |
| `--device` | `cuda:0` | 推理设备 |
| `--clip_model` | `./model/clip-vit-base-patch32` | 本地 CLIP 模型 |
| `--no_clip` | 关闭 | 不计算 CLIPScore |
| `--no_judge` | 关闭 | 不调用 LLM Judge |

默认提示词要求模型使用中文详细描述图像。

## 指标说明

### CLIPScore

衡量生成文本与图像在 CLIP 表征空间中的匹配程度。当前实现使用每个样本的第一条生成结果。

CLIPScore 适合观察图文一致性，但对细粒度事实、中文表达质量和复杂空间关系的覆盖有限。

### Self-BLEU

衡量同一图像多条生成结果之间的相似度。数值较高通常表示回答更相似、生成多样性更低。

该指标只有在每张图像生成多条回答时才有意义。

### LLM Judge

Judge 从五个维度评价回答：

1. 物体识别
2. 属性描述
3. 空间关系
4. 场景氛围
5. 语言流畅性

Judge 依赖外部 API，应配置：

```bash
export API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.minimaxi.com/v1"
```

也可使用 `OPENAI_API_KEY`。API 评分属于代理评价，应结合人工抽检使用。

### 长度与失败统计

报告还会统计生成文本长度以及加载、生成、指标计算等过程中的失败情况。比较模型时，应同时查看质量指标和失败数量，避免把缺失结果误判为性能提升。

## 输出文件

默认目录：

```text
eval_output/comprehensive_<timestamp>/
├── results.json
├── per_sample_base.jsonl
├── per_sample_sft.jsonl
├── per_sample_dpo.jsonl
├── report.md
└── comparison_report.md
```

- `results.json`：各阶段汇总结果。
- `per_sample_<stage>.jsonl`：逐样本生成文本、指标与错误信息。
- `report.md`：单模型汇总报告。
- `comparison_report.md`：评估多个阶段时生成的对比报告。

## 推荐评估流程

1. 先使用 5 至 10 张图像验证模型路径、processor、CLIP 和 API 配置。
2. 固定评估图像、提示词、采样参数和随机种子，比较不同检查点。
3. 使用 `--num_samples 1` 进行主要质量对比，再单独进行多样性实验。
4. 检查逐样本 JSONL 和失败统计，不只查看汇总均值。
5. 对关键实验进行人工盲评，确认代理指标与实际目标一致。

## 注意事项

- Base、SFT 合并模型目录与 DPO 权重的加载方式不同，请传入正确路径。
- Judge 请求会产生网络延迟和外部 API 成本。
- CLIP 模型必须存在于本地路径，或通过参数指向可用模型。
- 不同采样参数会显著影响结果，模型对比时必须保持一致。
- 当前评估主要面向图像描述，不代表模型在问答、OCR 或复杂推理任务上的完整能力。
