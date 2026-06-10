# MiniQ-VL 评估指南

本文档介绍 `eval_all.py` 综合评估脚本的使用方法。

---

## 概述

`eval_all.py` 是核心评估脚本，支持对 **Base / SFT / DPO** 三种模型在同一批图像上进行对比评估。

### 评估指标

| 指标 | 说明 |
|---|---|
| CLIPScore | 生成描述与图像的语义对齐度（基于 CLIP 模型） |
| LLM-Judge | 5 维度结构化评分（物体识别 / 属性描述 / 空间关系 / 场景氛围 / 语言流畅度） |
| Self-BLEU | 生成多样性（需 `--num_samples > 1`） |
| 描述长度 | 平均生成字符数 |
| 失败类型统计 | 未识别、幻觉、过短、属性缺失等 |

---

## 快速使用

```bash
# 三模型全面对比
python eval_all.py --mode all --max_samples 30

# 快速评估（跳过 LLM Judge，大幅加速）
python eval_all.py --mode all --max_samples 30 --no_judge

# 仅评估单个模型
python eval_all.py --mode base --max_samples 30

# 贪婪解码加速
python eval_all.py --mode all --max_samples 30 --temperature 0
```

---

## 评估图像

默认使用 `./dataset/eval_images/` 下的所有图像。准备方式：

```bash
# 把图像放入目录
mkdir -p ./dataset/eval_images/
cp /path/to/your/images/*.jpg ./dataset/eval_images/
```

> 图像越多、场景越多样，评估结果越可靠。建议至少 30 张。

---

## 常用参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--mode` | all | 评估模式：base / sft / dpo / all |
| `--max_samples` | 50 | 最多评估的图像数 |
| `--image_dir` | ./dataset/eval_images | 评估图像目录 |
| `--temperature` | 0.7 | 生成温度（0=贪婪解码，更快） |
| `--max_new_tokens` | 512 | 最大生成长度 |
| `--prompt` | 详细描述 | 生成 prompt |
| `--no_judge` | False | 跳过 LLM Judge（大幅加速） |
| `--no_clip` | False | 跳过 CLIPScore（略加速） |
| `--seed` | 42 | 随机种子 |

### 模型路径参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--base_model_path` | ./model/Qwen3-VL-2B-Instruct | 基座模型路径 |
| `--sft_checkpoint` | ./out/sft_vlm_merged | SFT 模型路径 |
| `--dpo_checkpoint` | ./out/dpo_20260610_183121/dpo_final.pt | DPO checkpoint |

---

## 输出文件

运行后在 `eval_output/comprehensive_YYYYMMDD_HHMMSS/` 生成：

| 文件 | 内容 |
|---|---|
| `report.md` | **完整可读报告**：综合指标 + 逐样本对比 + 自动结果分析 |
| `comparison_report.md` | 维度评分对比表 |
| `results.json` | 结构化汇总指标 |
| `per_sample_base.jsonl` | base 模型每条生成的完整数据 |
| `per_sample_sft.jsonl` | sft 模型每条生成的完整数据 |
| `per_sample_dpo.jsonl` | dpo 模型每条生成的完整数据 |

### report.md 内容结构

```
1. 综合指标对比表  (CLIPScore / 描述长度 / Judge总分)
2. 维度评分对比表  (5 维度并排)
3. 逐样本描述展示  (前 5 张图，base/sft/dpo 并排)
4. 结果分析
   - CLIPScore 分析 (谁最优、差距显著程度)
   - 描述长度趋势 (SFT后变长/变短、DPO是否精简)
   - 薄弱维度诊断 (哪些维度多个模型都低分)
   - 训练阶段有效性 (综合最优模型、SFT/DPO是否有效)
   - 后续优化建议
```

---

## 加速评估

| 手段 | 命令 | 效果 |
|---|---|---|
| 减少样本 | `--max_samples 10` | 最直接 |
| 跳过 Judge | `--no_judge` | 省 2~3s/张 |
| 贪婪解码 | `--temperature 0` | 生成快 ~30% |
| 仅 base | `--mode base` | 只跑 1 个模型 |
| 缩短输出 | `--max_new_tokens 256` | 减少解码步数 |

**推荐日常开发**：`--no_judge --temperature 0`，只看 CLIPScore + 人工抽检 `per_sample_*.jsonl`。
