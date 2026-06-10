"""
MiniQ-VL 综合评估脚本

评估模式:
1. base: 基模评估
2. sft: SFT 后模型评估
3. dpo: DPO 后模型评估
4. all: 同时评估所有阶段并对比

评估维度:
1. CLIPScore: 图文对齐度
2. Self-BLEU: 生成多样性（多样本时）
3. LLM-Judge 结构化评分 (5维度):
   - 物体识别 / 属性描述 / 空间关系 / 场景氛围 / 语言流畅度
4. Bad Case 追踪 + 失败类型统计
5. 阶段对比（base vs sft vs dpo）

使用:
    python eval_all.py --mode all --model_path ./out/dpo_vlm_final.pt
    python eval_all.py --mode sft --model_path ./out/sft_vlm_final.pt
    python eval_all.py --mode base --model_path ./model/Qwen3-VL-2B-Instruct
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import argparse
import json
import io
import re
from collections import defaultdict
from datetime import datetime
from typing import List, Optional, Dict

import torch
import warnings
from PIL import Image
from tqdm import tqdm

from model.qwen_vl import QwenVLM, QwenVLMConfig
from trainer.trainer_utils import setup_seed, get_model_params
from utils.api_client import APIClient, _DEFAULT_MODEL, _DEFAULT_API_KEY, _DEFAULT_BASE_URL
from trainer.grpo_utils import CLIPScorer

warnings.filterwarnings('ignore')


# ============================================================
# Self-BLEU
# ============================================================

class SelfBLEUScorer:
    """计算 Self-BLEU，衡量多样性"""

    def __init__(self):
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            self.sentence_bleu = sentence_bleu
            self.smoothing = SmoothingFunction().method1
            self.loaded = True
        except ImportError:
            self.loaded = False

    def _tokenize_zh(self, text: str) -> List[str]:
        return list(text.replace(" ", ""))

    def score(self, texts: List[str]) -> float:
        if len(texts) <= 1:
            return 0.0

        if self.loaded:
            scores = []
            tokenized = [self._tokenize_zh(t) for t in texts]
            for i, hypo in enumerate(tokenized):
                refs = [tokenized[j] for j in range(len(tokenized)) if j != i]
                s = self.sentence_bleu(refs, hypo, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=self.smoothing)
                scores.append(s)
            return sum(scores) / len(scores)
        else:
            scores = []
            for i, t1 in enumerate(texts):
                overlaps = []
                for j, t2 in enumerate(texts):
                    if i == j:
                        continue
                    set1, set2 = set(t1), set(t2)
                    overlap = len(set1 & set2) / max(len(set1 | set2), 1)
                    overlaps.append(overlap)
                scores.append(sum(overlaps) / len(overlaps) if overlaps else 0)
            return sum(scores) / len(scores)


# ============================================================
# LLM Judge
# ============================================================

class StructuredLLMJudge:
    """5 维度结构化评分"""

    JUDGE_PROMPT = """你是一个专业的图像描述质量评估专家。请对以下AI生成的图像描述进行5个维度的评分。

评分维度（每项 1-5 分）：
1. 物体识别：是否准确识别图中的主要物体和人物
2. 属性描述：是否包含颜色、形状、材质、姿态等细节属性
3. 空间关系：是否描述了物体间的位置、层次、遮挡等空间关系
4. 场景氛围：是否描述了整体场景、环境、氛围、光线等
5. 语言流畅度：表达是否自然通顺、逻辑连贯

同时请判断该描述的主要失败类型（可多选）：
- 属性缺失：缺少颜色、形状、材质等细节
- 场景缺失：缺少整体场景和环境描述
- 空间缺失：缺少物体间位置关系
- 幻觉错误：描述了图中不存在的内容
- 过于简略：描述过于简短，信息量不足
- 过于冗余：描述冗长重复，含无关内容
- 无明显失败：描述质量良好

请严格按以下 JSON 格式输出，不要输出其他内容：
{{"物体识别": X, "属性描述": X, "空间关系": X, "场景氛围": X, "语言流畅度": X, "失败类型": ["类型1", "类型2"]}}

待评估的描述：
{description}"""

    DIMENSIONS = ["物体识别", "属性描述", "空间关系", "场景氛围", "语言流畅度"]

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: str = _DEFAULT_MODEL):
        self.client = APIClient(api_key=api_key, base_url=base_url, model=model)
        self.loaded = bool(self.client.api_key)
        if not self.loaded:
            print("Warning: LLM-Judge not configured")

    def score(self, description: str) -> Dict:
        default_result = {d: 3 for d in self.DIMENSIONS}
        default_result["失败类型"] = ["未评估"]
        default_result["total"] = 15

        if not self.loaded:
            return default_result

        # 最多尝试 2 次
        text = ""
        for attempt in range(2):
            try:
                if attempt == 0:
                    prompt = self.JUDGE_PROMPT.format(description=description)
                else:
                    prompt = (
                        "请对以下图像描述进行评分，只输出 JSON，不要其他内容。\n"
                        '格式: {"物体识别": x, "属性描述": x, "空间关系": x, "场景氛围": x, "语言流畅度": x, "失败类型": ["..."]}\n'
                        f"描述: {description}"
                    )

                text = self.client.chat([{"role": "user", "content": prompt}], max_tokens=1024, temperature=0.0)

                # 去掉 <think>...</think> 标签（MiniMax 等模型的 thinking 模式）
                clean = text.strip()
                end_think = clean.find("</think>")
                if end_think >= 0:
                    clean = clean[end_think + len("</think>"):].strip()
                # 如果还剩 <think> (如未闭合标签)，继续去掉
                if clean.startswith("<think>"):
                    clean = clean[len("<think>"):].strip()
                # 兜底：用正则去掉任何残余的 think 标签
                clean = re.sub(r'</?think>', '', clean).strip()
                # 去掉可能的 ```json 包裹，找到 { 开头 } 结尾的内容
                for prefix in ["```json\n", "```json", "```\n", "```"]:
                    if clean.startswith(prefix):
                        clean = clean[len(prefix):].strip()
                if clean.endswith("```"):
                    clean = clean[:-3].strip()
                start = clean.find("{")
                end = clean.rfind("}") + 1
                if start >= 0 and end > start:
                    clean = clean[start:end]

                parsed = json.loads(clean)

                if not isinstance(parsed, dict):
                    if attempt == 0:
                        print(f"  [Judge] got {type(parsed).__name__} instead of dict, retrying... ({text[:80]})")
                        continue
                    break

                for d in self.DIMENSIONS:
                    val = parsed.get(d, 3)
                    try:
                        parsed[d] = max(1, min(5, int(val)))
                    except (ValueError, TypeError):
                        parsed[d] = 3

                if "失败类型" not in parsed or not isinstance(parsed.get("失败类型"), list):
                    parsed["失败类型"] = ["未识别"]
                parsed["total"] = sum(parsed[d] for d in self.DIMENSIONS)
                return parsed

            except json.JSONDecodeError as e:
                if attempt == 0:
                    print(f"  [Judge] JSON decode error: {e}")
                    print(f"    raw: {text[:300]}")
                    continue
            except Exception as e:
                if attempt == 0:
                    print(f"  [Judge] error: {e} | raw: {text[:200]}")
                    continue

        return default_result


# ============================================================
# 模型加载
# ============================================================

def load_model(args):
    """加载模型"""
    config = QwenVLMConfig(model_path=args.model_path)
    model = QwenVLM(config).to(args.device)
    processor = model.processor
    get_model_params(model)
    return model.eval(), processor


def load_checkpoint(args):
    """加载 checkpoint"""
    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        return checkpoint
    return None


# ============================================================
# 生成描述
# ============================================================

def generate_description(model, processor, image: Image.Image, prompt: str, args):
    """生成图像描述"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]
        }
    ]

    text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(args.device)

    gen_kwargs = {
        'max_new_tokens': args.max_new_tokens,
        'do_sample': True,
        'temperature': args.temperature,
        'top_p': args.top_p,
    }
    if 'image_grid_thw' in inputs:
        gen_kwargs['image_grid_thw'] = inputs['image_grid_thw']
    if 'pixel_values' in inputs:
        gen_kwargs['pixel_values'] = inputs['pixel_values']

    generated_ids = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        **gen_kwargs
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return response


def generate_k_samples(model, processor, image: Image.Image, prompt: str, args, k: int = 1):
    """生成 K 条描述"""
    samples = []
    for _ in range(k):
        desc = generate_description(model, processor, image, prompt, args)
        samples.append(desc)
    return samples


# ============================================================
# 数据加载
# ============================================================

def load_eval_data(args):
    """加载评估数据"""
    samples = []

    if args.data_path and os.path.exists(args.data_path):
        import pyarrow.parquet as pq
        import pyarrow as pa

        table = pa.Table.from_batches(pq.ParquetFile(args.data_path).iter_batches())
        total = min(len(table), args.max_samples)

        for i in tqdm(range(total), desc="Loading data"):
            try:
                conversations = json.loads(table['conversations'][i].as_py())

                reference = ""
                for turn in conversations:
                    if turn.get('role') == 'assistant':
                        reference = turn.get('content', '')
                        break

                user_prompt = ""
                for turn in conversations:
                    if turn.get('role') == 'user':
                        content = turn.get('content', '')
                        content = content.replace('<image>', '').strip()
                        user_prompt = content
                        break

                image_bytes = table['image_bytes'][i].as_py()
                if not isinstance(image_bytes, list):
                    image_bytes = [image_bytes]
                if not image_bytes:
                    continue

                image = Image.open(io.BytesIO(image_bytes[0])).convert('RGB')

                samples.append({
                    "image": image,
                    "prompt": user_prompt,
                    "reference": reference,
                    "source": f"data_{i}",
                })
            except Exception:
                continue

    elif args.image_dir and os.path.exists(args.image_dir):
        image_files = sorted([
            f for f in os.listdir(args.image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])

        for fname in image_files[:args.max_samples]:
            try:
                image = Image.open(os.path.join(args.image_dir, fname)).convert('RGB')
                samples.append({
                    "image": image,
                    "prompt": args.prompt,
                    "reference": "",
                    "source": fname,
                })
            except Exception:
                continue
    else:
        print("ERROR: 请指定 --data_path 或 --image_dir")
        exit(1)

    return samples


def load_dpo_data(args):
    """加载 DPO 数据"""
    samples = []

    if args.dpo_data_path and os.path.exists(args.dpo_data_path):
        with open(args.dpo_data_path, 'r', encoding='utf-8') as f:
            dpo_data = json.load(f)

        total = min(len(dpo_data), args.max_samples)

        for i, item in enumerate(tqdm(dpo_data[:total], desc="Loading DPO data")):
            samples.append({
                "prompt": item.get("prompt", ""),
                "chosen": item.get("chosen", ""),
                "rejected": item.get("rejected", ""),
                "source": f"dpo_{i}",
            })

    return samples


# ============================================================
# 单模型评估
# ============================================================

def evaluate_single(args, model_name: str, checkpoint_path: str = None):
    """评估单个模型"""
    print(f"\n{'='*60}")
    print(f"评估模型: {model_name}")
    print(f"{'='*60}")

    setup_seed(args.seed)
    model, processor = load_model(args)

    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        print(f"Loaded checkpoint from {checkpoint_path}")

    clip_scorer = CLIPScorer(clip_model_name=args.clip_model, device=args.device) if not args.no_clip else None
    clip_scorer = clip_scorer or CLIPScorer.__new__(CLIPScorer)  # dummy
    if args.no_clip:
        clip_scorer.loaded = False
    self_bleu_scorer = SelfBLEUScorer()
    llm_judge = StructuredLLMJudge(api_key=args.judge_api_key, base_url=args.judge_base_url, model=args.judge_model) if not args.no_judge else None

    samples = load_eval_data(args)
    print(f"Loaded {len(samples)} samples")

    all_results = []
    dimension_scores = defaultdict(list)
    failure_type_counts = defaultdict(int)

    for idx, sample in enumerate(tqdm(samples, desc=f"Evaluating {model_name}")):
        image = sample["image"]
        prompt = sample["prompt"]
        reference = sample["reference"]

        # 生成 K 条描述
        descriptions = generate_k_samples(model, processor, image, prompt, args, k=args.num_samples)

        # CLIPScore
        clip_score = clip_scorer.score(image, descriptions[0]) if (clip_scorer.loaded and not args.no_clip) else -1

        # Self-BLEU
        self_bleu = self_bleu_scorer.score(descriptions) if args.num_samples > 1 else 0.0

        # LLM Judge
        judge_result = llm_judge.score(descriptions[0]) if not args.no_judge else {d: 3 for d in StructuredLLMJudge.DIMENSIONS}

        result = {
            "index": idx,
            "source": sample["source"],
            "generated": descriptions[0],
            "all_samples": descriptions,
            "reference": reference,
            "clip_score": round(clip_score, 4),
            "self_bleu": round(self_bleu, 4),
            "judge": judge_result,
            "text_length": len(descriptions[0]),
        }
        all_results.append(result)

        for d in StructuredLLMJudge.DIMENSIONS:
            dimension_scores[d].append(judge_result.get(d, 3))
        for ft in judge_result.get("失败类型", []):
            failure_type_counts[ft] += 1

    # 计算指标
    valid_clip = [r["clip_score"] for r in all_results if r.get("clip_score", -1) >= 0]
    avg_clip = sum(valid_clip) / len(valid_clip) if valid_clip else -1
    avg_self_bleu = sum(r["self_bleu"] for r in all_results) / len(all_results) if all_results else 0
    avg_length = sum(r["text_length"] for r in all_results) / len(all_results) if all_results else 0

    dimension_avg = {}
    for d in StructuredLLMJudge.DIMENSIONS:
        scores = dimension_scores[d]
        dimension_avg[d] = round(sum(scores) / len(scores), 2) if scores else 0

    avg_total_judge = round(sum(dimension_avg.values()), 2)

    summary = {
        "model_name": model_name,
        "model_path": args.model_path,
        "checkpoint": checkpoint_path,
        "num_samples": len(all_results),
        "avg_clip_score": round(avg_clip, 4),
        "avg_self_bleu": round(avg_self_bleu, 4),
        "avg_text_length": round(avg_length, 1),
        "avg_judge_total": avg_total_judge,
        "dimension_avg": dimension_avg,
        "failure_type_counts": dict(failure_type_counts),
    }

    return summary, all_results


# ============================================================
# DPO 偏好评估
# ============================================================

def evaluate_dpo(args, model_name: str, checkpoint_path: str = None):
    """评估 DPO 模型的偏好对"""
    print(f"\n{'='*60}")
    print(f"评估 DPO 模型: {model_name}")
    print(f"{'='*60}")

    setup_seed(args.seed)
    model, processor = load_model(args)

    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        print(f"Loaded checkpoint from {checkpoint_path}")

    clip_scorer = CLIPScorer(clip_model_name=args.clip_model, device=args.device)
    llm_judge = StructuredLLMJudge(api_key=args.judge_api_key, base_url=args.judge_base_url, model=args.judge_model)

    samples = load_dpo_data(args)
    print(f"Loaded {len(samples)} DPO samples")

    all_results = []
    dimension_scores_chosen = defaultdict(list)
    dimension_scores_rejected = defaultdict(list)
    failure_type_counts = defaultdict(int)

    for idx, sample in enumerate(tqdm(samples, desc=f"Evaluating DPO {model_name}")):
        chosen = sample.get("chosen", "")
        rejected = sample.get("rejected", "")

        judge_chosen = llm_judge.score(chosen)
        judge_rejected = llm_judge.score(rejected)

        result = {
            "index": idx,
            "source": sample["source"],
            "prompt": sample.get("prompt", ""),
            "chosen": chosen,
            "rejected": rejected,
            "judge_chosen": judge_chosen,
            "judge_rejected": judge_rejected,
            "chosen_length": len(chosen),
            "rejected_length": len(rejected),
        }
        all_results.append(result)

        for d in StructuredLLMJudge.DIMENSIONS:
            dimension_scores_chosen[d].append(judge_chosen.get(d, 3))
            dimension_scores_rejected[d].append(judge_rejected.get(d, 3))

        for ft in judge_chosen.get("失败类型", []):
            failure_type_counts[ft] += 1
        for ft in judge_rejected.get("失败类型", []):
            failure_type_counts[ft] += 1

    # 计算指标
    avg_chosen_length = sum(r["chosen_length"] for r in all_results) / len(all_results) if all_results else 0
    avg_rejected_length = sum(r["rejected_length"] for r in all_results) / len(all_results) if all_results else 0

    dimension_avg_chosen = {}
    dimension_avg_rejected = {}
    for d in StructuredLLMJudge.DIMENSIONS:
        scores_c = dimension_scores_chosen[d]
        scores_r = dimension_scores_rejected[d]
        dimension_avg_chosen[d] = round(sum(scores_c) / len(scores_c), 2) if scores_c else 0
        dimension_avg_rejected[d] = round(sum(scores_r) / len(scores_r), 2) if scores_r else 0

    avg_judge_chosen_total = round(sum(dimension_avg_chosen.values()), 2)
    avg_judge_rejected_total = round(sum(dimension_avg_rejected.values()), 2)

    # Chosen 胜率
    chosen_wins = sum(1 for r in all_results if r["judge_chosen"]["total"] > r["judge_rejected"]["total"])
    win_rate = chosen_wins / len(all_results) if all_results else 0.5

    summary = {
        "model_name": model_name,
        "model_path": args.model_path,
        "checkpoint": checkpoint_path,
        "num_samples": len(all_results),
        "avg_judge_chosen_total": avg_judge_chosen_total,
        "avg_judge_rejected_total": avg_judge_rejected_total,
        "avg_length_chosen": round(avg_chosen_length, 1),
        "avg_length_rejected": round(avg_rejected_length, 1),
        "dimension_avg_chosen": dimension_avg_chosen,
        "dimension_avg_rejected": dimension_avg_rejected,
        "chosen_win_rate": round(win_rate, 4),
        "failure_type_counts": dict(failure_type_counts),
    }

    return summary, all_results


# ============================================================
# 主评估流程
# ============================================================

def evaluate(args):
    """主评估流程"""
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.output_dir == "./eval_output":
        args.output_dir = f"./eval_output/comprehensive_{timestamp}"

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    results = {}
    all_samples = {}

    # Base 模型评估
    if args.mode in ["base", "all"]:
        args.model_path = args.base_model_path or "./model/Qwen3-VL-2B-Instruct"
        summary, samples = evaluate_single(args, "base", None)
        results["base"] = summary
        all_samples["base"] = samples

    # SFT 模型评估
    if args.mode in ["sft", "all"]:
        sft_checkpoint = args.sft_checkpoint or "./out/sft_vlm_merged"
        if os.path.exists(sft_checkpoint):
            if sft_checkpoint.endswith('.pt') and os.path.isfile(sft_checkpoint):
                args.model_path = args.base_model_path or "./model/Qwen3-VL-2B-Instruct"
                summary, samples = evaluate_single(args, "sft", sft_checkpoint)
            else:
                args.model_path = sft_checkpoint
                summary, samples = evaluate_single(args, "sft", None)
            results["sft"] = summary
            all_samples["sft"] = samples
        else:
            print(f"Warning: SFT checkpoint not found: {sft_checkpoint}")

    # DPO 模型评估
    if args.mode in ["dpo", "all"]:
        dpo_checkpoint = args.dpo_checkpoint or "./out/dpo_20260610_183121/dpo_final.pt"
        if os.path.exists(dpo_checkpoint):
            args.model_path = args.base_model_path or "./model/Qwen3-VL-2B-Instruct"
            summary, samples = evaluate_single(args, "dpo", dpo_checkpoint)
            results["dpo"] = summary
            all_samples["dpo"] = samples
        else:
            print(f"Warning: DPO checkpoint not found: {dpo_checkpoint}")

    # 保存结果
    with open(os.path.join(args.output_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 保存逐样本数据
    for stage, samples in all_samples.items():
        sample_file = os.path.join(args.output_dir, f"per_sample_{stage}.jsonl")
        with open(sample_file, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"Per-sample saved: {sample_file} ({len(samples)} samples)")

    # 生成综合可读报告
    _save_full_report(args.output_dir, results, all_samples)

    # 生成对比报告
    if len(results) > 1:
        _save_comparison_report(args.output_dir, results)

    # 打印报告
    _print_report(results)

    return results


def _save_full_report(output_dir: str, results: Dict, all_samples: Dict):
    """生成可读的完整评估报告 (report.md)"""
    lines = ["# MiniQ-VL 图文描述评估报告\n"]
    lines.append(f"**评估时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 总体指标
    lines.append("## 综合指标对比\n")
    lines.append("| 模型 | 样本数 | CLIPScore | 描述长度 | Judge总分 |")
    lines.append("|---|---|---|---|---|")
    for stage, res in results.items():
        clip = f"{res.get('avg_clip_score', '-'):.4f}" if isinstance(res.get('avg_clip_score'), (int, float)) else "-"
        lines.append(f"| {stage.upper()} | {res.get('num_samples')} | {clip} | {res.get('avg_text_length')} | {res.get('avg_judge_total')} |")
    lines.append("")

    # 维度评分
    if any(res.get("dimension_avg") for res in results.values()):
        lines.append("## 维度评分对比\n")
        dims = StructuredLLMJudge.DIMENSIONS
        lines.append("| 维度 | " + " | ".join(k.upper() for k in results.keys()) + " |")
        lines.append("|---|---|" + "|".join(["---"] * len(results)) + "|")
        for d in dims:
            row = [d]
            for res in results.values():
                score = res.get("dimension_avg", {}).get(d, "-")
                row.append(f"{score:.2f}" if isinstance(score, (int, float)) else "-")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # 逐样本展示 (base / sft / dpo 并排)
    if all_samples:
        lines.append("## 逐样本描述 (base / sft / dpo 并排对比)\n")
        stages = sorted(all_samples.keys())
        n = min(len(all_samples.get(stages[0], [])), 5)  # 只展示前 5 张
        for i in range(n):
            image_name = all_samples[stages[0]][i].get("source", f"#{i+1}")
            lines.append(f"### 样本 {i+1}: `{image_name}`\n")
            for stage in stages:
                if i < len(all_samples.get(stage, [])):
                    s = all_samples[stage][i]
                    desc = s.get("generated", "无")[:200]
                    clip = f"{s.get('clip_score', -1):.4f}" if s.get('clip_score', -1) >= 0 else "N/A"
                    lines.append(f"**{stage.upper()}** (CLIPScore: {clip}, 长度: {s.get('text_length', 0)}):\n\n> {desc}\n")
            lines.append("---\n")

    # ==================== 结果分析 ====================
    lines.append("## 结果分析\n")
    stages = sorted(results.keys())

    # 1. CLIPScore 分析
    clip_scores = {}
    for stage in stages:
        cs = results[stage].get("avg_clip_score", -1)
        if isinstance(cs, (int, float)) and cs >= 0:
            clip_scores[stage] = cs
    if clip_scores:
        best_clip = max(clip_scores, key=clip_scores.get)
        worst_clip = min(clip_scores, key=clip_scores.get)
        lines.append("### CLIPScore (图文对齐度)\n")
        lines.append(f"- 最优: **{best_clip.upper()}** ({clip_scores[best_clip]:.4f})")
        if len(clip_scores) > 1:
            gap = clip_scores[best_clip] - clip_scores[worst_clip]
            if gap < 0.01:
                lines.append(f"- 模型间差异极小 (差值 {gap:.4f})，SFT/DPO 对图文对齐影响不显著\n")
            elif gap < 0.05:
                lines.append(f"- {best_clip.upper()} 领先 {worst_clip.upper()} {gap:.4f}，略有提升\n")
            else:
                lines.append(f"- {best_clip.upper()} 显著领先 {worst_clip.upper()} {gap:.4f}\n")
        if all(v < 0.3 for v in clip_scores.values()):
            lines.append("- 所有模型 CLIPScore 偏低 (< 0.3)，可能原因：\n")
            lines.append("  - 生成的描述与图像内容偏差较大（幻觉）\n")
            lines.append("  - 描述风格与 CLIP 训练数据差异大（如过于文学化）\n")
            lines.append("  - 建议检查生成描述是否准确反映图中内容\n")
        elif all(v > 0.5 for v in clip_scores.values()):
            lines.append("- 所有模型 CLIPScore 良好 (> 0.5)，图文对齐度较高\n")

    # 2. 描述长度分析
    lengths = {}
    for stage in stages:
        ln = results[stage].get("avg_text_length", 0)
        if ln:
            lengths[stage] = ln
    if lengths:
        lines.append("### 描述长度\n")
        best_len = max(lengths, key=lengths.get)
        worst_len = min(lengths, key=lengths.get)
        lines.append(f"- {best_len.upper()} 最长 ({lengths[best_len]:.0f} 字符)，{worst_len.upper()} 最短 ({lengths[worst_len]:.0f} 字符)")
        if len(lengths) > 1:
            if "sft" in lengths and "base" in lengths:
                if lengths["sft"] > lengths["base"] * 1.2:
                    lines.append("- SFT 后描述明显变长，模型学会了更详细地描述\n")
                elif lengths["sft"] < lengths["base"] * 0.8:
                    lines.append("- SFT 后描述变短，模型可能被训练得更简洁\n")
            if "dpo" in lengths and "sft" in lengths:
                if lengths["dpo"] < lengths["sft"] * 0.9:
                    lines.append("- DPO 后描述缩短，冗余减少，描述更精炼\n")
                elif lengths["dpo"] > lengths["sft"] * 1.1:
                    lines.append("- DPO 后描述增长，需关注是否引入冗余\n")
        lines.append("")

    # 3. Judge 维度薄弱分析
    weak_dims = []
    for res in results.values():
        dim_avg = res.get("dimension_avg", {})
        for d, score in dim_avg.items():
            if isinstance(score, (int, float)) and score < 3.5:
                weak_dims.append(d)
    if weak_dims:
        from collections import Counter
        weak_counter = Counter(weak_dims)
        lines.append("### LLM-Judge 薄弱维度\n")
        for d, count in weak_counter.most_common(3):
            lines.append(f"- **{d}**: 多个模型评分偏低 ({count} 个模型 < 3.5)，建议针对性优化\n")
        lines.append("")

    # 4. 训练阶段有效性分析
    if len(stages) > 1:
        lines.append("### 训练阶段有效性\n")
        best_overall = None
        best_score = -1
        for stage, res in results.items():
            clip = res.get("avg_clip_score", -1)
            judge = res.get("avg_judge_total", 0)
            score = (clip if isinstance(clip, (int, float)) else 0) + (judge if isinstance(judge, (int, float)) else 0)
            if score > best_score:
                best_score = score
                best_overall = stage

        lines.append(f"- 综合最优模型: **{best_overall.upper()}**\n")
        if "base" in stages and "sft" in stages:
            base_clip = results["base"].get("avg_clip_score", -1)
            sft_clip = results["sft"].get("avg_clip_score", -1)
            if isinstance(base_clip, (int, float)) and isinstance(sft_clip, (int, float)) and sft_clip > base_clip + 0.02:
                lines.append("- SFT 有效提升了图文对齐度\n")
            else:
                lines.append("- SFT 未显著提升 CLIPScore，可检查训练数据质量\n")
        if "sft" in stages and "dpo" in stages:
            sft_len = results["sft"].get("avg_text_length", 0)
            dpo_len = results["dpo"].get("avg_text_length", 0)
            if dpo_len < sft_len:
                lines.append("- DPO 有效精简了描述，模型更高效\n")
            else:
                lines.append("- DPO 未显著改变描述长度\n")

        lines.append("\n### 后续建议\n")
        if best_overall == "base":
            lines.append("- 当前 SFT/DPO 训练未带来提升，建议：\n")
            lines.append("  - 检查 SFT 训练数据质量和规模\n")
            lines.append("  - 调整学习率/epoch 等超参数\n")
            lines.append("  - 验证模型是否正确加载了训练权重\n")
        elif best_overall == "sft":
            lines.append("- SFT 效果优于基座，DPO 需要优化\n")
        else:
            lines.append("- DPO 表现最优，可考虑进一步迭代\n")

    report_path = os.path.join(output_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report saved: {report_path}")


def _save_comparison_report(output_dir: str, results: Dict):
    """生成对比报告"""
    lines = []
    lines.append(f"# MiniQ-VL 模型对比评估报告")
    lines.append(f"")
    lines.append(f"## 整体指标对比")
    lines.append(f"")
    lines.append(f"| 模型 | CLIPScore | Self-BLEU | 描述长度 | Judge总分 |")
    lines.append(f"|---|---|---|---|---|")
    for stage, res in results.items():
        clip = res.get("avg_clip_score", "-")
        bleu = res.get("avg_self_bleu", "-")
        length = res.get("avg_text_length", "-")
        judge = res.get("avg_judge_total", "-")
        lines.append(f"| {stage.upper()} | {clip} | {bleu} | {length} | {judge} |")
    lines.append(f"")

    lines.append(f"## 维度评分对比")
    lines.append(f"")
    lines.append(f"| 维度 | " + " | ".join(k.upper() for k in results.keys()) + " |")
    lines.append(f"|---|---|" + "|".join(["---"] * len(results)) + "|")
    for dim in StructuredLLMJudge.DIMENSIONS:
        row = [dim]
        for res in results.values():
            dim_avg = res.get("dimension_avg", {})
            row.append(str(dim_avg.get(dim, "-")))
        lines.append("| " + " | ".join(row) + " |")
    lines.append(f"")

    with open(os.path.join(output_dir, "comparison_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _print_report(results: Dict):
    """打印评估报告"""
    print("\n" + "=" * 70)
    print("MiniQ-VL 综合评估报告")
    print("=" * 70)

    for stage, res in results.items():
        print(f"\n--- {stage.upper()} 模型 ---")
        print(f"样本数: {res.get('num_samples', '-')}")
        print(f"CLIPScore: {res.get('avg_clip_score', '-')}")
        print(f"Self-BLEU: {res.get('avg_self_bleu', '-')}")
        print(f"描述长度: {res.get('avg_text_length', '-')}")
        print(f"Judge总分: {res.get('avg_judge_total', '-')}")

        print(f"\n维度评分:")
        for d, score in res.get("dimension_avg", {}).items():
            bar = "★" * int(score) + "☆" * (5 - int(score))
            print(f"  {d}: {score:.2f} {bar}")

    print("\n" + "=" * 70)


def parse_args():
    p = argparse.ArgumentParser(
        description="MiniQ-VL 综合评估脚本\n"
                    "对 base / sft / dpo 模型在同一套 eval_images 图片集上生成描述，\n"
                    "评估 CLIPScore、LLM-Judge 5 维度评分、Self-BLEU 多样性等指标。",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ==================== 评估模式 ====================
    p.add_argument("--mode", type=str, default="all", choices=["base", "sft", "dpo", "all"],
                   help="评估模式:\n"
                        "  base  - 只评估基座模型\n"
                        "  sft   - 只评估 SFT 模型\n"
                        "  dpo   - 只评估 DPO 模型\n"
                        "  all   - 同时评估全部三个模型并生成对比报告 (默认)")

    # ==================== 模型路径 ====================
    p.add_argument("--base_model_path", type=str, default="./model/Qwen3-VL-2B-Instruct",
                   help="基座模型路径 (默认 ./model/Qwen3-VL-2B-Instruct)")
    p.add_argument("--model_path", type=str, default=None,
                   help="(废弃参数) 保留兼容")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="(废弃参数) 保留兼容")
    p.add_argument("--save_dir", type=str, default="./out",
                   help="训练输出目录 (默认 ./out)")

    # ==================== SFT / DPO 模型路径 ====================
    p.add_argument("--sft_checkpoint", type=str, default="./out/sft_vlm_merged",
                   help="SFT 模型路径\n"
                        "   - 如果是完整模型目录 (如 ./out/sft_vlm_merged), 直接加载\n"
                        "   - 如果是 .pt 文件, 加载基座模型后再载入权重\n"
                        "默认: ./out/sft_vlm_merged")
    p.add_argument("--dpo_checkpoint", type=str, default="./out/dpo_20260610_183121/dpo_final.pt",
                   help="DPO checkpoint .pt 文件路径 (默认 ./out/dpo_20260610_183121/dpo_final.pt)")

    # ==================== 评估数据 ====================
    p.add_argument("--data_path", type=str, default=None,
                   help="parquet 数据路径 (有则优先使用)\n"
                        "不指定时自动切换到 --image_dir 模式")
    p.add_argument("--dpo_data_path", type=str, default=None,
                   help="DPO 偏好数据 JSON 路径 (仅 dpo 模式使用)")
    p.add_argument("--image_dir", type=str, default="./dataset/eval_images/",
                   help="评估图像目录 (默认 ./dataset/eval_images/)\n"
                        "支持 .jpg/.png/.jpeg/.bmp 格式, 按文件名排序后取前 max_samples 张")
    p.add_argument("--max_samples", type=int, default=100,
                   help="最大评估样本数 (默认 100, 设为 0 表示全部)")

    # ==================== 生成参数 ====================
    p.add_argument("--prompt", type=str, default="请详细描述这张图片中的内容，包括物体、场景、颜色、位置关系等所有细节。",
                   help="图像描述 prompt (默认: 详细描述)")
    p.add_argument("--max_new_tokens", type=int, default=512,
                   help="最大生成长度 (默认 512)")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="生成温度 (默认 0.7, 设为 0 为贪婪解码)")
    p.add_argument("--top_p", type=float, default=0.9,
                   help="top-p 采样 (默认 0.9)")
    p.add_argument("--num_samples", type=int, default=1,
                   help="每张图生成 K 条描述 (默认 1, >1 时额外计算 Self-BLEU 多样性)")

    # ==================== 硬件 ====================
    p.add_argument("--device", type=str, default="cuda:0",
                   help="运行设备 (默认 cuda:0, cpu 可设为 cpu)")

    # ==================== CLIP 评分 ====================
    p.add_argument("--clip_model", type=str, default="./model/clip-vit-base-patch32",
                   help="CLIP 模型路径 (默认 ./model/clip-vit-base-patch32)\n"
                        "用于计算 CLIPScore 图文对齐度")
    p.add_argument("--no_clip", action="store_true",
                   help="跳过 CLIPScore 评分（加速评估）")
    p.add_argument("--no_judge", action="store_true",
                   help="跳过 LLM-Judge 评分（大幅加速，跳过 API 调用）")

    # ==================== LLM-Judge ====================
    p.add_argument("--judge_api_key", type=str, default=None,
                   help="Judge API key (留空用 api_client.py 默认值)")
    p.add_argument("--judge_base_url", type=str, default=None,
                   help="Judge base URL (留空用 api_client.py 默认值)")
    p.add_argument("--judge_model", type=str, default=_DEFAULT_MODEL,
                   help=f"Judge 模型 (默认 {_DEFAULT_MODEL})")

    # ==================== 其他 ====================
    p.add_argument("--seed", type=int, default=42,
                   help="随机种子 (默认 42)")

    # ==================== 输出 ====================
    p.add_argument("--output_dir", type=str, default="./eval_output",
                   help="输出目录 (默认 ./eval_output, 会自动追加时间戳)")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)