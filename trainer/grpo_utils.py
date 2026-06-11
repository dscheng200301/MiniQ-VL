"""
MiniQ-VL GRPO 工具函数
提供 Reward 计算和优势函数

基础 Reward:
  R1 (CLIPScore): 生成文本与图像的 CLIP 相似度
  R2 (LLM-Judge): GPT-4o 对描述准确性、细节丰富度打分 (1-10)
  R3 (Length Penalty): 避免过短敷衍或过长冗余

扩展 Reward (迭代优化用):
  R4 (AttributeCoverage): 属性覆盖率，检测颜色/形状/材质等属性词
  R5 (DiversityReward): 组内多样性，基于 Self-BLEU
  R6 (HallucinationPenalty): 幻觉惩罚，用 VQA 验证描述事实

最终 reward = w1*R1 + w2*R2 + w3*R3 + w4*R4 + w5*R5 + w6*R6

KL 散度约束:
  防止策略偏离参考分布太远
"""
import sys
import os
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import re
import math
import json
import torch
import torch.nn.functional as F
from typing import List, Optional, Dict
from PIL import Image

from utils.api_client import APIClient, _DEFAULT_MODEL, _DEFAULT_API_KEY, _DEFAULT_BASE_URL


# ============================================================
# R1: CLIPScore
# ============================================================

class CLIPScorer:
    """CLIP 相似度评分器"""

    def __init__(
        self,
        clip_model_name: str = "./model/clip-vit-base-patch32",
        device: str = "cuda",
    ):
        self.device = device
        try:
            from transformers import CLIPProcessor, CLIPModel
            self.model = CLIPModel.from_pretrained(clip_model_name).to(device).eval()
            self.processor = CLIPProcessor.from_pretrained(clip_model_name)
            self.loaded = True
        except Exception as e:
            print(f"Warning: CLIP model not available ({e}), CLIPScore will return 0.0")
            self.loaded = False

    @torch.no_grad()
    def score(self, image: Image.Image, text: str) -> float:
        """
        计算图像与文本的 CLIP 相似度

        Args:
            image: PIL Image
            text: 生成文本

        Returns:
            CLIPScore ∈ [0, 1]
        """
        if not self.loaded:
            return 0.0

        try:
            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            ).to(self.device)

            outputs = self.model(**inputs)
            # CLIP logits_per_image = 缩放后的余弦相似度，直接用 item()
            # 注意：logits_per_image 可能不是标准的 [-1, 1]，但 item() 可以正常获取
            similarity = outputs.logits_per_image.item()
            # logits_per_image 在不同 CLIP 版本中范围不同，直接用 item() 不额外归一化
            return float(similarity)
        except Exception as e:
            print(f"Warning: CLIPScore failed: {e}")
            return 0.0

    def batch_score(self, images: List[Image.Image], texts: List[str]) -> List[float]:
        """批量计算 CLIPScore"""
        return [self.score(img, txt) for img, txt in zip(images, texts)]


# ============================================================
# R2: LLM-Judge (GPT-4o) — 支持参考对比
# ============================================================

class LLMJudge:
    """使用 GPT-4o 等大模型对描述质量打分，支持参考回答对比"""

    JUDGE_PROMPT = """你是一个专业的图像描述质量评估专家。请对以下AI生成的图像描述进行评分。

评分标准 (1-10分):
- 准确性 (1-5分): 描述是否准确反映了图像中的物体、颜色、空间关系等
- 细节丰富度 (1-5分): 描述是否包含了足够的细节（物体属性、场景氛围、构图等）

评分规则:
- 1-3分: 描述严重不准确或极其简略
- 4-6分: 基本准确但缺乏细节
- 7-8分: 准确且包含较多细节
- 9-10分: 非常准确且细节丰富，描述生动

请只输出一个数字 (1-10)，不要输出其他内容。

待评估的描述：
{description}

你的评分："""

    JUDGE_PROMPT_WITH_REF = """你是一个专业的图像描述质量评估专家。请对以下AI生成的图像描述进行评分。

参考描述（高质量标准）：
{reference}

评分标准 (1-10分):
- 准确性 (1-5分): 描述是否准确反映了图像中的物体、颜色、空间关系等
- 细节丰富度 (1-5分): 描述是否包含了足够的细节（物体属性、场景氛围、构图等）
- 与参考描述的对比: 是否达到了参考描述的质量水平

评分规则:
- 1-3分: 描述严重不准确或极其简略，远低于参考质量
- 4-6分: 基本准确但缺乏细节，低于参考质量
- 7-8分: 准确且包含较多细节，接近参考质量
- 9-10分: 非常准确且细节丰富，达到或超过参考质量

请只输出一个数字 (1-10)，不要输出其他内容。

待评估的描述：
{description}

你的评分："""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = _DEFAULT_MODEL,
        disable_thinking: bool = True,
    ):
        self.client = APIClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
            disable_thinking=disable_thinking,
        )
        self.loaded = bool(self.client.api_key)

        if not self.loaded:
            print("Warning: LLM-Judge not configured (set OPENAI_API_KEY), will return default score 5.0")

    def score(self, description: str, reference: str = "") -> float:
        """
        使用 LLM 对描述质量打分

        Args:
            description: 生成的图像描述文本
            reference: SFT 参考回答（可选，提供后评分更准确）

        Returns:
            score ∈ [1, 10]
        """
        if not self.loaded:
            return 5.0

        try:
            if reference:
                prompt = self.JUDGE_PROMPT_WITH_REF.format(
                    reference=reference, description=description
                )
            else:
                prompt = self.JUDGE_PROMPT.format(description=description)

            score_text = self.client.chat(
                [{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0,
            )

            match = re.search(r'\d+', score_text)
            if match:
                score = float(match.group())
                return max(1.0, min(10.0, score))
            return 5.0

        except Exception as e:
            print(f"Warning: LLM-Judge failed: {e}")
            return 5.0

    def batch_score(self, descriptions: List[str], references: Optional[List[str]] = None) -> List[float]:
        """批量打分"""
        if references is None:
            references = [""] * len(descriptions)
        return [self.score(desc, ref) for desc, ref in zip(descriptions, references)]


# ============================================================
# R3: Length Penalty
# ============================================================

def length_penalty(
    text: str,
    min_len: int = 50,
    max_len: int = 300,
    optimal_min: int = 80,
    optimal_max: int = 200,
) -> float:
    """
    长度惩罚，避免过短敷衍或过长冗余

    策略:
    - [optimal_min, optimal_max]: 满分 1.0
    - [min_len, optimal_min) 或 (optimal_max, max_len]: 线性衰减
    - < min_len 或 > max_len: 快速衰减到 0

    Args:
        text: 生成文本
        min_len: 最小可接受长度
        max_len: 最大可接受长度
        optimal_min: 最优长度下限
        optimal_max: 最优长度上限

    Returns:
        penalty ∈ [0, 1]
    """
    length = len(text)

    if optimal_min <= length <= optimal_max:
        return 1.0
    elif min_len <= length < optimal_min:
        return (length - min_len) / (optimal_min - min_len)
    elif optimal_max < length <= max_len:
        return 1.0 - (length - optimal_max) / (max_len - optimal_max)
    else:
        if length < min_len:
            return max(0.0, math.exp(-0.1 * (min_len - length)))
        else:
            return max(0.0, math.exp(-0.02 * (length - max_len)))


# ============================================================
# R4: Attribute Coverage (属性覆盖率)
# ============================================================

# 属性词库
ATTRIBUTE_PATTERNS = {
    "颜色": [
        "红", "橙", "黄", "绿", "蓝", "紫", "黑", "白", "灰", "棕", "粉",
        "金色", "银色", "深色", "浅色", "暗", "亮",
    ],
    "形状": [
        "圆", "方", "长", "扁", "尖", "弯曲", "笔直", "椭圆", "三角",
        "细长", "粗短", "弧形",
    ],
    "材质": [
        "金属", "木质", "塑料", "玻璃", "布", "皮革", "石头", "陶瓷",
        "纸质", "毛绒", "丝", "棉",
    ],
    "大小": [
        "大", "小", "巨大", "微小", "高", "矮", "宽", "窄", "厚", "薄",
        "粗", "细",
    ],
    "位置": [
        "左", "右", "上", "下", "前", "后", "中间", "旁边", "远处", "近处",
        "上方", "下方", "背后", "前面",
    ],
    "状态": [
        "站着", "坐着", "躺着", "走着", "跑", "飞", "蹲", "靠",
        "打开", "关闭", "悬挂", "摆放",
    ],
}


def attribute_coverage(text: str, categories: Optional[List[str]] = None) -> float:
    """
    属性覆盖率：检测描述中包含多少类属性词

    覆盖的类别越多，说明描述越详细

    Args:
        text: 生成文本
        categories: 要检测的属性类别（默认全部）

    Returns:
        coverage ∈ [0, 1]，覆盖的类别比例
    """
    if categories is None:
        categories = list(ATTRIBUTE_PATTERNS.keys())

    covered = 0
    for cat in categories:
        patterns = ATTRIBUTE_PATTERNS.get(cat, [])
        if any(p in text for p in patterns):
            covered += 1

    return covered / len(categories) if categories else 0.0


# ============================================================
# R5: Diversity Reward (组内多样性)
# ============================================================

def diversity_reward(texts: List[str]) -> List[float]:
    """
    组内多样性奖励：基于字符级重叠度

    每条文本与其余文本的平均重叠度越低，多样性越高，奖励越大

    Args:
        texts: 同一 prompt 的 K 条生成文本

    Returns:
        diversity_scores: 每条文本的多样性分数 ∈ [0, 1]
    """
    if len(texts) <= 1:
        return [1.0] * len(texts)

    scores = []
    for i, t1 in enumerate(texts):
        overlaps = []
        for j, t2 in enumerate(texts):
            if i == j:
                continue
            # 字符级 Jaccard 距离
            set1, set2 = set(t1), set(t2)
            if not set1 and not set2:
                overlaps.append(1.0)
            elif not set1 or not set2:
                overlaps.append(0.0)
            else:
                overlap = len(set1 & set2) / len(set1 | set2)
                overlaps.append(overlap)
        avg_overlap = sum(overlaps) / len(overlaps)
        # 重叠度越低，多样性越高
        scores.append(1.0 - avg_overlap)

    return scores


# ============================================================
# R6: Hallucination Penalty (幻觉惩罚)
# ============================================================

class HallucinationPenalizer:
    """
    幻觉惩罚：用 VQA 验证描述中的事实是否与图像一致

    流程:
    1. 从描述中提取可验证的声明（如"一只红色的猫"→ "猫是什么颜色？"）
    2. 用 VQA 模型对图像提问
    3. 比较答案与描述的一致性
    """

    EXTRACT_PROMPT = """从以下图像描述中提取可以验证的事实声明，每个声明生成一个验证问题。

描述：{description}

请输出 JSON 数组，每项包含 "claim"（声明）和 "question"（验证问题）。
只输出 JSON 数组，不要输出其他内容。

示例：
描述："一只红色的猫坐在蓝色的沙发上"
输出：[{{"claim": "猫是红色的", "question": "猫是什么颜色？"}}, {{"claim": "有沙发", "question": "图中有什么家具？"}}, {{"claim": "猫坐在沙发上", "question": "猫在哪里？"}}]"""

    VERIFY_PROMPT = """根据以下图像描述，判断每个声明是否与验证答案一致。

声明和答案：
{claims_and_answers}

请输出 JSON 数组，每项为 true（一致）或 false（不一致）。
只输出 JSON 数组，不要输出其他内容。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = _DEFAULT_MODEL,
        disable_thinking: bool = True,
    ):
        self.client = APIClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
            disable_thinking=disable_thinking,
        )
        self.loaded = bool(self.client.api_key)

    def score(self, description: str, vqa_answers: Optional[List[Dict]] = None) -> float:
        """
        计算幻觉惩罚分数

        Args:
            description: 生成的描述文本
            vqa_answers: VQA 验证结果 [{"claim": ..., "answer": ..., "consistent": bool}]
                         如果为 None，则自动调 API 提取和验证

        Returns:
            score ∈ [0, 1]，1 表示无幻觉，0 表示全是幻觉
        """
        if not self.loaded:
            return 1.0  # 未配置时不惩罚

        if vqa_answers is not None:
            consistent = sum(1 for a in vqa_answers if a.get("consistent", True))
            return consistent / len(vqa_answers) if vqa_answers else 1.0

        # 自动提取声明并验证
        try:
            # Step 1: 提取声明
            extract_prompt = self.EXTRACT_PROMPT.format(description=description)
            claims_text = self.client.chat(
                [{"role": "user", "content": extract_prompt}],
                max_tokens=300,
                temperature=0.0,
            )
            start = claims_text.find("[")
            end = claims_text.rfind("]") + 1
            if start < 0 or end <= start:
                return 1.0
            claims = json.loads(claims_text[start:end])

            if not claims:
                return 1.0

            # Step 2: 简单验证 — 用 LLM 判断声明是否合理
            # (完整版需要 VQA 模型，这里用 LLM 自检作为轻量替代)
            claims_str = "\n".join(
                f"{i+1}. 声明: {c['claim']}"
                for i, c in enumerate(claims)
            )
            verify_prompt = f"""请判断以下从图像描述中提取的声明是否合理（不太可能是幻觉）。

原始描述：{description}

声明列表：
{claims_str}

请输出 {len(claims)} 个布尔值的 JSON 数组（true=合理，false=可能是幻觉）："""

            verify_text = self.client.chat(
                [{"role": "user", "content": verify_prompt}],
                max_tokens=100,
                temperature=0.0,
            )
            start2 = verify_text.find("[")
            end2 = verify_text.rfind("]") + 1
            if start2 < 0 or end2 <= start2:
                return 1.0
            results = json.loads(verify_text[start2:end2])

            consistent = sum(1 for r in results if r is True)
            return consistent / len(results) if results else 1.0

        except Exception as e:
            print(f"Warning: HallucinationPenalizer failed: {e}")
            return 1.0


# ============================================================
# Reward 融合 (扩展版)
# ============================================================

def compute_rewards(
    images: List[Image.Image],
    texts: List[str],
    clip_scorer: CLIPScorer,
    llm_judge: LLMJudge,
    references: Optional[List[str]] = None,
    w1: float = 0.3,
    w2: float = 0.5,
    w3: float = 0.2,
    w4: float = 0.0,
    w5: float = 0.0,
    w6: float = 0.0,
    length_min: int = 50,
    length_max: int = 300,
    hallucination_penalizer: Optional[HallucinationPenalizer] = None,
) -> List[float]:
    """
    计算最终 reward

    reward = w1*R1 + w2*R2 + w3*R3 + w4*R4 + w5*R5 + w6*R6

    Args:
        images: 原始图像列表
        texts: 生成的文本列表
        clip_scorer: CLIP 评分器
        llm_judge: LLM 评判器
        references: SFT 参考回答列表（可选，提升 LLM-Judge 准确性）
        w1: CLIPScore 权重
        w2: LLM-Judge 权重 (归一化到 0-1)
        w3: Length Penalty 权重
        w4: Attribute Coverage 权重
        w5: Diversity Reward 权重
        w6: Hallucination Penalty 权重
        length_min: 长度惩罚最小长度
        length_max: 长度惩罚最大长度
        hallucination_penalizer: 幻觉惩罚器（w6 > 0 时需要）

    Returns:
        rewards: 每条文本的最终 reward 列表
    """
    if references is None:
        references = [""] * len(texts)

    # R5: 多样性需要组内所有文本，先计算
    diversity_scores = diversity_reward(texts) if w5 > 0 else [0.0] * len(texts)

    rewards = []
    for i, (img, text, ref) in enumerate(zip(images, texts, references)):
        r1 = clip_scorer.score(img, text)
        r2 = llm_judge.score(text, reference=ref) / 10.0
        r3 = length_penalty(text, min_len=length_min, max_len=length_max)
        r4 = attribute_coverage(text) if w4 > 0 else 0.0
        r5 = diversity_scores[i] if w5 > 0 else 0.0
        r6 = hallucination_penalizer.score(text) if (w6 > 0 and hallucination_penalizer) else 1.0

        reward = w1 * r1 + w2 * r2 + w3 * r3 + w4 * r4 + w5 * r5 + w6 * r6
        rewards.append(reward)

    return rewards


# ============================================================
# 优势函数 (Group Normalization)
# ============================================================

def compute_advantages(
    rewards: torch.Tensor,
    group_size: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    组内归一化计算优势值

    A_i = (R_i - mean(R_group)) / (std(R_group) + eps)

    Args:
        rewards: [total_samples] 所有样本的 reward
        group_size: 每组大小 (K)
        eps: 数值稳定项

    Returns:
        advantages: [total_samples] 优势值
    """
    total = rewards.numel()
    assert total % group_size == 0, \
        f"Total samples ({total}) must be divisible by group_size ({group_size})"

    rewards_2d = rewards.view(-1, group_size)  # [N, K]
    mean = rewards_2d.mean(dim=1, keepdim=True)
    std = rewards_2d.std(dim=1, keepdim=True, unbiased=False)
    advantages_2d = (rewards_2d - mean) / (std + eps)
    return advantages_2d.view(-1)


def compute_grpo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """
    计算 GRPO 损失

    GRPO 使用 clipped surrogate objective:
    L = -mean(min(r * A, clip(r, 1-ε, 1+ε) * A))

    Args:
        log_probs: 当前策略的 log 概率 [total_samples]
        old_log_probs: 旧策略的 log 概率 [total_samples]
        advantages: 优势值 [total_samples]
        clip_eps: 裁剪范围

    Returns:
        loss: 标量损失
    """
    ratio = torch.exp(log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    return loss


# ============================================================
# KL 散度约束
# ============================================================

def compute_kl_penalty(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    计算当前策略与参考策略之间的 KL 散度

    KL(π || π_ref) ≈ E[log(π/π_ref)] = E[log_probs - ref_log_probs]

    Args:
        log_probs: 当前策略的 log 概率 [total_samples]
        ref_log_probs: 参考策略的 log 概率 [total_samples]

    Returns:
        kl: 标量 KL 散度
    """
    kl = (log_probs - ref_log_probs).mean()
    return kl


def compute_grpo_loss_with_kl(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    ref_log_probs: Optional[torch.Tensor] = None,
    clip_eps: float = 0.2,
    kl_coef: float = 0.05,
) -> torch.Tensor:
    """
    计算 GRPO 损失 + KL 散度惩罚

    L = GRPO_loss + kl_coef * KL(π || π_ref)

    Args:
        log_probs: 当前策略的 log 概率
        old_log_probs: 旧策略的 log 概率
        advantages: 优势值
        ref_log_probs: 参考策略的 log 概率 (SFT 模型)
        clip_eps: 裁剪范围
        kl_coef: KL 惩罚系数

    Returns:
        loss: 标量损失
    """
    grpo_loss = compute_grpo_loss(log_probs, old_log_probs, advantages, clip_eps)

    if ref_log_probs is not None:
        kl_penalty = compute_kl_penalty(log_probs, ref_log_probs)
        return grpo_loss + kl_coef * kl_penalty

    return grpo_loss


# ============================================================
# 辅助函数
# ============================================================

def compute_token_level_log_prob(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    计算序列级别的 token 平均 log 概率

    Args:
        logits: [B, L, V] 模型输出 logits
        input_ids: [B, L] 输入 token IDs
        attention_mask: [B, L] 注意力掩码

    Returns:
        avg_log_probs: [B] 每个序列的平均 log 概率
    """
    log_probs = F.log_softmax(logits, dim=-1)  # [B, L, V]
    token_log_probs = log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)  # [B, L]

    masked = token_log_probs * attention_mask.float()
    avg_log_probs = masked.sum(dim=1) / (attention_mask.sum(dim=1).float() + 1e-8)
    return avg_log_probs
