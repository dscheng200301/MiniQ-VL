"""
MiniQ-VL GRPO 数据集模块
从 SFT 数据中筛选视觉描述类样本，用于 GRPO 训练

设计原则:
1. __getitem__ 只返回原始数据 (image_bytes + prompt文本 + 参考回答)
2. tokenize/processor 推迟到训练脚本按需执行，避免重复计算
3. 支持 LLM API 智能筛选（批量分类 + 本地缓存），关键词匹配作为兜底
4. 保留 SFT 中的 assistant 参考回答，供 Reward 和 KL 约束使用
"""
import sys
import os
__package__ = "dataset"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import hashlib
from typing import List, Optional
from torch.utils.data import Dataset
import pyarrow as pa
import pyarrow.parquet as pq

from utils.api_client import APIClient, _DEFAULT_MODEL, _DEFAULT_API_KEY, _DEFAULT_BASE_URL


# 兜底关键词匹配（无 API key 时使用）
FALLBACK_KEYWORDS = [
    "请描述这张图",
    "描述这张图片",
    "描述一下这张图",
    "请描述图片",
    "描述这张图",
    "请详细描述",
    "描述图片",
    "描述一下图片",
    "看看图里有什么",
    "图上是什么",
    "描述一下画面",
    "描述这幅图",
    "描述这个画面",
    "图片内容",
    "图里有什么",
    "介绍一下这张图",
]

# LLM 批量分类的 prompt
CLASSIFY_SYSTEM_PROMPT = """你是一个数据分类助手。你的任务是判断用户消息是否在请求对图像进行视觉描述。

视觉描述请求的判断标准：
- 用户希望 AI 描述、介绍、说明图像中的内容
- 包括但不限于：描述图片、说明画面、介绍图里有什么、看图说话等
- 不包括：针对图像的具体问答（如"图里有几只猫"、"这是什么颜色"）、OCR、代码理解等

请对每条消息输出 JSON 数组，每项为 true（是视觉描述请求）或 false（不是）。
只输出 JSON 数组，不要输出其他内容。"""

CLASSIFY_USER_TEMPLATE = """请判断以下 {count} 条用户消息是否属于"视觉描述请求"：

{messages}

请输出 {count} 个布尔值的 JSON 数组（true/false）："""


class LLMFilter:
    """使用 LLM API 批量判断样本是否为视觉描述请求，结果本地缓存"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = _DEFAULT_MODEL,
        cache_path: Optional[str] = None,
        batch_size: int = 20,
    ):
        self.client = APIClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
        self.batch_size = batch_size
        self.cache_path = cache_path
        self.cache = {}

        # 加载缓存
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                self.cache = json.load(f)
            print(f"LLMFilter: loaded {len(self.cache)} cached results from {cache_path}")

    def _make_key(self, content: str) -> str:
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def _save_cache(self):
        if self.cache_path:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False)

    def filter_batch(self, contents: List[str]) -> List[bool]:
        """
        批量判断消息是否为视觉描述请求

        优先使用缓存，未命中的批量调 LLM API

        Args:
            contents: 用户消息列表

        Returns:
            布尔列表，True 表示是视觉描述请求
        """
        results = [None] * len(contents)
        uncached_indices = []

        # 1. 查缓存
        for i, content in enumerate(contents):
            key = self._make_key(content)
            if key in self.cache:
                results[i] = self.cache[key]
            else:
                uncached_indices.append(i)

        if not uncached_indices:
            return results

        # 2. 未命中的分批调 API
        if not self.client.api_key:
            print("LLMFilter: no API key, using keyword fallback for uncached samples")
            for i in uncached_indices:
                results[i] = any(kw in contents[i] for kw in FALLBACK_KEYWORDS)
            return results

        for batch_start in range(0, len(uncached_indices), self.batch_size):
            batch_indices = uncached_indices[batch_start:batch_start + self.batch_size]
            batch_contents = [contents[i] for i in batch_indices]

            messages_str = "\n".join(f"{j+1}. {c}" for j, c in enumerate(batch_contents))
            user_prompt = CLASSIFY_USER_TEMPLATE.format(
                count=len(batch_contents), messages=messages_str
            )

            try:
                text = self.client.chat(
                    [
                        {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=200,
                    temperature=0.0,
                )
                # 解析 JSON 数组
                parsed = json.loads(text)
                if not isinstance(parsed, list) or len(parsed) != len(batch_contents):
                    raise ValueError(f"Unexpected response format: {text[:100]}")

                for j, idx in enumerate(batch_indices):
                    results[idx] = bool(parsed[j])
                    self.cache[self._make_key(contents[idx])] = bool(parsed[j])

            except Exception as e:
                print(f"LLMFilter: API call failed ({e}), using keyword fallback for this batch")
                for idx in batch_indices:
                    results[idx] = any(kw in contents[idx] for kw in FALLBACK_KEYWORDS)

        # 3. 保存缓存
        self._save_cache()
        return results


class GRPODataset(Dataset):
    """
    GRPO 数据集 — 轻量版

    只返回原始数据，不做 tokenize：
    - image_bytes: 原始图像二进制（避免多进程序列化 PIL Image）
    - prompt_text: 统一的 GRPO prompt 文本
    - reference: SFT 中 assistant 的参考回答（供 Reward/KL 使用）

    筛选策略（按优先级）：
    1. LLM API 智能筛选：批量判断 + 本地缓存，准确覆盖各种表述
    2. 关键词匹配兜底：无 API key 时自动降级
    """

    GRPO_PROMPT = "请详细描述这张图片。"

    def __init__(
        self,
        parquet_path: str,
        filter_mode: str = "auto",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        llm_model: str = _DEFAULT_MODEL,
        cache_path: Optional[str] = None,
        filter_keywords: Optional[List[str]] = None,
        prefiltered_path: Optional[str] = None,
    ):
        """
        Args:
            parquet_path: 源 parquet 数据文件路径
            filter_mode: 筛选模式
                - "auto": 有 API key 用 LLM，否则关键词兜底
                - "llm": 强制使用 LLM API
                - "keyword": 仅使用关键词匹配
            api_key: OpenAI API key（默认从环境变量读取）
            base_url: API base URL（默认从环境变量读取）
            llm_model: LLM 模型名（默认 deepseek-v4-flash, 见 utils/api_client.py）
            cache_path: 缓存文件路径（默认与 parquet 同目录）
            filter_keywords: 关键词匹配时的关键词列表
            prefiltered_path: 预筛选数据集路径（通常由 dataset/prepare_grpo_dataset.py 生成）
                优先级最高: 给定且文件存在 → 直接加载, 跳过筛选
        """
        super().__init__()

        # 优先级 1: 加载预筛选数据集 (推荐, 节省时间)
        if prefiltered_path and os.path.exists(prefiltered_path):
            print(f"GRPO Dataset: loading prefiltered dataset from {prefiltered_path}")
            self.table = pa.Table.from_batches(pq.ParquetFile(prefiltered_path).iter_batches())
            self.filtered_indices = list(range(len(self.table)))
            print(f"GRPO Dataset (prefiltered): {len(self.filtered_indices)} samples loaded")
            return

        # 优先级 2: 从源 parquet 加载并在线筛选
        # 加载数据
        self.table = pa.Table.from_batches(pq.ParquetFile(parquet_path).iter_batches())

        # 默认缓存路径
        if cache_path is None:
            cache_path = parquet_path.replace(".parquet", "_grpo_filter_cache.json")

        # 筛选样本
        if filter_mode == "keyword":
            keywords = filter_keywords or FALLBACK_KEYWORDS
            self.filtered_indices = self._filter_by_keywords(self.table, keywords)
            print(f"GRPO Dataset (keyword): {len(self.filtered_indices)} / {len(self.table)} samples matched")
        else:
            # auto 或 llm 模式
            effective_api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
            if filter_mode == "llm" and not effective_api_key:
                raise ValueError("filter_mode='llm' but no OPENAI_API_KEY configured")

            if filter_mode == "auto" and not effective_api_key:
                print("GRPO Dataset: no API key, falling back to keyword matching")
                keywords = filter_keywords or FALLBACK_KEYWORDS
                self.filtered_indices = self._filter_by_keywords(self.table, keywords)
                print(f"GRPO Dataset (keyword fallback): {len(self.filtered_indices)} / {len(self.table)} samples matched")
            else:
                llm_filter = LLMFilter(
                    api_key=api_key, base_url=base_url,
                    model=llm_model, cache_path=cache_path,
                )
                self.filtered_indices = self._filter_by_llm(self.table, llm_filter)
                print(f"GRPO Dataset (LLM): {len(self.filtered_indices)} / {len(self.table)} samples matched")

    def _filter_by_keywords(self, table, keywords: List[str]) -> List[int]:
        """关键词匹配筛选（兜底方案）"""
        indices = []
        for i in range(len(table)):
            try:
                conversations = json.loads(table['conversations'][i].as_py())
                if not conversations or len(conversations) == 0:
                    continue
                first_msg = conversations[0]
                if first_msg.get('role') != 'user':
                    continue
                content = first_msg.get('content', '')
                if any(kw in content for kw in keywords):
                    indices.append(i)
            except Exception:
                continue
        return indices

    def _filter_by_llm(self, table, llm_filter: LLMFilter) -> List[int]:
        """LLM API 批量筛选"""
        # 先提取所有 user 消息
        contents = []
        valid_indices = []
        for i in range(len(table)):
            try:
                conversations = json.loads(table['conversations'][i].as_py())
                if conversations and len(conversations) > 0:
                    first_msg = conversations[0]
                    if first_msg.get('role') == 'user':
                        contents.append(first_msg.get('content', ''))
                        valid_indices.append(i)
            except Exception:
                continue

        # 批量分类
        print(f"LLM filtering {len(contents)} samples (batch_size={llm_filter.batch_size})...")
        flags = llm_filter.filter_batch(contents)

        # 筛选
        return [valid_indices[i] for i, flag in enumerate(flags) if flag]

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, index: int):
        real_idx = self.filtered_indices[index]

        # 原始图像字节
        image_bytes = self.table['image_bytes'][real_idx].as_py()
        if not isinstance(image_bytes, list):
            image_bytes = [image_bytes]

        # 提取 assistant 参考回答
        conversations = json.loads(self.table['conversations'][real_idx].as_py())
        reference = ""
        for turn in conversations:
            if turn.get('role') == 'assistant':
                reference = turn.get('content', '')
                break

        return {
            'image_bytes': image_bytes,
            'prompt_text': self.GRPO_PROMPT,
            'reference': reference,
        }


def grpo_collate_fn(batch):
    """GRPO DataLoader 整理函数 — 直接传递 list"""
    return {
        'image_bytes': [b['image_bytes'] for b in batch],
        'prompt_text': [b['prompt_text'] for b in batch],
        'reference': [b['reference'] for b in batch],
    }