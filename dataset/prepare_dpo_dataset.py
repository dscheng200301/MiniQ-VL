"""
Construct DPO dataset
Sample from SFT data and use LLM API to generate rejected answers
Optimized: streaming read to avoid OOM, rate limiting: RPM < 500, TPM < 20,000,000
"""
import sys
import os
__package__ = "dataset"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import random
import gc
import time
from collections import deque
from tqdm import tqdm
from typing import List, Dict, Optional, Iterator
import pyarrow.parquet as pq

from utils.api_client import APIClient


# ============== Config ==============
SFT_PARQUET_PATH = "/root/autodl-tmp/MiniQ-VL/dataset/minimind-v_dataset/sft_i2t.parquet"
OUTPUT_JSON_PATH = "/root/autodl-tmp/MiniQ-VL/dataset/minimind-v_dataset/dpo_i2t.json"
NUM_SAMPLES = 3000

# 统一的图像描述 Prompt（用于构建专注的图像描述模型）
IMAGE_DESCRIPTION_PROMPTS = [
    "请详细描述这张图片",
    "请描述这张图片的内容",
    "描述一下这张图片",
    "详细描述图片中的场景和物体",
    "请对这张图片进行详细描述",
]

# API Config
API_KEY = os.environ.get("API_KEY", "")
BASE_URL = "https://api.minimaxi.com/v1"
MODEL = "MiniMax-M2.7-highspeed"

# Rate limit config (RPM < 500, TPM < 20,000,000)
MAX_RPM = 450  # 批量后请求数减少，可提高
MAX_TPM = 18000000  # 批量后每次token增多
WINDOW_SECONDS = 60


class RateLimiter:
    """Rate limiter: control RPM and TPM"""
    
    def __init__(self, max_rpm: int, max_tpm: int, window_seconds: int = 60):
        self.max_rpm = max_rpm
        self.max_tpm = max_tpm
        self.window_seconds = window_seconds
        
        self.timestamps = deque()
        self.token_counts = deque()
        
        self.estimated_tokens_per_request = 200
    
    def _clean_old_records(self, current_time: float):
        """Remove expired records"""
        cutoff_time = current_time - self.window_seconds
        while self.timestamps and self.timestamps[0] < cutoff_time:
            self.timestamps.popleft()
            self.token_counts.popleft()
    
    def _get_current_counts(self) -> tuple:
        """Get current window request count and token count"""
        self._clean_old_records(time.time())
        return len(self.timestamps), sum(self.token_counts)
    
    def wait_if_needed(self, tokens_used: int = None):
        """
        If rate limit exceeded, wait until can send request
        tokens_used: tokens consumed in this request
        """
        if tokens_used is None:
            tokens_used = self.estimated_tokens_per_request
        
        while True:
            current_time = time.time()
            self._clean_old_records(current_time)
            
            rpm_count, tpm_count = self._get_current_counts()
            
            wait_time_rpm = 0
            wait_time_tpm = 0
            
            if rpm_count >= self.max_rpm:
                oldest = self.timestamps[0]
                wait_time_rpm = (oldest + self.window_seconds) - current_time
            
            if tpm_count + tokens_used > self.max_tpm:
                wait_time_tpm = 1.0
            
            wait_time = max(wait_time_rpm, wait_time_tpm, 0)
            
            if wait_time <= 0:
                break
            
            time.sleep(min(wait_time, 1.0))
    
    def record(self, tokens_used: int = None):
        """Record a request"""
        if tokens_used is None:
            tokens_used = self.estimated_tokens_per_request
        
        current_time = time.time()
        self.timestamps.append(current_time)
        self.token_counts.append(tokens_used)


REJECTED_SYSTEM_PROMPT = """你是一个图像描述助手。请为图像描述生成"中等质量"的回答。

中等质量回答的特征（与高质量回答有明显区别）：
- 描述了主要内容但不够详细具体
- 遗漏了部分重要细节（如颜色、材质、数量等）
- 缺乏对场景氛围和上下文的描述
- 描述准确但缺乏深度和完整性
- 可能遗漏了次要但重要的物体

注意：
- 不要生成过于简短或敷衍的回答（如"这是一张图"）
- 不要生成错误的回答
- 要生成"还行但不够好"的回答

输出格式：每行只输出一个中等质量回答（50字以内），不要编号，不要解释。"""

REJECTED_USER_TEMPLATE = """请为图像描述生成{count}个中等质量的回答：

要求：
- 不要过于简短
- 不要太敷衍
- 回答是合理的，但相比详细描述有所欠缺

示例：
- "图片中有人和背景"
- "图中有一个物体"
- "场景比较简单"
- "有些模糊的物体"

每行只输出一个回答，不要其他内容："""


def stream_parquet_samples(parquet_path: str, batch_size: int = 10000) -> Iterator[Dict]:
    """Stream read parquet file, yield samples batch by batch"""
    pf = pq.ParquetFile(parquet_path)
    total_rows = pf.metadata.num_rows
    print(f"Total rows in parquet: {total_rows}")
    
    for batch in pf.iter_batches(batch_size=batch_size, columns=['conversations', 'image_bytes']):
        table = batch.to_pandas()
        for _, row in table.iterrows():
            yield {
                'conversations': row['conversations'],
                'image_bytes': row['image_bytes'],
            }
        del table, batch
        gc.collect()


# 图像描述类关键词（与 GRPO 保持一致）
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


def is_image_description_prompt(prompt: str) -> bool:
    """
    判断 prompt 是否是图像描述类（使用关键词匹配，与 GRPO 一致）
    """
    if not prompt:
        return False
    
    # 精确匹配关键词
    return any(kw in prompt for kw in FALLBACK_KEYWORDS)


def extract_prompt_chosen(conversations_json: str) -> tuple:
    """Extract prompt and chosen answer from conversations"""
    try:
        conversations = json.loads(conversations_json)
    except:
        return None, None
    
    user_content = ""
    for conv in conversations:
        if conv.get('role') == 'user':
            content = conv.get('content', '')
            content = content.replace('<image>', '').strip()
            user_content = content
            break
    
    assistant_content = ""
    for conv in conversations:
        if conv.get('role') == 'assistant':
            assistant_content = conv.get('content', '')
            break
    
    if not user_content or not assistant_content:
        return None, None
    
    return user_content, assistant_content


def generate_rejected_batch(client: APIClient, count: int, rate_limiter: RateLimiter) -> List[Optional[str]]:
    """Generate rejected (medium quality) answers for image description task"""
    rate_limiter.wait_if_needed(tokens_used=80 * count)
    
    messages = [
        {"role": "system", "content": REJECTED_SYSTEM_PROMPT},
        {"role": "user", "content": REJECTED_USER_TEMPLATE.format(count=count)}
    ]
    
    try:
        response = client.chat(
            messages,
            max_tokens=80 * count,  # 中等质量回答稍长
            temperature=0.9,
        )
        rate_limiter.record(tokens_used=80 * count)
        
        # Parse response - one answer per line
        lines = response.strip().split('\n')
        results = []
        for line in lines:
            line = line.strip().lstrip('0123456789.-) ').strip()
            if line and len(line) >= 5:  # 过滤太短的
                results.append(line)
        
        # Pad if needed
        while len(results) < count:
            results.append(results[-1] if results else "图片中有一些物体")
        
        return results[:count]
    except Exception as e:
        print(f"\nBatch Error: {e}")
        return ["图片中有一些物体"] * count


def main():
    print(f"Starting DPO dataset preparation...")
    print(f"Target samples: {NUM_SAMPLES}")
    print(f"Rate limit: RPM < {MAX_RPM}, TPM < {MAX_TPM}")
    
    client = APIClient(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL,
    )
    
    rate_limiter = RateLimiter(MAX_RPM, MAX_TPM, WINDOW_SECONDS)
    
    # ========== 阶段1: 筛选图像描述类数据 ==========
    print("\n[Stage 1] Filtering image description prompts...")
    filtered_data = []
    total_seen = 0
    filtered_count = 0
    
    for conv_json in tqdm(stream_parquet_samples(SFT_PARQUET_PATH, batch_size=50000), desc="Filtering"):
        total_seen += 1
        
        original_prompt, chosen = extract_prompt_chosen(conv_json['conversations'])
        image_bytes = conv_json.get('image_bytes')
        
        if not chosen or len(chosen) <= 10 or not image_bytes:
            continue
        
        # 筛选图像描述类 prompt
        if is_image_description_prompt(original_prompt):
            filtered_data.append({
                'original_prompt': original_prompt,
                'chosen': chosen,
                'image_bytes': image_bytes,
            })
            filtered_count += 1
        
        if total_seen % 100000 == 0:
            print(f"  Processed: {total_seen}, Filtered: {filtered_count}")
    
    print(f"\n[Stage 1] Completed!")
    print(f"  Total processed: {total_seen}")
    print(f"  Filtered image description samples: {filtered_count}")
    
    if len(filtered_data) == 0:
        print("No image description samples found!")
        return
    
    # ========== 阶段2: 随机选取 NUM_SAMPLES 条 ==========
    print(f"\n[Stage 2] Random sampling {NUM_SAMPLES} from {len(filtered_data)} filtered samples...")
    
    random.seed(42)
    if len(filtered_data) <= NUM_SAMPLES:
        sampled_data = filtered_data
        print(f"  Using all {len(filtered_data)} samples (less than target)")
    else:
        sampled_data = random.sample(filtered_data, NUM_SAMPLES)
        print(f"  Sampled {NUM_SAMPLES} samples")
    
    # 替换为统一的图像描述 prompt
    for sample in sampled_data:
        sample['prompt'] = random.choice(IMAGE_DESCRIPTION_PROMPTS)
        del sample['original_prompt']  # 删除原始 prompt
    
    print(f"  Valid samples collected: {len(sampled_data)}")
    
    # 释放内存
    del filtered_data
    gc.collect()
    
    print("\nGenerating rejected answers (batch mode with rate limiting)...")
    print("Using unified image description prompts for image description specialist model...")
    dpo_data = []
    BATCH_SIZE = 20  # 每次批量处理的问题数
    
    for batch_start in tqdm(range(0, len(sampled_data), BATCH_SIZE)):
        batch = sampled_data[batch_start:batch_start + BATCH_SIZE]
        
        # 统一使用图像描述任务生成 rejected answers
        rejected_list = generate_rejected_batch(client, len(batch), rate_limiter)
        
        for i, sample in enumerate(batch):
            rejected = rejected_list[i] if i < len(rejected_list) else None
            if rejected:
                # 将 bytes 转换为 base64 字符串以便 JSON 序列化
                image_bytes = sample.get('image_bytes')
                if image_bytes and isinstance(image_bytes, bytes):
                    import base64
                    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                else:
                    image_b64 = image_bytes
                
                dpo_data.append({
                    'prompt': sample['prompt'],
                    'chosen': sample['chosen'],
                    'rejected': rejected,
                    'image_bytes': image_b64,
                })
        
        # 每批次保存
        if len(dpo_data) % 500 == 0:
            with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
                json.dump(dpo_data, f, ensure_ascii=False, indent=2)
            gc.collect()
    
    print(f"\nSaving {len(dpo_data)} DPO samples to {OUTPUT_JSON_PATH}")
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(dpo_data, f, ensure_ascii=False, indent=2)
    
    if dpo_data:
        print("\n" + "=" * 50)
        print("Sample DPO pair:")
        sample = dpo_data[0]
        print(f"Prompt: {sample['prompt'][:80]}...")
        print(f"Chosen: {sample['chosen'][:100]}...")
        print(f"Rejected: {sample['rejected'][:50]}...")
        print("=" * 50)
    
    print("\nDone!")


if __name__ == "__main__":
    main()