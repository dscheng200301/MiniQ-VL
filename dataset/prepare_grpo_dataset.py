"""
prepare_grpo_dataset.py
=======================
从 SFT 数据中筛选"视觉描述类"样本, 保存为新的 parquet, 供 GRPO 训练复用.

**本脚本只使用关键词匹配** (不调 LLM, 速度快, 不消耗 token).
关键词见 dataset.grpo_dataset.FALLBACK_KEYWORDS.

用法:
    python dataset/prepare_grpo_dataset.py \
        --src ./dataset/minimind-v_dataset/sft_i2t.parquet \
        --dst ./dataset/minimind-v_dataset/grpo_i2t.parquet

    # 自定义关键词 (覆盖默认的 FALLBACK_KEYWORDS)
    python dataset/prepare_grpo_dataset.py \
        --src <...> --dst <...> \
        --keywords "请描述这张图" "描述图片" "看图说话"

输出:
    <dst>.parquet      筛选后的样本 (完整保留原 schema: image_bytes + conversations)
    <dst>_meta.json    筛选元信息: 总数 / 命中数 / keywords / 时间戳
"""

import os
import sys
import json
import argparse
from datetime import datetime

__package__ = "dataset"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pyarrow as pa
import pyarrow.parquet as pq

from dataset.grpo_dataset import FALLBACK_KEYWORDS


def extract_user_messages(table):
    """
    从 parquet 提取每条样本的第一条 user 消息
    Returns:
        contents: list[str]
        valid_indices: list[int]  原始 parquet 中有效行号
    """
    contents, valid_indices = [], []
    n = len(table)
    conv_col = table['conversations']
    for i in range(n):
        try:
            convs = json.loads(conv_col[i].as_py())
            if not convs or len(convs) == 0:
                continue
            first = convs[0]
            if first.get('role') != 'user':
                continue
            content = first.get('content', '')
            if not content:
                continue
            contents.append(content)
            valid_indices.append(i)
        except Exception:
            continue
    print(f"  Extracted {len(contents)} / {n} user messages")
    return contents, valid_indices


def filter_by_keywords(contents, keywords):
    """纯关键词匹配: 命中至少一个关键词即视为"视觉描述请求"."""
    return [any(kw in c for kw in keywords) for c in contents]


def save_filtered_parquet(src_table, valid_indices, flags, dst_path):
    """
    把命中 flag=True 的行写到新 parquet
    """
    keep_indices = [valid_indices[i] for i, f in enumerate(flags) if f]
    if not keep_indices:
        print("  No samples passed filter, nothing to save.")
        return 0

    sub_table = src_table.take(keep_indices)
    pq.write_table(sub_table, dst_path)
    print(f"  Saved {len(keep_indices)} samples to {dst_path}")
    return len(keep_indices)


def main():
    parser = argparse.ArgumentParser(description="用关键词匹配筛选 GRPO 视觉描述数据集")
    parser.add_argument("--src", type=str, required=True,
                        help="SFT 源 parquet (含 image_bytes + conversations)")
    parser.add_argument("--dst", type=str, required=True,
                        help="输出 parquet 路径 (建议 *_i2t.parquet 后缀)")
    parser.add_argument("--keywords", type=str, nargs="+", default=None,
                        help=f"自定义关键词列表 (留空用内置 {len(FALLBACK_KEYWORDS)} 条: 描述这张图, 看图说话 等)")
    args = parser.parse_args()

    if not os.path.exists(args.src):
        print(f"ERROR: src not found: {args.src}")
        sys.exit(1)

    keywords = args.keywords or FALLBACK_KEYWORDS
    meta_path = args_dst_to_meta(args.dst)
    os.makedirs(os.path.dirname(args.dst) or ".", exist_ok=True)

    print(f"=== Prepare GRPO dataset (keyword only) ===")
    print(f"  src:      {args.src}")
    print(f"  dst:      {args.dst}")
    print(f"  keywords: {len(keywords)} 条")
    if len(keywords) <= 12:
        for k in keywords:
            print(f"           - {k!r}")

    # 1. 加载源表
    print("\n[1/4] 加载源 parquet...")
    src_table = pa.Table.from_batches(pq.ParquetFile(args.src).iter_batches())
    total = len(src_table)
    print(f"  Total rows: {total}")

    # 2. 提取 user 消息
    print("\n[2/4] 提取 user 消息...")
    contents, valid_indices = extract_user_messages(src_table)

    # 3. 关键词匹配
    print(f"\n[3/4] 关键词匹配 ({len(keywords)} 条)...")
    flags = filter_by_keywords(contents, keywords)
    n_kept = sum(1 for f in flags if f)
    print(f"  Matched: {n_kept} / {len(flags)} (filtered out {len(flags) - n_kept})")

    # 4. 写新 parquet + meta
    print(f"\n[4/4] 写出新 parquet...")
    n_saved = save_filtered_parquet(src_table, valid_indices, flags, args.dst)

    meta = {
        "src": os.path.abspath(args.src),
        "dst": os.path.abspath(args.dst),
        "mode": "keyword",
        "keywords": list(keywords),
        "n_keywords": len(keywords),
        "total_rows": total,
        "valid_rows": len(valid_indices),
        "matched_rows": n_saved,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  Meta: {meta_path}")

    print("\n=== Done ===")
    print(f"  → 训练时使用: --prefiltered_path {args.dst}")


def args_dst_to_meta(dst):
    base, _ = os.path.splitext(dst)
    return f"{base}_meta.json"


if __name__ == "__main__":
    main()
