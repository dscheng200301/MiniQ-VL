"""
MiniQ-VL 数据集处理模块
基于 minimind-v 的数据集格式，支持 parquet 格式的图文数据
适配 Qwen3-VL-2B-Instruct:
- SigLIP-2 视觉编码器 (patch size 16x16)
- DeepStack 多层次特征融合
- Interleaved-MRoPE 3D 位置编码
"""
import sys
import os
__package__ = "dataset"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import random
import torch
import io
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pyarrow as pa
import pyarrow.parquet as pq


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def pre_processing_chat(conversations, add_system_ratio=0.2):
    """预处理对话，概率性添加 system prompt"""
    if any(conv.get('tools') for conv in conversations):
        return conversations

    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是 MiniQ，一个基于 Qwen3-VL 的视觉语言助手。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "You are a helpful AI assistant.",
        "You are MiniQ, a visual language assistant based on Qwen3-VL."
    ]
    
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations


def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    """后处理对话，移除空思考标签"""
    if '\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content


class VLMDataset(Dataset):
    """
    VLM 数据集类
    支持从 parquet 文件加载图文对话数据
    
    数据格式:
    - conversations: JSON 格式的多轮对话
    - image_bytes: 图像二进制数据
    """
    
    def __init__(
        self,
        parquet_path: str,
        processor,
        max_length: int = 2048,
        image_token_len: int = 256
    ):
        super().__init__()
        self.table = pa.Table.from_batches(pq.ParquetFile(parquet_path).iter_batches())
        self.processor = processor
        self.max_length = max_length
        self.image_token_len = image_token_len
    
    def __len__(self):
        return len(self.table)
    
    def generate_labels(self, input_ids, text=None):
        """
        生成 labels，仅 assistant 部分计算 Loss

        通过在 input_ids 中搜索 assistant 标记的 token 序列来精确定位。
        Qwen3-VL chat template: ...<|im_start|>assistant\n{回复}<|im_end|>...
        """
        labels = [-100] * len(input_ids)

        try:
            tokenizer = self.processor.tokenizer
            # 编码 assistant 起止标记
            assistant_start_tokens = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
            assistant_end_tokens = tokenizer.encode("<|im_end|>", add_special_tokens=False)

            input_ids_list = input_ids.tolist() if hasattr(input_ids, 'tolist') else list(input_ids)
            n = len(input_ids_list)
            len_start = len(assistant_start_tokens)
            len_end = len(assistant_end_tokens)

            search_pos = 0
            while search_pos < n:
                # 查找 assistant 起始标记
                start_pos = -1
                for i in range(search_pos, n - len_start + 1):
                    if input_ids_list[i:i + len_start] == assistant_start_tokens:
                        start_pos = i + len_start  # 内容从标记之后开始
                        break

                if start_pos == -1:
                    break

                # 查找对应的 <|im_end|>
                end_pos = -1
                for i in range(start_pos, n - len_end + 1):
                    if input_ids_list[i:i + len_end] == assistant_end_tokens:
                        end_pos = i
                        break

                if end_pos == -1:
                    end_pos = n  # 最后一段可能没有结束标记

                # 标记 assistant 回复部分的 token 参与 loss 计算
                for i in range(start_pos, end_pos):
                    if input_ids_list[i] != tokenizer.pad_token_id:
                        labels[i] = input_ids_list[i]

                search_pos = end_pos + len_end if end_pos < n else n

        except Exception:
            # 定位失败，回退到全量计算
            for i, token_id in enumerate(input_ids):
                if token_id != self.processor.tokenizer.pad_token_id:
                    labels[i] = token_id

        return labels
    
    def __getitem__(self, index: int):
        conversations = json.loads(self.table['conversations'][index].as_py())
        image_bytes = self.table['image_bytes'][index].as_py()
        
        if not isinstance(image_bytes, list):
            image_bytes = [image_bytes]
        
        # 预处理对话
        conversations = pre_processing_chat(conversations)
        
        # 解码图像
        images = []
        for img_data in image_bytes:
            try:
                img = Image.open(io.BytesIO(img_data))
                if img.mode in ['RGBA', 'LA']:
                    img = img.convert('RGB')
                images.append(img)
            except Exception as e:
                # 如果图像解析失败，使用空白图像
                images.append(Image.new('RGB', (224, 224)))
        
        # 构建消息格式
        # 图像只添加到第一条 user 消息中，assistant 消息不包含图像
        image_added = False
        messages = []
        for turn in conversations:
            if turn.get('role') == 'system':
                messages.append(turn)
            elif turn.get('role') == 'user' and not image_added:
                # 提取文本中的 <image> token，适配 minimind-v 格式
                text_content = turn['content']
                has_image_token = '<image>' in text_content
                
                # 移除 <image> token（图像已单独添加）
                text_content = text_content.replace('<image>', '').strip()
                
                # 构建消息：图像在文本之前（Qwen3-VL 格式要求）
                content_parts = []
                for img in images:
                    content_parts.append({"type": "image", "image": img})
                if text_content:
                    content_parts.append({"type": "text", "text": text_content})
                
                messages.append({
                    "role": turn['role'],
                    "content": content_parts
                })
                image_added = True
            else:
                # 后续 user/assistant 消息：纯文本
                messages.append({
                    "role": turn['role'],
                    "content": turn['content']
                })
        
        # 使用 processor 处理
        try:
            text = self.processor.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.processor(
                text=text,
                images=images,
                padding='max_length',
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            )
            
            # 保留完整的 2D tensor 格式 [batch=1, seq_len]，不要去 batch 维度
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask
            
            # 处理 labels（仅计算 assistant 部分的 loss）
            labels = self.generate_labels(input_ids[0], text=text)
            labels = torch.tensor(labels, dtype=torch.long)
            pixel_values = inputs.get('pixel_values')
            image_grid_thw = inputs.get('image_grid_thw')
            
            # 获取 mm_token_type_ids（Qwen3-VL 多模态 RoPE 必需）
            # 手动计算 mm_token_type_ids，确保形状和值正确
            # 0 = text token, 1 = image token
            try:
                image_token_id = self.processor.image_token_id
                # input_ids 形状: (1, seq_len)
                # mm_token_type_ids 形状: (1, seq_len)
                mm_token_type_ids = (input_ids == image_token_id).long()
            except (AttributeError, Exception) as e:
                # 备选：使用 processor 返回的 mm_token_type_ids
                mm_token_type_ids = inputs.get('mm_token_type_ids')
                if mm_token_type_ids is not None:
                    if isinstance(mm_token_type_ids, torch.Tensor):
                        if mm_token_type_ids.dim() == 1:
                            mm_token_type_ids = mm_token_type_ids.unsqueeze(0)
                    elif isinstance(mm_token_type_ids, (list, tuple)):
                        mm_token_type_ids = torch.tensor(mm_token_type_ids, dtype=torch.long)
                        if mm_token_type_ids.dim() == 1:
                            mm_token_type_ids = mm_token_type_ids.unsqueeze(0)
            
            # 确保 image_grid_thw 格式正确：应该是 (num_images, 3) 的 2D tensor
            # Qwen3-VL processor 可能返回多种格式
            if image_grid_thw is not None:
                # 如果是嵌套的列表结构，先展平
                if isinstance(image_grid_thw, (list, tuple)):
                    # 尝试展平嵌套结构
                    flat_list = []
                    for item in image_grid_thw:
                        if isinstance(item, (list, tuple)):
                            flat_list.extend(item)
                        else:
                            flat_list.append(item)
                    image_grid_thw = torch.tensor(flat_list, dtype=torch.long)
                
                # 然后确保是 2D tensor (num_images, 3)
                if not hasattr(image_grid_thw, 'dim'):
                    image_grid_thw = torch.tensor(image_grid_thw)
                
                if image_grid_thw.dim() == 1:
                    image_grid_thw = image_grid_thw.unsqueeze(0)
            
        except Exception as e:
            # 如果处理失败，返回默认值（2D tensor 格式）
            print(f"Warning: Failed to process sample {index}: {e}")
            input_ids = torch.zeros(1, self.max_length, dtype=torch.long)
            attention_mask = torch.zeros(1, self.max_length, dtype=torch.long)
            labels = torch.full((1, self.max_length), -100, dtype=torch.long)
            pixel_values = None
            image_grid_thw = None
            mm_token_type_ids = None
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'pixel_values': pixel_values,
            'image_grid_thw': image_grid_thw,
            'mm_token_type_ids': mm_token_type_ids
        }


def vlm_collate_fn(batch):
    """DataLoader 整理函数"""
    input_ids = torch.stack([b['input_ids'] for b in batch])
    attention_mask = torch.stack([b['attention_mask'] for b in batch])
    labels = torch.stack([b['labels'] for b in batch])
    
    # 处理 pixel_values（可能是不同形状）
    pixel_values = [b['pixel_values'] for b in batch]
    image_grid_thw = [b['image_grid_thw'] for b in batch]
    
    # 处理 mm_token_type_ids（Qwen3-VL 多模态 RoPE 必需）
    # 手动计算后，每个样本返回的 mm_token_type_ids 形状为 (1, seq_len)
    # 需要 stack 成 (batch, seq_len)
    mm_token_type_ids_list = []
    for b in batch:
        mm = b['mm_token_type_ids']
        if mm is None:
            mm_token_type_ids_list.append(None)
        elif isinstance(mm, torch.Tensor):
            if mm.dim() == 2:
                # (1, seq_len) -> squeeze 成 (seq_len,) 用于 stack
                mm_token_type_ids_list.append(mm.squeeze(0))
            elif mm.dim() == 1:
                mm_token_type_ids_list.append(mm)
            else:
                mm_token_type_ids_list.append(None)
        else:
            mm_token_type_ids_list.append(None)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'pixel_values': pixel_values,
        'image_grid_thw': image_grid_thw,
        'mm_token_type_ids': mm_token_type_ids_list
    }


# 测试数据读取
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
    
    for path in ['pretrain_i2t.parquet', 'sft_i2t.parquet']:
        if not os.path.exists(path):
            print(f"文件不存在: {path}")
            continue
            
        pf = pq.ParquetFile(path)
        n = pf.num_row_groups
        t = pa.concat_tables([pf.read_row_group(i * n // 5).slice(0, 1) for i in range(5)])
        
        fig, ax = plt.subplots(1, 5, figsize=(20, 4))
        for i in range(5):
            img_data = t['image_bytes'][i].as_py()
            img_data = img_data[0] if isinstance(img_data, list) else img_data
            ax[i].imshow(Image.open(io.BytesIO(img_data)))
            ax[i].axis('off')
            ax[i].set_title(
                json.loads(t['conversations'][i].as_py())[1]['content'][:30],
                fontsize=8
            )
        
        out = path.replace('.parquet', '_preview.png')
        plt.savefig(out)
        print(f'已保存 {out}, 共 {pf.metadata.num_rows} 条')