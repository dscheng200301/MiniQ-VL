"""
MiniQ-VL DPO 数据集（支持图像）
"""
import sys
import os
__package__ = "dataset"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import io
from typing import Optional


class DPODataset(Dataset):
    """
    DPO 数据集（支持图像）
    数据格式: {"prompt": str, "chosen": str, "rejected": str, "image_bytes": bytes}
    """
    
    def __init__(self, json_path: str, processor, max_length: int = 2048):
        super().__init__()
        
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.processor = processor
        self.max_length = max_length
        
        # 特殊 token id
        self.image_token_id = processor.tokenizer.encode("<|image_pad|>", add_special_tokens=False)[0]
    
    def __len__(self):
        return len(self.data)
    
    def _tokenize(self, text: str):
        """Tokenize 文本"""
        messages = [
            {"role": "user", "content": text},
            {"role": "assistant", "content": ""},
        ]
        text = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        return text
    
    def _load_images(self, image_bytes):
        """Load images from bytes"""
        images = []
        if not image_bytes:
            return images
        
        if not isinstance(image_bytes, list):
            image_bytes = [image_bytes]
        
        for img_data in image_bytes:
            if img_data:
                try:
                    if isinstance(img_data, str):
                        # base64 string
                        import base64
                        img_bytes = base64.b64decode(img_data)
                    else:
                        img_bytes = img_data
                    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                    images.append(img)
                except Exception:
                    images.append(Image.new('RGB', (224, 224)))
            else:
                images.append(Image.new('RGB', (224, 224)))
        
        return images
    
    def __getitem__(self, index: int):
        item = self.data[index]
        
        prompt = item['prompt']
        chosen = item['chosen']
        rejected = item['rejected']
        image_bytes = item.get('image_bytes')
        
        # 加载图像
        images = self._load_images(image_bytes)
        
        # 构建完整对话
        chosen_text = self._tokenize(prompt) + chosen + "<|im_end|>"
        rejected_text = self._tokenize(prompt) + rejected + "<|im_end|>"
        
        # 使用 processor 处理（包含图像）
        try:
            chosen_inputs = self.processor(
                text=chosen_text,
                images=images if images else None,
                padding='max_length',
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            )
            
            rejected_inputs = self.processor(
                text=rejected_text,
                images=images if images else None,
                padding='max_length',
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            )
        except Exception as e:
            # fallback: 仅文本
            chosen_inputs = self.processor.tokenizer(
                chosen_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )
            rejected_inputs = self.processor.tokenizer(
                rejected_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )
        
        # 提取图像信息
        pixel_values = chosen_inputs.get('pixel_values')
        image_grid_thw = chosen_inputs.get('image_grid_thw')
        mm_token_type_ids = chosen_inputs.get('mm_token_type_ids')
        
        # 确保 image_grid_thw 格式正确
        if image_grid_thw is not None:
            if isinstance(image_grid_thw, (list, tuple)):
                flat_list = []
                for item in image_grid_thw:
                    if isinstance(item, (list, tuple)):
                        flat_list.extend(item)
                    else:
                        flat_list.append(item)
                image_grid_thw = torch.tensor(flat_list, dtype=torch.long)
            if not hasattr(image_grid_thw, 'dim'):
                image_grid_thw = torch.tensor(image_grid_thw)
            if image_grid_thw.dim() == 1:
                image_grid_thw = image_grid_thw.unsqueeze(0)
        
        # 获取 mm_token_type_ids（Qwen3-VL 多模态 RoPE 必需）
        try:
            if mm_token_type_ids is None:
                input_ids = chosen_inputs['input_ids']
                image_token_id = self.processor.image_token_id
                mm_token_type_ids = (input_ids == image_token_id).long()
        except Exception:
            pass
        
        return {
            'prompt': prompt,
            'chosen_ids': chosen_inputs['input_ids'].squeeze(0),
            'chosen_mask': chosen_inputs['attention_mask'].squeeze(0),
            'rejected_ids': rejected_inputs['input_ids'].squeeze(0),
            'rejected_mask': rejected_inputs['attention_mask'].squeeze(0),
            'pixel_values': pixel_values,
            'image_grid_thw': image_grid_thw,
            'mm_token_type_ids': mm_token_type_ids,
        }


def dpo_collate_fn(batch):
    """
    DPO batch collate 函数
    """
    result = {
        'prompt': [item['prompt'] for item in batch],
        'chosen_ids': torch.stack([item['chosen_ids'] for item in batch]),
        'chosen_mask': torch.stack([item['chosen_mask'] for item in batch]),
        'rejected_ids': torch.stack([item['rejected_ids'] for item in batch]),
        'rejected_mask': torch.stack([item['rejected_mask'] for item in batch]),
    }
    
    # 处理 pixel_values（可能是不同形状）
    pixel_values = [item.get('pixel_values') for item in batch]
    image_grid_thw = [item.get('image_grid_thw') for item in batch]
    mm_token_type_ids = [item.get('mm_token_type_ids') for item in batch]
    
    # 过滤 None
    pixel_values = [pv for pv in pixel_values if pv is not None]
    image_grid_thw = [thw for thw in image_grid_thw if thw is not None]
    mm_token_type_ids = [mt for mt in mm_token_type_ids if mt is not None]
    
    if pixel_values:
        result['pixel_values'] = pixel_values[0] if len(pixel_values) == 1 else pixel_values
    if image_grid_thw:
        result['image_grid_thw'] = image_grid_thw[0] if len(image_grid_thw) == 1 else image_grid_thw
    if mm_token_type_ids:
        result['mm_token_type_ids'] = mm_token_type_ids[0] if len(mm_token_type_ids) == 1 else mm_token_type_ids
    
    return result
