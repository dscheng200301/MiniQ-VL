"""
MiniQ-VL DPO 训练脚本
==============================
专注图像描述质量提升 - Direct Preference Optimization

使用:
    python trainer/train_dpo.py --data_path ./dataset/minimind-v_dataset/dpo_i2t.json
"""

import os
import sys
import io
import json
import time
import math
import random
import traceback
import argparse
import warnings
import base64
import copy
from datetime import datetime

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from PIL import Image
from tqdm import tqdm

from model.qwen_vl import QwenVLM, QwenVLMConfig

warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ==================== 工具函数 ====================

def is_main_process():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_path(path):
    """将相对路径转换为基于项目根目录的绝对路径"""
    if path is None or os.path.isabs(path):
        return path
    if "/" in path and not path.startswith("./") and not path.startswith("../"):
        return path  # HuggingFace Hub ID
    return os.path.join(get_project_root(), path)


def setup_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) != -1:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank
    return 0


def unwrap_model(model):
    """剥开 DDP 包装"""
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


# ==================== DPO 数据集 ====================

class DPODataset(Dataset):
    """DPO 数据集"""
    
    def __init__(self, json_path, processor, max_length=2048):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.processor = processor
        self.max_length = max_length
        self.pad_token_id = processor.tokenizer.pad_token_id or 0
    
    def __len__(self):
        return len(self.data)
    
    def _load_image(self, image_b64):
        """从 base64 加载图像"""
        try:
            img_bytes = base64.b64decode(image_b64)
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            return img
        except Exception:
            return Image.new('RGB', (224, 224))
    
    def _encode(self, prompt, response, image):
        user_content = [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]
        prompt_messages = [{"role": "user", "content": user_content}]
        full_messages = prompt_messages + [{"role": "assistant", "content": response}]

        prompt_text = self.processor.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
        )
        full_text = self.processor.tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False,
        )
        prompt_inputs = self.processor(
            text=[prompt_text], images=[image], truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        inputs = self.processor(
            text=[full_text], images=[image], padding="max_length",
            truncation=True, max_length=self.max_length, return_tensors="pt",
        )

        prompt_len = min(int(prompt_inputs["attention_mask"].sum()), self.max_length)
        valid_len = int(inputs["attention_mask"].sum())
        response_mask = torch.zeros(self.max_length, dtype=torch.long)
        response_mask[prompt_len:valid_len] = 1
        return inputs, response_mask

    def __getitem__(self, index):
        item = self.data[index]
        prompt = item['prompt']
        image_b64 = item.get('image_bytes', '')
        img = self._load_image(image_b64) if image_b64 else Image.new('RGB', (224, 224))

        chosen_inputs, chosen_response_mask = self._encode(prompt, item['chosen'], img)
        rejected_inputs, rejected_response_mask = self._encode(prompt, item['rejected'], img)

        image_grid_thw = chosen_inputs.get('image_grid_thw')
        if image_grid_thw is not None and image_grid_thw.dim() == 1:
            image_grid_thw = image_grid_thw.unsqueeze(0)

        mm_token_type_ids = chosen_inputs.get('mm_token_type_ids')
        if mm_token_type_ids is None:
            image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            mm_token_type_ids = (chosen_inputs['input_ids'] == image_token_id).long()

        return {
            'prompt': prompt,
            'chosen_ids': chosen_inputs['input_ids'].squeeze(0),
            'chosen_mask': chosen_inputs['attention_mask'].squeeze(0),
            'chosen_response_mask': chosen_response_mask,
            'rejected_ids': rejected_inputs['input_ids'].squeeze(0),
            'rejected_mask': rejected_inputs['attention_mask'].squeeze(0),
            'rejected_response_mask': rejected_response_mask,
            'pixel_values': chosen_inputs.get('pixel_values'),
            'image_grid_thw': image_grid_thw,
            'mm_token_type_ids': mm_token_type_ids,
        }


def dpo_collate_fn(batch):
    """DPO collate 函数"""
    # 收集所有图像 tensor，过滤掉 None
    pixel_list = [b['pixel_values'] for b in batch if b['pixel_values'] is not None]
    grid_list = [b['image_grid_thw'] for b in batch if b['image_grid_thw'] is not None]
    mm_list = [b['mm_token_type_ids'] for b in batch if b['mm_token_type_ids'] is not None]
    
    # 拼接成全批次 tensor（每个样本 shape [1, ...]，cat 后为 [batch, ...]）
    pixel_values = torch.cat(pixel_list, dim=0) if pixel_list else None
    image_grid_thw = torch.cat(grid_list, dim=0) if grid_list else None
    mm_token_type_ids = torch.cat(mm_list, dim=0) if mm_list else None
    
    return {
        'prompt': [b['prompt'] for b in batch],
        'chosen_ids': torch.stack([b['chosen_ids'] for b in batch]),
        'chosen_mask': torch.stack([b['chosen_mask'] for b in batch]),
        'chosen_response_mask': torch.stack([b['chosen_response_mask'] for b in batch]),
        'rejected_ids': torch.stack([b['rejected_ids'] for b in batch]),
        'rejected_mask': torch.stack([b['rejected_mask'] for b in batch]),
        'rejected_response_mask': torch.stack([b['rejected_response_mask'] for b in batch]),
        'pixel_values': pixel_values,
        'image_grid_thw': image_grid_thw,
        'mm_token_type_ids': mm_token_type_ids,
    }


# ==================== DPO 损失函数 ====================

def compute_dpo_loss(
    model,
    ref_model,
    chosen_ids, chosen_mask,
    rejected_ids, rejected_mask,
    chosen_response_mask, rejected_response_mask,
    pixel_values=None,
    image_grid_thw=None,
    mm_token_type_ids=None,
    beta=0.1
):
    """计算 DPO 损失"""
    # 构建模型输入参数
    model_kwargs = {'use_cache': False}
    
    if pixel_values is not None:
        model_kwargs['pixel_values'] = pixel_values
    if image_grid_thw is not None:
        model_kwargs['image_grid_thw'] = image_grid_thw
    if mm_token_type_ids is not None:
        model_kwargs['mm_token_type_ids'] = mm_token_type_ids
    
    # 计算 log probabilities
    def get_logps(target_model, ids, mask, response_mask):
        model_kwargs['input_ids'] = ids
        model_kwargs['attention_mask'] = mask
        outputs = target_model(**model_kwargs)
        logits = outputs.logits  # [batch, seq_len, vocab]
        
        # 计算 log probs
        log_probs = F.log_softmax(logits, dim=-1)  # [batch, seq_len, vocab]
        
        # 移位：预测下一个 token
        target_ids = ids[:, 1:]  # [batch, seq_len-1]
        target_mask = response_mask[:, 1:].to(log_probs.dtype)
        
        # gather 操作
        batch_size, seq_len_minus_1, vocab_size = log_probs.shape
        
        # 确保 target_ids 在范围内
        target_ids_clamped = torch.clamp(target_ids, 0, vocab_size - 1)
        
        # gather
        target_log_probs = torch.gather(
            log_probs[:, :-1, :],  # [batch, seq_len-1, vocab]
            dim=2,
            index=target_ids_clamped.unsqueeze(2)  # [batch, seq_len-1, 1]
        ).squeeze(2)  # [batch, seq_len-1]
        
        # 应用 mask
        target_log_probs = target_log_probs * target_mask
        
        return target_log_probs.sum(dim=1)

    chosen_logps = get_logps(model, chosen_ids, chosen_mask, chosen_response_mask)
    rejected_logps = get_logps(model, rejected_ids, rejected_mask, rejected_response_mask)

    with torch.no_grad():
        if ref_model is not None:
            ref_chosen_logps = get_logps(
                ref_model, chosen_ids, chosen_mask, chosen_response_mask
            )
            ref_rejected_logps = get_logps(
                ref_model, rejected_ids, rejected_mask, rejected_response_mask
            )
        else:
            ref_chosen_logps = torch.zeros_like(chosen_logps)
            ref_rejected_logps = torch.zeros_like(rejected_logps)

    policy_margin = chosen_logps - rejected_logps
    reference_margin = ref_chosen_logps - ref_rejected_logps
    loss = -F.logsigmoid(beta * (policy_margin - reference_margin)).mean()

    return loss, chosen_logps.mean(), rejected_logps.mean()


# ==================== 训练函数 ====================

def train(args):
    """DPO 训练"""
    # 分布式
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(args.seed)
    
    device = args.device
    
    # 目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(get_project_root(), args.save_dir, f"dpo_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存配置
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # A merged SFT directory is a complete Hugging Face model and should be
    # loaded directly. Loading its inner state dict into the wrapper would
    # silently miss every key because of the wrapper's "model." prefix.
    sft_path = resolve_path(args.sft_checkpoint)
    merged_sft_dir = (
        sft_path if sft_path and os.path.isdir(sft_path)
        and os.path.exists(os.path.join(sft_path, "config.json"))
        else None
    )
    model_path = merged_sft_dir or resolve_path(args.model_path)
    config = QwenVLMConfig(model_path=model_path)
    model = QwenVLM(config).to(device)
    
    # 加载 SFT 权重
    if merged_sft_dir:
        print(f"[Rank {local_rank}] Loaded merged SFT model from {merged_sft_dir}")
    elif sft_path and os.path.exists(sft_path):
        try:
            if os.path.isdir(sft_path):
                model_file = os.path.join(sft_path, "model.safetensors")
                if os.path.exists(model_file):
                    from safetensors.torch import load_file
                    state_dict = load_file(model_file)
                    unwrap_model(model).load_state_dict(state_dict, strict=False)
                else:
                    model_file = os.path.join(sft_path, "pytorch_model.bin")
                    if os.path.exists(model_file):
                        state_dict = torch.load(model_file, map_location='cpu')
                        unwrap_model(model).load_state_dict(state_dict, strict=False)
            else:
                state_dict = torch.load(sft_path, map_location='cpu')
                if 'model' in state_dict:
                    unwrap_model(model).load_state_dict(state_dict['model'], strict=False)
                else:
                    unwrap_model(model).load_state_dict(state_dict, strict=False)
            print(f"[Rank {local_rank}] Loaded SFT checkpoint from {sft_path}")
        except Exception as e:
            print(f"[Rank {local_rank}] Failed to load SFT: {e}")
    
    # 冻结策略
    if args.freeze_vision:
        for name, param in unwrap_model(model).named_parameters():
            if 'visual' in name or 'vit' in name or 'vision' in name:
                param.requires_grad = False
    
    if args.freeze_language:
        for name, param in unwrap_model(model).named_parameters():
            if 'visual' not in name and 'vit' not in name and 'vision' not in name:
                param.requires_grad = False

    ref_model = None
    if args.use_ref_model:
        ref_model = copy.deepcopy(model).eval()
        for param in ref_model.parameters():
            param.requires_grad = False

    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # 数据集
    data_path = resolve_path(args.data_path)
    dataset = DPODataset(data_path, unwrap_model(model).processor, max_length=args.max_seq_len)
    
    if args.max_samples > 0 and args.max_samples < len(dataset):
        dataset = torch.utils.data.Subset(dataset, range(args.max_samples))
    
    sampler = DistributedSampler(dataset) if dist.is_initialized() else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=dpo_collate_fn,
        num_workers=args.num_workers,
        shuffle=(sampler is None)
    )
    
    # 优化器
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, unwrap_model(model).parameters()),
        lr=args.learning_rate,
        weight_decay=0.01
    )
    
    # 学习率调度器：余弦退火
    num_training_steps = len(dataloader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_training_steps,
        eta_min=args.learning_rate * 0.1  # 最大学习率的 10%
    )
    
    # wandb
    wandb_run = None
    if args.use_wandb and is_main_process():
        try:
            import wandb
            wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
            wandb_run = wandb
        except Exception as e:
            print(f"Wandb init failed: {e}")
    
    # 训练循环
    model.train()
    global_step = 0
    
    for epoch in range(args.epochs):
        if sampler:
            sampler.set_epoch(epoch)
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}") if is_main_process() else dataloader
        
        for batch in pbar:
            chosen_ids = batch['chosen_ids'].to(device)
            chosen_mask = batch['chosen_mask'].to(device)
            rejected_ids = batch['rejected_ids'].to(device)
            rejected_mask = batch['rejected_mask'].to(device)
            chosen_response_mask = batch['chosen_response_mask'].to(device)
            rejected_response_mask = batch['rejected_response_mask'].to(device)
            
            # 处理图像相关参数（已经是 batched tensor）
            pixel_values = batch.get('pixel_values')
            image_grid_thw = batch.get('image_grid_thw')
            mm_token_type_ids = batch.get('mm_token_type_ids')
            
            if pixel_values is not None:
                pixel_values = pixel_values.to(device)
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(device)
            if mm_token_type_ids is not None:
                mm_token_type_ids = mm_token_type_ids.to(device)
            
            optimizer.zero_grad()
            
            loss, chosen_logps, rejected_logps = compute_dpo_loss(
                unwrap_model(model),
                ref_model,
                chosen_ids, chosen_mask,
                rejected_ids, rejected_mask,
                chosen_response_mask, rejected_response_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                beta=args.beta
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()  # 更新学习率
            
            if is_main_process() and global_step % args.log_interval == 0:
                lr = get_lr(optimizer)
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'chosen': f'{chosen_logps.item():.2f}',
                    'rejected': f'{rejected_logps.item():.2f}'
                })
                
                if wandb_run:
                    wandb_run.log({
                        'loss': loss.item(),
                        'chosen_logps': chosen_logps.item(),
                        'rejected_logps': rejected_logps.item(),
                        'logps_diff': chosen_logps.item() - rejected_logps.item(),
                        'lr': lr,
                        'step': global_step
                    })
            
            # 保存（只保存模型权重，不保存 optimizer，减小磁盘占用）
            if is_main_process() and global_step > 0 and global_step % args.save_interval == 0:
                save_path = os.path.join(save_dir, f"dpo_step{global_step}.pt")
                torch.save({
                    'model': unwrap_model(model).state_dict(),
                    'step': global_step
                }, save_path)
                print(f"Saved: {save_path}")

                # 只保留最近 2 个 checkpoint，删除旧的
                checkpoint_files = sorted(
                    [f for f in os.listdir(save_dir) if f.startswith('dpo_step') and f.endswith('.pt')],
                    key=lambda x: int(x.replace('dpo_step', '').replace('.pt', ''))
                )
                while len(checkpoint_files) > 2:
                    old_file = checkpoint_files.pop(0)
                    old_path = os.path.join(save_dir, old_file)
                    os.remove(old_path)
                    print(f"Removed old checkpoint: {old_file}")

            global_step += 1

    # 最终保存（只存模型权重）
    if is_main_process():
        final_path = os.path.join(save_dir, "dpo_final.pt")
        torch.save({
            'model': unwrap_model(model).state_dict(),
        }, final_path)
        print(f"Training complete! Saved to {final_path}")
    
    if wandb_run:
        wandb_run.finish()
    
    if dist.is_initialized():
        dist.destroy_process_group()


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description="MiniQ-VL DPO Training")
    
    # 模型
    parser.add_argument("--model_path", type=str, 
                       default="./model/Qwen3-VL-2B-Instruct")
    parser.add_argument("--sft_checkpoint", type=str,
                       default="./out/sft_vlm_merged")
    parser.add_argument("--save_dir", type=str, default="./out")
    
    # 训练
    parser.add_argument("--data_path", type=str,
                       default="./dataset/minimind-v_dataset/dpo_i2t.json")
    parser.add_argument("--max_samples", type=int, default=0, help="0=使用全部样本")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4, help="根据显存调整")
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--use_ref_model", type=int, default=1, choices=[0, 1],
                       help="Use a frozen reference model for standard DPO")
    parser.add_argument("--num_workers", type=int, default=0)
    
    # 冻结
    parser.add_argument("--freeze_vision", type=int, default=1)
    parser.add_argument("--freeze_language", type=int, default=0)
    
    # 日志
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=100)
    
    # wandb
    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--wandb_project", type=str, default="MiniQ-VL-DPO")
    parser.add_argument("--wandb_entity", type=str, default=None)
    
    # 其他
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    try:
        train(args)
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"Training failed: {e}")
        print(f"{'='*60}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
