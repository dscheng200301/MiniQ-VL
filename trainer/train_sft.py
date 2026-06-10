"""
MiniQ-VL SFT 训练脚本
完全重写以正确处理 Qwen3-VL 的 mm_token_type_ids
"""
import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import io
import time
import warnings
import random
import math
import traceback
import numpy as np
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

from model.qwen_vl import QwenVLM, QwenVLMConfig

warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ===================== 工具函数 =====================

def is_main_process():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def get_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def get_rank():
    return dist.get_rank() if dist.is_initialized() else 0


def find_latest_checkpoint(save_dir):
    """查找最新的 checkpoint 目录"""
    if not os.path.exists(save_dir):
        return None
    ckp_dirs = [d for d in os.listdir(save_dir) if d.startswith("checkpoint-") and os.path.isdir(os.path.join(save_dir, d))]
    if not ckp_dirs:
        return None
    # 按 step 排序，取最大
    ckp_dirs.sort(key=lambda x: int(x.split("-")[-1]))
    return os.path.join(save_dir, ckp_dirs[-1])


def load_training_state(save_dir):
    """加载训练状态"""
    ckp_dir = find_latest_checkpoint(save_dir)
    if ckp_dir is None:
        return None
    ckp_path = os.path.join(ckp_dir, "training_state.pt")
    if not os.path.exists(ckp_path):
        return None
    ckp_data = torch.load(ckp_path, map_location='cpu', weights_only=False)
    return ckp_data, ckp_dir


def cleanup_old_checkpoints(save_dir, keep_last_n=2):
    """只保留最近的 N 个 checkpoint（默认 2 个），删除其余的"""
    import shutil
    if not os.path.exists(save_dir):
        return
    ckp_dirs = [d for d in os.listdir(save_dir)
                if d.startswith("checkpoint-") and os.path.isdir(os.path.join(save_dir, d))]
    if len(ckp_dirs) <= keep_last_n:
        return
    # 按 step 排序（升序）
    ckp_dirs.sort(key=lambda x: int(x.split("-")[-1]))
    # 删除最旧的（保留最后 N 个）
    to_delete = ckp_dirs[:-keep_last_n] if keep_last_n > 0 else ckp_dirs
    for d in to_delete:
        path = os.path.join(save_dir, d)
        try:
            shutil.rmtree(path)
            print(f"Removed old checkpoint: {path}")
        except Exception as e:
            print(f"Failed to remove {path}: {e}")


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def setup_error_logging(args):
    """配置全局异常处理，自动将未捕获异常写入错误日志"""
    error_log_dir = os.path.join(args.save_dir, "error_logs")
    try:
        os.makedirs(error_log_dir, exist_ok=True)
    except OSError as e:
        # 磁盘满等情况下，回退到 /tmp
        print(f"[WARNING] Failed to create error log dir: {e}. Falling back to /tmp")
        error_log_dir = "/tmp/miniq_vl_error_logs"
        try:
            os.makedirs(error_log_dir, exist_ok=True)
        except OSError:
            error_log_dir = None

    def log_excepthook(exc_type, exc_value, exc_tb):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if error_log_dir is None:
            print(f"\n[ERROR] Training failed! Traceback:", flush=True)
            traceback.print_exception(exc_type, exc_value, exc_tb)
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return

        log_path = os.path.join(error_log_dir, f"error_{timestamp}.log")
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write("MiniQ-VL SFT Training Error Log\n")
                f.write(f"Time: {timestamp}\n")
                f.write("=" * 80 + "\n\n")

                f.write("## Training Arguments\n")
                for k, v in vars(args).items():
                    f.write(f"  {k}: {v}\n")

                f.write("\n## Error Traceback\n")
                traceback.print_exception(exc_type, exc_value, exc_tb, file=f)

                f.write("\n## Full Error Message\n")
                f.write(str(exc_value) + "\n")
        except Exception as e:
            print(f"[ERROR] Failed to write error log: {e}", flush=True)

        # 打印到终端
        print(f"\n[ERROR] Training failed! Log saved to: {log_path}", flush=True)
        # 调用原始 excepthook（确保正确的退出码）
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = log_excepthook


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_path(path):
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    if path.startswith("./") or path.startswith("../"):
        return os.path.join(get_project_root(), path)
    return path


def get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) != -1:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank
    return 0


# ===================== 数据集 =====================

class MiniQVLMDataset(Dataset):
    """
    VLM 数据集 - 直接使用 Qwen3-VL processor 处理
    """
    def __init__(self, parquet_path, processor, max_length=2048):
        self.table = pa.Table.from_batches(pq.ParquetFile(parquet_path).iter_batches())
        self.processor = processor
        self.max_length = max_length
        self.image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.pad_token_id = processor.tokenizer.pad_token_id

    def __len__(self):
        return len(self.table)

    def _build_messages(self, conversations, images):
        """构建 messages 格式"""
        messages = []
        image_added = False
        for turn in conversations:
            role = turn.get('role')
            content = turn.get('content', '')

            if role == 'system':
                messages.append({'role': 'system', 'content': content})
            elif role == 'user' and not image_added:
                # 移除 <image> token
                text_content = content.replace('<image>', '').strip()
                content_parts = [{"type": "image", "image": img} for img in images]
                if text_content:
                    content_parts.append({"type": "text", "text": text_content})
                messages.append({'role': 'user', 'content': content_parts})
                image_added = True
            else:
                messages.append({'role': role, 'content': content})
        return messages

    def _make_labels(self, input_ids_1d, text):
        """
        生成 labels，只计算 assistant 部分的 loss
        input_ids_1d: 1D tensor (seq_len,)
        """
        labels = torch.full_like(input_ids_1d, -100)
        tokenizer = self.processor.tokenizer

        assistant_start_str = "<|im_start|>assistant\n"
        im_end_str = "<|im_end|>"

        pos = 0
        text_len = len(text)
        while pos < text_len:
            start_idx = text.find(assistant_start_str, pos)
            if start_idx == -1:
                break

            end_idx = text.find(im_end_str, start_idx)
            if end_idx == -1:
                end_idx = text_len
            else:
                end_idx += len(im_end_str)

            # 编码前后文以确定 token 范围
            prefix_text = text[:start_idx + len(assistant_start_str)]
            full_text = text[:end_idx]
            prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
            full_ids = tokenizer.encode(full_text, add_special_tokens=False)

            start_token = len(prefix_ids)
            end_token = len(full_ids)

            for i in range(start_token, min(end_token, len(labels))):
                labels[i] = input_ids_1d[i]

            pos = end_idx

        return labels

    def __getitem__(self, index):
        try:
            # 加载数据
            conversations = json.loads(self.table['conversations'][index].as_py())
            image_bytes = self.table['image_bytes'][index].as_py()
            if not isinstance(image_bytes, list):
                image_bytes = [image_bytes]

            # 解码图像
            images = []
            for img_data in image_bytes:
                try:
                    img = Image.open(io.BytesIO(img_data))
                    if img.mode in ['RGBA', 'LA']:
                        img = img.convert('RGB')
                    images.append(img)
                except Exception:
                    images.append(Image.new('RGB', (224, 224)))

            # 构建 messages
            messages = self._build_messages(conversations, images)

            # 使用 processor 处理
            text = self.processor.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )

            inputs = self.processor(
                text=[text],
                images=images,
                padding='max_length',
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            )

            # inputs.input_ids 形状: (1, seq_len)
            input_ids = inputs.input_ids[0]  # (seq_len,)
            attention_mask = inputs.attention_mask[0]  # (seq_len,)

            # 手动计算 mm_token_type_ids
            # 0 = text, 1 = image
            mm_token_type_ids = (input_ids == self.image_token_id).long()

            # 获取 pixel_values 和 image_grid_thw
            pixel_values = inputs.get('pixel_values')
            image_grid_thw = inputs.get('image_grid_thw')

            if image_grid_thw is not None and isinstance(image_grid_thw, torch.Tensor):
                if image_grid_thw.dim() == 3:
                    image_grid_thw = image_grid_thw.squeeze(0)
                elif image_grid_thw.dim() == 1:
                    image_grid_thw = image_grid_thw.unsqueeze(0)

            # 生成 labels
            labels = self._make_labels(input_ids, text)

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'pixel_values': pixel_values,
                'image_grid_thw': image_grid_thw,
                'mm_token_type_ids': mm_token_type_ids,
            }

        except Exception as e:
            print(f"Warning: Failed to process sample {index}: {e}")
            # 返回空样本
            input_ids = torch.zeros(self.max_length, dtype=torch.long)
            attention_mask = torch.zeros(self.max_length, dtype=torch.long)
            labels = torch.full((self.max_length,), -100, dtype=torch.long)
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'pixel_values': None,
                'image_grid_thw': None,
                'mm_token_type_ids': torch.zeros(self.max_length, dtype=torch.long),
            }


def collate_fn(batch):
    """自定义 collate 函数"""
    input_ids = torch.stack([b['input_ids'] for b in batch])
    attention_mask = torch.stack([b['attention_mask'] for b in batch])
    labels = torch.stack([b['labels'] for b in batch])
    mm_token_type_ids = torch.stack([b['mm_token_type_ids'] for b in batch])

    # 合并所有图像的 pixel_values
    pixel_values_list = []
    image_grid_thw_list = []
    for b in batch:
        pv = b['pixel_values']
        thw = b['image_grid_thw']
        if pv is not None:
            pixel_values_list.append(pv)
            image_grid_thw_list.append(thw)

    if pixel_values_list:
        pixel_values = torch.cat(pixel_values_list, dim=0)
        valid_thws = [t for t in image_grid_thw_list if t is not None]
        if valid_thws:
            image_grid_thw = torch.cat(valid_thws, dim=0)
        else:
            image_grid_thw = None
    else:
        pixel_values = None
        image_grid_thw = None

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'pixel_values': pixel_values,
        'image_grid_thw': image_grid_thw,
        'mm_token_type_ids': mm_token_type_ids,
    }


# ===================== 训练循环 =====================

def get_gpu_memory_gb():
    """获取当前 GPU 显存使用（GB）"""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def train_epoch(epoch, loader, iters, model, optimizer, scaler, autocast_ctx, args,
               start_step=0, ema_loss_init=0.0, wandb_run=None):
    start_time = time.time()

    # 初始化 EMA loss
    ema_loss = ema_loss_init
    ema_beta = 0.9  # EMA 衰减因子

    # 创建 tqdm 进度条（仅在主进程）
    pbar = None
    if is_main_process():
        pbar = tqdm(
            total=iters,
            initial=start_step,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            unit="step",
            ncols=160,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
        )

    for step, batch in enumerate(loader, start=start_step + 1):
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 数据移到设备
        input_ids = batch['input_ids'].to(args.device)
        attention_mask = batch['attention_mask'].to(args.device)
        labels = batch['labels'].to(args.device)
        mm_token_type_ids = batch['mm_token_type_ids'].to(args.device)

        pixel_values = None
        if batch['pixel_values'] is not None:
            pixel_values = batch['pixel_values'].to(args.device)
        image_grid_thw = None
        if batch['image_grid_thw'] is not None:
            image_grid_thw = batch['image_grid_thw'].to(args.device)

        # 统计 batch 信息
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        num_image_tokens = int((mm_token_type_ids == 1).sum().item())
        num_text_tokens = int((mm_token_type_ids == 0).sum().item())
        num_images = image_grid_thw.shape[0] if image_grid_thw is not None else 0

        step_start_time = time.time()

        # 前向传播
        with autocast_ctx:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
            )
            loss = outputs.loss / args.accumulation_steps

        # 反向传播
        scaler.scale(loss).backward()

        # 梯度更新
        grad_norm = 0.0
        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # 计算当前 loss 和 EMA
        current_loss = loss.item() * args.accumulation_steps
        if step == start_step + 1:
            ema_loss = current_loss
        else:
            ema_loss = ema_beta * ema_loss + (1 - ema_beta) * current_loss

        # 实时更新进度条（单行显示，不滚动）
        if pbar is not None:
            mem_gb = get_gpu_memory_gb()
            spend_time = time.time() - start_time
            eta_sec = spend_time / max(step - start_step, 1) * (iters - step)
            eta_str = f"{int(eta_sec // 3600):02d}h{int((eta_sec % 3600) // 60):02d}m{int(eta_sec % 60):02d}s"
            step_time = time.time() - step_start_time
            samples_per_sec = batch_size / step_time

            pbar.set_postfix_str(
                f"loss={ema_loss:.4f} | lr={lr:.2e} | gnorm={grad_norm:.2f} | "
                f"mem={mem_gb:.1f}GB | {samples_per_sec:.2f}it/s | eta={eta_str}"
            )
            pbar.update(1)

        # wandb 日志（按 log_interval 记录）
        if step % args.log_interval == 0 or step == iters:
            if wandb_run and is_main_process():
                mem_gb = get_gpu_memory_gb()
                step_time = time.time() - step_start_time
                samples_per_sec = batch_size / step_time
                tokens_per_sec = batch_size * seq_len / step_time
                wandb_run.log({
                    # 训练指标
                    "train/loss": current_loss,
                    "train/ema_loss": ema_loss,
                    "train/learning_rate": lr,
                    "train/grad_norm": grad_norm,
                    "train/epoch": epoch + 1,
                    "train/step": step,
                    "train/progress": step / iters,
                    # 性能指标
                    "perf/samples_per_sec": samples_per_sec,
                    "perf/tokens_per_sec": tokens_per_sec,
                    "perf/step_time_sec": step_time,
                    "perf/gpu_memory_gb": mem_gb,
                    # 数据指标
                    "data/batch_size": batch_size,
                    "data/seq_len": seq_len,
                    "data/num_images": num_images,
                    "data/num_image_tokens": num_image_tokens,
                    "data/num_text_tokens": num_text_tokens,
                    "data/image_token_ratio": num_image_tokens / max(num_image_tokens + num_text_tokens, 1),
                })

        # 保存
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            save_dir = os.path.join(args.save_dir, f"checkpoint-{step}")
            os.makedirs(save_dir, exist_ok=True)
            model_to_save = model.module if isinstance(model, DistributedDataParallel) else model
            if args.use_lora:
                model_to_save.save_pretrained(save_dir)
            else:
                model_to_save.model.save_pretrained(save_dir)
            if hasattr(model_to_save, 'processor') and model_to_save.processor is not None:
                model_to_save.processor.save_pretrained(save_dir)

            # 保存训练状态（用于断点续传）
            ckp_data = {
                "model": model_to_save.state_dict() if not args.use_lora else None,
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "step": step,
                "epoch": epoch,
                "ema_loss": ema_loss,
                "args": vars(args),
                "rng_state": {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all(),
                },
                "wandb_id": wandb_run.id if wandb_run is not None else None,
            }
            ckp_path = os.path.join(save_dir, "training_state.pt")
            torch.save(ckp_data, ckp_path)

            save_msg = f"Saved checkpoint to {save_dir}"
            if pbar is not None:
                pbar.write(save_msg)
            else:
                print(save_msg)

            # 清理旧 checkpoint（只保留最近 N 个）
            cleanup_old_checkpoints(args.save_dir, keep_last_n=args.keep_last_n)

        # 清理显存
        del input_ids, attention_mask, labels, pixel_values, outputs, loss

    # 关闭进度条
    if pbar is not None:
        pbar.close()


# ===================== 主函数 =====================

def main():
    parser = argparse.ArgumentParser(description="MiniQ-VL SFT Training")

    # 模型配置
    parser.add_argument("--model_path", type=str, default="./model/Qwen3-VL-2B-Instruct",
                       help="Qwen-VL 模型路径")
    parser.add_argument("--save_dir", type=str, default="./out", help="模型保存目录")
    parser.add_argument("--save_weight", type=str, default="sft_vlm", help="保存权重名称")

    # 冻结配置
    parser.add_argument("--freeze_vision", type=int, default=1, choices=[0, 1],
                       help="是否冻结视觉编码器 (0=否，1=是)")
    parser.add_argument("--freeze_language", type=int, default=0, choices=[0, 1],
                       help="是否冻结语言模型 (0=否，1=是)")

    # 训练配置
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="学习率")
    parser.add_argument("--device", type=str, default="cuda:0", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=2, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--max_seq_len", type=int, default=512, help="最大序列长度")
    parser.add_argument("--use_grad_checkpoint", type=int, default=1, choices=[0, 1],
                       help="是否使用梯度检查点（节省显存）")

    # 日志配置
    parser.add_argument("--log_interval", type=int, default=10, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=500, help="模型保存间隔")
    parser.add_argument("--keep_last_n", type=int, default=2, help="只保留最近 N 个 checkpoint")

    # 数据配置
    parser.add_argument("--data_path", type=str, default="./dataset/minimind-v_dataset/sft_i2t.parquet",
                       help="训练数据路径")
    parser.add_argument("--max_samples", type=int, default=200000,
                       help="最多使用 N 条样本（None=使用全部）")
    parser.add_argument("--from_resume", type=int, default=0, choices=[0, 1],
                       help="是否从检查点恢复训练")
    parser.add_argument("--pretrain_checkpoint", type=str, default=None,
                       help="Pretrain 检查点路径")
    parser.add_argument("--use_pretrain_model", type=int, default=1, choices=[0, 1],
                       help="是否自动加载最新的 Pretrain 模型")

    # wandb 配置
    parser.add_argument("--use_wandb", type=int, default=1, choices=[0, 1],
                       help="是否使用 wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniQ-VL-SFT",
                       help="wandb 项目名")
    parser.add_argument("--wandb_entity", type=str, default=None,
                       help="wandb 实体/团队名")

    # 其他
    parser.add_argument("--use_compile", type=int, default=1, choices=[0, 1],
                       help="是否使用 torch.compile 加速")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    # LoRA 配置
    parser.add_argument("--use_lora", type=int, default=1, choices=[0, 1],
                       help="是否使用 LoRA 训练 (1=是，0=否)")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha 缩放因子")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--lora_target_modules", type=str,
                       default="q_proj,k_proj,v_proj,o_proj",
                       help="LoRA 目标模块（逗号分隔）")
    parser.add_argument("--lora_merge_and_save", type=int, default=1, choices=[0, 1],
                       help="训练结束后是否合并 LoRA 权重并保存完整模型")

    args = parser.parse_args()

    # 解析路径
    args.model_path = resolve_path(args.model_path)
    args.data_path = resolve_path(args.data_path)
    args.save_dir = resolve_path(args.save_dir)

    # 创建输出目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 配置错误日志（自动捕获未处理异常）
    setup_error_logging(args)
    if is_main_process():
        print(f"Error logs will be saved to: {os.path.join(args.save_dir, 'error_logs')}")

    # 分布式
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))

    # 打印分布式状态
    world_size = get_world_size()
    rank = get_rank()
    if is_main_process():
        if dist.is_initialized():
            print(f"DDP initialized: world_size={world_size}, rank={rank}, local_rank={local_rank}")
        else:
            print(f"Single-GPU training (no DDP). Use torchrun to enable DDP.")
            print(f"  Example: torchrun --nproc_per_node=N trainer/train_sft.py")

    # wandb
    wandb_run = None
    if args.use_wandb and is_main_process():
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
        )

    # 加载模型
    if is_main_process():
        print(f"Loading model from {args.model_path}...")
    config = QwenVLMConfig(
        model_path=args.model_path,
        freeze_vision=bool(args.freeze_vision),
        freeze_language=bool(args.freeze_language),
        max_length=args.max_seq_len,
    )
    model = QwenVLM(config)
    model = model.to(args.device)

    # 启用梯度检查点
    if args.use_grad_checkpoint:
        try:
            inner = model.module if isinstance(model, DistributedDataParallel) else model
            inner.model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")
        except Exception as e:
            print(f"Warning: Could not enable gradient checkpointing: {e}")

    # DDP
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    # LoRA 训练
    if args.use_lora:
        try:
            from peft import LoraConfig, get_peft_model, TaskType

            # 解析 target_modules
            target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]

            # 冻结所有参数
            inner = model.module if isinstance(model, DistributedDataParallel) else model
            for param in inner.parameters():
                param.requires_grad = False

            # 配置 LoRA - 仅对语言模型的 attention 层
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=target_modules,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )

            # 应用 LoRA
            inner = model.module if isinstance(model, DistributedDataParallel) else model
            model_peft = get_peft_model(inner, lora_config)

            # 替换原 model 引用
            if isinstance(model, DistributedDataParallel):
                model.module = model_peft
            else:
                model = model_peft

            if is_main_process():
                model_peft.print_trainable_parameters()
                print(f"LoRA applied with rank={args.lora_r}, alpha={args.lora_alpha}, "
                      f"target_modules={target_modules}")

        except ImportError:
            print("Warning: peft not installed. Run: pip install peft")
            print("Falling back to full fine-tuning")
            args.use_lora = 0

    # 统计参数
    if is_main_process():
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        print(f"Model Params: {total_params:.2f}M, Trainable: {trainable_params:.2f}M")

    # 加载数据集
    if is_main_process():
        print(f"Loading dataset from {args.data_path}...")
    processor = model.module.processor if isinstance(model, DistributedDataParallel) else model.processor
    train_ds = MiniQVLMDataset(
        parquet_path=args.data_path,
        processor=processor,
        max_length=args.max_seq_len,
    )
    total_samples = len(train_ds)
    # 限制样本数
    if args.max_samples is not None and args.max_samples > 0 and args.max_samples < total_samples:
        train_ds.table = train_ds.table.slice(0, args.max_samples)
        if is_main_process():
            print(f"Limited to {args.max_samples} samples (from {total_samples} total)")
    if is_main_process():
        print(f"Dataset size: {len(train_ds)}")

    # DataLoader
    if dist.is_initialized():
        sampler = DistributedSampler(train_ds)
    else:
        sampler = None

    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 优化器
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # 混合精度
    autocast_ctx = nullcontext()
    if args.dtype == "bfloat16":
        autocast_ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16)
    elif args.dtype == "float16":
        autocast_ctx = torch.amp.autocast('cuda', dtype=torch.float16)

    scaler = torch.amp.GradScaler('cuda', enabled=(args.dtype == "float16"))

    # 断点续传
    start_epoch = 0
    start_step = 0
    resume_ema_loss = 0.0
    wandb_id = None

    if args.from_resume == 1:
        # 先在指定 save_dir 找，找不到就自动搜索最近的
        result = load_training_state(args.save_dir)
        if result is None:
            print(f"No checkpoint in {args.save_dir}, searching for the latest...")
            # 向上/向下搜索可能的训练目录
            search_roots = [
                args.save_dir,
                os.path.dirname(args.save_dir),
                os.path.dirname(os.path.dirname(args.save_dir)),
            ]
            for root in search_roots:
                if not os.path.exists(root):
                    continue
                for sub in os.listdir(root):
                    sub_path = os.path.join(root, sub)
                    if os.path.isdir(sub_path) and ("sft" in sub.lower() or "sft" in os.path.basename(args.save_dir).lower()):
                        r = load_training_state(sub_path)
                        if r is not None:
                            result = r
                            args.save_dir = sub_path
                            print(f"  → found in: {sub_path}")
                            break
                if result is not None:
                    break

        if result is not None:
            ckp_data, ckp_dir = result
            print(f"Resuming from checkpoint: {ckp_dir}")
            print(f"  - saved step: {ckp_data.get('step', 0)}")
            print(f"  - saved epoch: {ckp_data.get('epoch', 0)}")

            # 恢复模型权重（非 LoRA 训练）
            if not args.use_lora and ckp_data.get('model') is not None:
                model_to_load = model.module if isinstance(model, DistributedDataParallel) else model
                model_to_load.load_state_dict(ckp_data['model'], strict=False)
                print("  - model weights loaded")

            # 恢复优化器
            if 'optimizer' in ckp_data and ckp_data['optimizer'] is not None:
                optimizer.load_state_dict(ckp_data['optimizer'])
                print("  - optimizer state loaded")

            # 恢复 scaler
            if 'scaler' in ckp_data and ckp_data['scaler'] is not None and scaler is not None:
                scaler.load_state_dict(ckp_data['scaler'])
                print("  - scaler state loaded")

            # 恢复 RNG 状态
            if 'rng_state' in ckp_data:
                rng = ckp_data['rng_state']
                if 'python' in rng: random.setstate(rng['python'])
                if 'numpy' in rng: np.random.set_state(rng['numpy'])
                if 'torch' in rng: torch.set_rng_state(rng['torch'])
                if 'cuda' in rng:
                    if dist.is_initialized():
                        torch.cuda.set_rng_state(rng['cuda'][local_rank], device=local_rank)
                    else:
                        torch.cuda.set_rng_state(rng['cuda'][0])
                print("  - RNG state loaded")

            start_epoch = ckp_data.get('epoch', 0)
            start_step = ckp_data.get('step', 0)
            resume_ema_loss = ckp_data.get('ema_loss', 0.0)
            wandb_id = ckp_data.get('wandb_id', None)

            print(f"Resuming from epoch {start_epoch}, step {start_step}, ema_loss={resume_ema_loss:.4f}")
        else:
            print("No checkpoint found, starting from scratch")

    # 训练
    iters = len(loader)
    if is_main_process():
        print(f"Starting training: {args.epochs} epochs, {iters} iters/epoch")
    for epoch in range(start_epoch, args.epochs):
        if dist.is_initialized() and sampler is not None:
            sampler.set_epoch(epoch)
        # 如果是恢复的 epoch，从 start_step 继续；否则从 0 开始
        epoch_start_step = start_step if epoch == start_epoch else 0
        train_epoch(epoch, loader, iters, model, optimizer, scaler, autocast_ctx, args,
                   start_step=epoch_start_step, ema_loss_init=resume_ema_loss, wandb_run=wandb_run)
        # 恢复后，完整跑完剩余 step
        start_step = 0

    # 保存最终模型
    if is_main_process():
        final_dir = os.path.join(args.save_dir, args.save_weight)
        os.makedirs(final_dir, exist_ok=True)
        model_to_save = model.module if isinstance(model, DistributedDataParallel) else model

        if args.use_lora:
            # 保存 LoRA 适配器
            model_to_save.save_pretrained(final_dir)
            print(f"Saved LoRA adapter to {final_dir}")

            # 可选：合并 LoRA 权重到基础模型并保存
            if args.lora_merge_and_save:
                merged_dir = final_dir + "_merged"
                os.makedirs(merged_dir, exist_ok=True)
                merged_model = model_to_save.merge_and_unload()
                merged_model.model.save_pretrained(merged_dir)
                if hasattr(merged_model, 'processor') and merged_model.processor is not None:
                    merged_model.processor.save_pretrained(merged_dir)
                print(f"Saved merged model to {merged_dir}")
        else:
            model_to_save.model.save_pretrained(final_dir)
            if hasattr(model_to_save, 'processor') and model_to_save.processor is not None:
                model_to_save.processor.save_pretrained(final_dir)
            print(f"Saved final model to {final_dir}")

    if wandb_run:
        wandb_run.finish()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
