"""
MiniQ-VL Pretrain 训练脚本
魔改点:
1. 基模替换为 Qwen3-VL-2B-Instruct
2. 使用 wandb 监控（替换 swanlab）
3. 集成 tqdm 进度条
4. 仅训练 vision projection（冻结其他参数）
"""
import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from model.qwen_vl import QwenVLM, QwenVLMConfig
from dataset.sft_dataset import VLMDataset, vlm_collate_fn
from trainer.trainer_utils import (
    get_lr, Logger, is_main_process, init_distributed_mode, setup_seed,
    init_vlm_model, vlm_checkpoint, SkipBatchSampler, get_model_params,
    TrainingProgressBar
)

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, model, optimizer, scaler, autocast_ctx, args,
               start_step=0, wandb_run=None, pbar=None):
    """训练一个 epoch"""
    start_time = time.time()
    last_step = start_step
    
    for step, batch in enumerate(loader, start=start_step + 1):
        last_step = step
        
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        input_ids = batch['input_ids'].to(args.device)
        attention_mask = batch['attention_mask'].to(args.device)
        labels = batch['labels'].to(args.device)
        
        # 处理 pixel_values（动态分辨率可能导致形状不一致）
        pixel_values = None
        if batch['pixel_values'][0] is not None:
            pv_list = batch['pixel_values']
            try:
                # 尝试堆叠（形状一致时）
                pixel_values = torch.stack(pv_list).to(args.device)
            except (TypeError, RuntimeError):
                # 形状不一致时，逐样本处理
                pixel_values = pv_list
        
        # 提取 image_grid_thw
        image_grid_thw = None
        if batch.get('image_grid_thw') and batch['image_grid_thw'][0] is not None:
            thw_list = batch['image_grid_thw']
            try:
                image_grid_thw = torch.stack(thw_list).to(args.device)
            except (TypeError, RuntimeError):
                image_grid_thw = thw_list
        
        with autocast_ctx:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw
            )
            loss = outputs.loss / args.accumulation_steps
        
        scaler.scale(loss).backward()
        
        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            
            log_msg = (
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), "
                f"loss: {current_loss:.4f}, lr: {lr:.2e}, "
                f"eta: {eta_min:.1f}min"
            )
            
            if pbar:
                pbar.update(1)
                pbar.write(log_msg)
            else:
                Logger(log_msg)
            
            if wandb_run and is_main_process():
                wandb_run.log({
                    "loss": current_loss,
                    "learning_rate": lr,
                    "epoch": epoch + 1,
                    "step": step
                })
        
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            vlm_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=step,
                wandb_run=wandb_run,
                save_dir=args.save_dir,
                save_weight=args.save_weight,
                scaler=scaler
            )
            model.train()
        
        del input_ids, attention_mask, labels, pixel_values, outputs, loss
    
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniQ-VL Pretrain Training")
    
    # 模型配置
    parser.add_argument("--model_path", type=str, default="./model/Qwen3-VL-2B-Instruct",
                       help="Qwen-VL 模型路径")
    parser.add_argument("--save_dir", type=str, default="./out", help="模型保存目录")
    parser.add_argument("--save_weight", type=str, default="pretrain_vlm", help="保存权重名称")
    
    # 训练配置
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--device", type=str, default="cuda:0", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=2, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="最大序列长度")
    
    # 日志配置
    parser.add_argument("--log_interval", type=int, default=10, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    
    # 冻结配置（预训练默认冻结 LLM）
    parser.add_argument("--freeze_vision", type=int, default=1, choices=[0, 1],
                       help="是否冻结视觉编码器 (0=否，1=是)")
    parser.add_argument("--freeze_language", type=int, default=1, choices=[0, 1],
                       help="是否冻结语言模型 (0=否，1=是)")
    
    # 数据配置
    parser.add_argument("--data_path", type=str, default="./dataset/minimind-v_dataset/pretrain_i2t.parquet",
                       help="训练数据路径")
    parser.add_argument("--from_resume", type=int, default=0, choices=[0, 1],
                       help="是否从检查点恢复训练")
    
    # wandb 配置（默认开启）
    parser.add_argument("--use_wandb", type=int, default=1, choices=[0, 1],
                       help="是否使用 wandb (1=是，0=否)")
    parser.add_argument("--wandb_project", type=str, default="MiniQ-VL-Pretrain",
                       help="wandb 项目名")
    parser.add_argument("--wandb_entity", type=str, default=None,
                       help="wandb 实体/团队名")
    
    # 其他
    parser.add_argument("--use_compile", type=int, default=0, choices=[0, 1],
                       help="是否使用 torch.compile 加速")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    # ========== 1. 初始化环境 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    
    setup_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 目录配置（自动生成带时间戳的目录）==========
    from datetime import datetime
    
    # 创建基准目录
    base_dir = os.path.join(os.path.dirname(__file__), args.save_dir)
    os.makedirs(base_dir, exist_ok=True)
    
    # 自动生成带时间戳的子目录（仅首次训练时）
    if args.from_resume == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 判断是否使用冻结配置
        freeze_status = "freeze" if args.freeze_vision else "full"
        sub_dir_name = f"{timestamp}_pretrain_{freeze_status}"
        save_dir = os.path.join(base_dir, sub_dir_name)
        os.makedirs(save_dir, exist_ok=True)
        args.save_dir = save_dir
        Logger(f"Output directory: {save_dir}")
    else:
        save_dir = base_dir
    
    # ========== 3. 检查点加载 ==========
    ckp_data = None
    if args.from_resume == 1:
        ckp_data = vlm_checkpoint(save_dir=args.save_dir, save_weight=args.save_weight)
    
    # ========== 4. 混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 5. 初始化 wandb ==========
    wandb_run = None
    if args.use_wandb and is_main_process():
        import wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            id=wandb_id,
            resume=resume,
            config=vars(args)
        )
        Logger(f"Wandb initialized: {wandb_run.url}")
    
    # ========== 6. 加载模型和数据 ==========
    model, processor = init_vlm_model(
        model_path=args.model_path,
        freeze_vision=bool(args.freeze_vision),
        freeze_language=bool(args.freeze_language),
        device=args.device
    )
    
    train_ds = VLMDataset(
        args.data_path,
        processor=processor,
        max_length=args.max_seq_len
    )
    
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    
    # ========== 7. 优化器和混合精度 ==========
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=0.01
    )
    
    # ========== 8. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 9. 恢复训练状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        if 'scaler' in ckp_data and ckp_data['scaler']:
            try:
                scaler.load_state_dict(ckp_data['scaler'])
            except Exception:
                Logger("Warning: scaler state_dict mismatch, using fresh scaler")
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
        Logger(f'Resumed from epoch {start_epoch}, step {start_step}')
    
    # ========== 10. 开始训练 ==========
    total_steps = args.epochs * len(train_ds) // args.batch_size
    
    with TrainingProgressBar(total_steps, desc="MiniQ-VL Pretrain", log_interval=args.log_interval) as pbar:
        for epoch in range(start_epoch, args.epochs):
            if train_sampler:
                train_sampler.set_epoch(epoch)
            
            setup_seed(args.seed + epoch)
            indices = list(range(len(train_ds)))
            
            skip = start_step if epoch == start_epoch and start_step > 0 else 0
            batch_sampler = SkipBatchSampler(indices, args.batch_size, skip)
            
            loader = DataLoader(
                train_ds,
                batch_sampler=batch_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
                collate_fn=vlm_collate_fn
            )
            
            train_epoch(
                epoch, loader, len(loader) + skip,
                model, optimizer, scaler, autocast_ctx, args,
                start_step if skip > 0 else 0,
                wandb_run, pbar
            )
    
    # ========== 11. 结束 ==========
    if dist.is_initialized():
        dist.destroy_process_group()
    
    if wandb_run:
        wandb_run.finish()
    
    Logger("Pretrain completed!")