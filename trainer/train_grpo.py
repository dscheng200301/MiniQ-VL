"""
MiniQ-VL GRPO 训练脚本 (重写版)
================================

视觉描述质量提升 - Group Relative Policy Optimization

训练流程:
    1. 从 SFT 数据筛选视觉描述类样本 (dataset.grpo_dataset)
    2. 每个 prompt 采样 K 条回答 (group_size, 默认 4)
    3. 计算 Reward: CLIPScore + LLM-Judge + Length Penalty (+ 可选 R4/R5/R6)
    4. 组内归一化计算优势值 (group normalization)
    5. 策略梯度更新 (GRPO clipped objective + 可选 KL 约束)

与 SFT 的关键差异:
    - 训练数据需要先采样 K 条, 然后计算 reward/advantage, 再做一次正反向
    - 视觉特征只编码一次并复用 (避免重复 pixel_values 计算)
    - 可选 ref_model 模式计算 KL 散度防止策略漂移
    - 每个 rank 各自采样一个 group, 不做 all-gather 同步

使用:
    torchrun --standalone --nproc_per_node=N trainer/train_grpo.py <args>
"""

import os
import sys
import math
import time
import io
import json
import argparse
import traceback
import warnings
from contextlib import nullcontext
from datetime import datetime

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F

from utils.api_client import _DEFAULT_MODEL
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset
from PIL import Image

from dataset.grpo_dataset import GRPODataset, grpo_collate_fn
from trainer.trainer_utils import (
    get_lr, Logger, is_main_process, init_distributed_mode, setup_seed,
    init_vlm_model, vlm_checkpoint, SkipBatchSampler,
    TrainingProgressBar, find_latest_checkpoint,
)
from trainer.grpo_utils import (
    CLIPScorer, LLMJudge, compute_rewards, compute_advantages,
    compute_grpo_loss_with_kl,
)

warnings.filterwarnings('ignore')


# ============================================================
# 工具函数
# ============================================================

def decode_image(image_bytes_list):
    """从字节列表解码图像 (多模态数据集可能给出 bytes 列表, 这里取首张)"""
    try:
        raw = image_bytes_list[0] if isinstance(image_bytes_list, list) else image_bytes_list
        img = Image.open(io.BytesIO(raw))
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')
        return img
    except Exception:
        return Image.new('RGB', (224, 224))


def resolve_project_path(path):
    """将相对路径解析为项目根目录下的绝对路径; 绝对路径/ HF Hub ID 保持原样"""
    if path is None or os.path.isabs(path):
        return path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, path)


def unwrap_model(model):
    """剥开 DDP / torch.compile 包装, 拿到原始 QwenVLM"""
    raw = model
    if isinstance(raw, DistributedDataParallel):
        raw = raw.module
    if hasattr(raw, '_orig_mod'):  # torch.compile
        raw = raw._orig_mod
    return raw


# Qwen 模型的 eos 是 <|im_end|>, pad_token 不能用它. 下面这些是 Qwen 词表中
# 明确不等于 eos 的 token, 优先选第一个存在的.
_QWEN_PAD_CANDIDATES = [
    "<|endoftext|>",          # Qwen 主 eod, 一定存在
    "<|fim_pad|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|placeholder1|>",
    "<|placeholder2|>",
    "<|object_ref_start|>",
    "<|reserved_special_token_0|>",
    "<|reserved_special_token_1|>",
]


def _pick_pad_token(tok):
    """
    从 _QWEN_PAD_CANDIDATES 里选一个不等于 eos/unk 的 token.
    全部失败就 add_special_tokens('<|pad|>') 并返回 (新 token 字符串, 是否新增).
    """
    eos = tok.eos_token_id
    unk = tok.unk_token_id
    for cand in _QWEN_PAD_CANDIDATES:
        tid = tok.convert_tokens_to_ids(cand)
        # 有效 id: 不为 None, 不等于 unk, 不等于 eos
        if tid is not None and tid != unk and tid != eos:
            return cand, tid, False
    # 兜底: 新加 pad token (需要 resize embedding, 由调用方处理)
    return "<|pad|>", None, True


def ensure_pad_token(processor, model=None):
    """
    显式设置一个与 eos 不同的 pad_token, 避免 generate 时被刷 warning.

    重要: Qwen 系列 tokenizer 的 eos_token_id == <|im_end|> 的 id,
          之前直接 pad_token='<|im_end|>' 等于没改, 所以这里要选
          明确不等于 eos 的 token.

    Args:
        processor: QwenVLM 的 processor
        model: 可选, 若传入则同步设置其 config / generation_config / text_config
    """
    tok = processor.tokenizer
    eos = tok.eos_token_id
    unk = tok.unk_token_id

    need_change = (
        tok.pad_token is None
        or tok.pad_token_id == eos
        or tok.pad_token_id == unk
    )

    if need_change:
        cand, cand_id, need_resize = _pick_pad_token(tok)
        if not need_resize:
            tok.pad_token = cand
            tok.pad_token_id = cand_id
            Logger(f"Set pad_token to {cand!r} (id={cand_id}, eos_id={eos})")
        else:
            # 全部候选都没法用, 新加 token
            Logger(f"Warning: no suitable existing token, adding new {cand!r}")
            tok.add_special_tokens({"pad_token": cand})
            new_pad_id = tok.convert_tokens_to_ids(cand)
            tok.pad_token_id = new_pad_id
            if model is not None:
                try:
                    inner = unwrap_model(model).model
                    if hasattr(inner, "resize_token_embeddings"):
                        inner.resize_token_embeddings(len(tok))
                        Logger(f"Resized model embeddings to {len(tok)}")
                except Exception as e:
                    Logger(f"Warning: resize_token_embeddings failed: {e}")

    # 同步到模型所有可能的位置, 避免 generate 内部仍用 eos
    if model is not None:
        try:
            inner = unwrap_model(model).model
            pad_id = tok.pad_token_id

            if hasattr(inner, "config") and inner.config is not None:
                inner.config.pad_token_id = pad_id
                # 嵌套 text_config (多模态模型常见)
                if hasattr(inner.config, "text_config") and inner.config.text_config is not None:
                    inner.config.text_config.pad_token_id = pad_id

            if hasattr(inner, "generation_config") and inner.generation_config is not None:
                inner.generation_config.pad_token_id = pad_id
                if hasattr(inner.generation_config, "text_config") and inner.generation_config.text_config is not None:
                    inner.generation_config.text_config.pad_token_id = pad_id

            Logger(f"Synced pad_token_id={pad_id} to model config / generation_config")
        except Exception as e:
            Logger(f"Warning: failed to sync pad_token to model config: {e}")

    return tok


# ============================================================
# Prompt tokenize (主进程按需做, 避免 dataloader 多进程重复计算)
# ============================================================

def build_chat_input(processor, image, prompt_text, max_length=2048):
    """
    构造 chat 模板并 tokenize, 返回 processor 输出的 dict
    """
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ],
    }]
    text = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(
        text=text,
        images=[image],
        padding=False,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    return inputs


# ============================================================
# 单 prompt 采样 K 条回答
# ============================================================

@torch.no_grad()
def sample_group(raw_model, processor, prompt_input, group_size, gen_kwargs, device):
    """
    对一个 prompt 采样 K 条回答

    Returns:
        list of dict: {full_ids, prompt_len, text}
    """
    input_ids = prompt_input['input_ids'].to(device)
    attention_mask = prompt_input['attention_mask'].to(device)
    prompt_len = int(attention_mask.sum().item())

    gen_inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        # 显式传 pad_token_id, 避免 generate 内部用 eos 推断 (会刷 warning)
        'pad_token_id': processor.tokenizer.pad_token_id,
        'max_new_tokens': gen_kwargs.get('max_new_tokens', 256),
        'do_sample': True,
        'temperature': gen_kwargs.get('temperature', 0.8),
        'top_p': gen_kwargs.get('top_p', 0.9),
    }
    pv = prompt_input.get('pixel_values')
    if pv is not None:
        gen_inputs['pixel_values'] = pv.to(device)
    thw = prompt_input.get('image_grid_thw')
    if thw is not None:
        gen_inputs['image_grid_thw'] = thw.to(device)

    results = []
    for _ in range(group_size):
        full_ids = raw_model.generate(**gen_inputs)[0].cpu()
        response_ids = full_ids[prompt_len:]
        results.append({
            'text': processor.tokenizer.decode(response_ids, skip_special_tokens=True),
            'full_ids': full_ids,
            'prompt_len': prompt_len,
        })
    return results


# ============================================================
# 序列填充
# ============================================================

def pad_full_sequences(full_ids_list, pad_id, max_len):
    """
    将长度不一的 full_ids 填充到统一 max_len

    Returns:
        padded: [N, max_len] long
        attn:   [N, max_len] long
    """
    N = len(full_ids_list)
    padded = torch.full((N, max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((N, max_len), dtype=torch.long)
    for i, seq in enumerate(full_ids_list):
        length = min(seq.size(0), max_len)
        padded[i, :length] = seq[:length]
        attn[i, :length] = 1
    return padded, attn


def replicate_visual(prompt_inputs_list, group_size, device):
    """
    把 B 个 prompt 的视觉相关张量复制为 B*K, 并拼接
    """
    out = {}
    for key in ('pixel_values', 'image_grid_thw'):
        chunks = []
        for pi in prompt_inputs_list:
            v = pi.get(key)
            if v is None:
                continue
            chunks.append(v.to(device))
        if not chunks:
            out[key] = None
            continue
        # 同一 prompt 重复 K 次
        expanded = []
        for c in chunks:
            for _ in range(group_size):
                expanded.append(c)
        # 优先 cat, cat 失败就 stack
        try:
            out[key] = torch.cat(expanded, dim=0)
        except Exception:
            out[key] = torch.stack(expanded, dim=0)
    return out['pixel_values'], out['image_grid_thw']


def build_mm_token_type_ids(prompt_inputs_list, group_size, max_len, pad_id, device):
    """
    构造 mm_token_type_ids: 视觉 token 处为 1, 文本处为 0
    Qwen3-VL processor 在第一次 tokenize 时会给出 mm_token_type_ids
    这里按 (B -> B*K) 复制并 pad 到 max_len
    """
    chunks = []
    for pi in prompt_inputs_list:
        v = pi.get('mm_token_type_ids')
        if v is None:
            return None
        chunks.append(v.to(device).squeeze(0))  # [L]

    if not chunks:
        return None

    N = len(chunks) * group_size
    out = torch.zeros(N, max_len, dtype=torch.long, device=device)
    for i, c in enumerate(chunks):
        for k in range(group_size):
            row = i * group_size + k
            length = min(c.size(0), max_len)
            out[row, :length] = c[:length]
    return out


# ============================================================
# 计算回答部分的平均 log 概率
# ============================================================

def compute_avg_log_probs(model, input_ids, attn_mask, prompt_lens,
                          pixel_values=None, image_grid_thw=None, mm_token_type_ids=None):
    """
    计算 response 部分的 token 平均 log 概率

    Args:
        model:  VLM 模型 (DDP 包装也 OK)
        input_ids: [N, L]
        attn_mask: [N, L]
        prompt_lens: list[int]  每个样本的 prompt 长度
        pixel_values / image_grid_thw / mm_token_type_ids: 视觉相关

    Returns:
        avg_log_probs: [N]  tensor
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attn_mask = attn_mask.to(device)

    fwd_kwargs = {}
    if pixel_values is not None:
        fwd_kwargs['pixel_values'] = pixel_values
    if image_grid_thw is not None:
        fwd_kwargs['image_grid_thw'] = image_grid_thw
    if mm_token_type_ids is not None:
        fwd_kwargs['mm_token_type_ids'] = mm_token_type_ids
    # 训练时不要 KV cache, 避免和 grad checkpointing 冲突 (避免刷 use_cache 警告)
    fwd_kwargs['use_cache'] = False

    out = model(input_ids=input_ids, attention_mask=attn_mask, **fwd_kwargs)
    logits = out.logits  # [N, L, V]

    log_probs = F.log_softmax(logits, dim=-1)
    shift_log_probs = log_probs[:, :-1, :]
    shift_ids = input_ids[:, 1:]

    token_lp = shift_log_probs.gather(-1, shift_ids.unsqueeze(-1)).squeeze(-1)  # [N, L-1]

    # mask: 1 for response tokens
    resp_mask = attn_mask[:, 1:].clone().float()
    for i, plen in enumerate(prompt_lens):
        resp_mask[i, : max(plen - 1, 0)] = 0.0

    masked = token_lp * resp_mask
    avg_lp = masked.sum(dim=1) / (resp_mask.sum(dim=1) + 1e-8)
    return avg_lp


# ============================================================
# 单个 epoch 训练循环
# ============================================================

def train_one_epoch(epoch, loader, model, processor, optimizer, scaler,
                    ref_model, clip_scorer, llm_judge, args,
                    start_step=0, wandb_run=None, pbar=None,
                    hallucination_penalizer=None):
    """训练一个 epoch, 包含 GRPO 完整流程"""
    rank = dist.get_rank() if dist.is_initialized() else 0
    device = args.device
    iters = len(loader)
    start_time = time.time()
    last_step = start_step

    raw_model = unwrap_model(model)

    for step, batch in enumerate(loader, start=start_step + 1):
        last_step = step

        # ------- LR 调度 -------
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        try:
            # ------- 1. 解码图像 + tokenize -------
            images, prompt_inputs_list = [], []
            for img_bytes_list, prompt_text in zip(batch['image_bytes'], batch['prompt_text']):
                img = decode_image(img_bytes_list)
                images.append(img)
                prompt_inputs_list.append(
                    build_chat_input(processor, img, prompt_text, max_length=args.max_seq_len)
                )

            # ------- 2. 每个 prompt 采样 K 条 -------
            raw_model.eval()  # generate 阶段关 BN/Dropout
            gen_kwargs = {
                'max_new_tokens': args.max_new_tokens,
                'temperature': args.temperature,
                'top_p': args.top_p,
            }
            all_responses = []
            for pi in prompt_inputs_list:
                all_responses.extend(
                    sample_group(raw_model, processor, pi, args.group_size, gen_kwargs, device)
                )
            raw_model.train()

            # ------- 3. 计算 Reward -------
            texts = [r['text'] for r in all_responses]
            # 复制 image 使其与 group 内 K 份一一对应
            expanded_images = []
            for i_img, img in enumerate(images):
                expanded_images.extend([img] * args.group_size)
            references = []
            for ref, _ in zip(batch['reference'], range(len(images))):
                references.extend([ref] * args.group_size)

            rewards = compute_rewards(
                expanded_images, texts, clip_scorer, llm_judge,
                references=references,
                w1=args.w1, w2=args.w2, w3=args.w3,
                w4=args.w4, w5=args.w5, w6=args.w6,
                length_min=args.length_min, length_max=args.length_max,
                hallucination_penalizer=hallucination_penalizer,
            )

            # ------- 4. 优势值 -------
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
            advantages = compute_advantages(rewards_tensor, args.group_size)

            # ------- 5. 序列填充 + 视觉特征复制 -------
            pad_id = processor.tokenizer.pad_token_id or 0
            full_ids_list = [r['full_ids'] for r in all_responses]
            padded, attn_mask = pad_full_sequences(
                full_ids_list, pad_id, max_len=args.max_seq_len
            )

            pv_rep, thw_rep = replicate_visual(prompt_inputs_list, args.group_size, device)
            mtt_rep = build_mm_token_type_ids(
                prompt_inputs_list, args.group_size, args.max_seq_len, pad_id, device
            )

            prompt_lens = [r['prompt_len'] for r in all_responses]

            # ------- 6. old_log_probs (无梯度) -------
            with torch.no_grad():
                old_log_probs = compute_avg_log_probs(
                    model, padded, attn_mask, prompt_lens,
                    pixel_values=pv_rep, image_grid_thw=thw_rep,
                    mm_token_type_ids=mtt_rep,
                ).detach()

            # ------- 7. ref_log_probs (无梯度) -------
            ref_log_probs = None
            if ref_model is not None:
                with torch.no_grad():
                    ref_log_probs = compute_avg_log_probs(
                        ref_model, padded, attn_mask, prompt_lens,
                        pixel_values=pv_rep, image_grid_thw=thw_rep,
                        mm_token_type_ids=mtt_rep,
                    ).detach()

            # ------- 8. 当前策略 log_probs (需梯度) -------
            new_log_probs = compute_avg_log_probs(
                model, padded, attn_mask, prompt_lens,
                pixel_values=pv_rep, image_grid_thw=thw_rep,
                mm_token_type_ids=mtt_rep,
            )

            # ------- 9. GRPO Loss -------
            loss = compute_grpo_loss_with_kl(
                new_log_probs, old_log_probs, advantages,
                ref_log_probs=ref_log_probs,
                clip_eps=args.clip_eps,
                kl_coef=args.kl_coef,
            )
            loss_for_log = loss.item()
            loss = loss / args.accumulation_steps

            # ------- 10. 反向 -------
            scaler.scale(loss).backward()

            if step % args.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        except torch.cuda.OutOfMemoryError as oom:
            Logger(f"[rank{rank}] OOM at step {step}: {oom}. 跳过本 step.", pbar)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            Logger(f"[rank{rank}] step {step} failed: {e}", pbar)
            raise

        # ------- 11. 日志 -------
        if step % args.log_interval == 0 or step == iters:
            spend = time.time() - start_time
            eta_min = spend / max(step - start_step, 1) * (iters - step) // 60
            avg_reward = rewards_tensor.mean().item()
            log_msg = (
                f"GRPO E[{epoch+1}/{args.epochs}] "
                f"({step}/{iters}) loss={loss_for_log:.4f} "
                f"reward={avg_reward:.4f} lr={lr:.2e} eta={eta_min:.1f}min"
            )
            if pbar:
                pbar.update(1)
                pbar.write(log_msg)
            else:
                Logger(log_msg)

            if wandb_run and is_main_process():
                log_dict = {
                    "grpo_loss": loss_for_log,
                    "avg_reward": avg_reward,
                    "learning_rate": lr,
                    "epoch": epoch + 1,
                    "step": step,
                }
                if ref_log_probs is not None:
                    log_dict["kl_divergence"] = (new_log_probs - ref_log_probs).mean().item()
                wandb_run.log(log_dict)

        # ------- 12. 保存 -------
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            raw_model.eval()
            vlm_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=step,
                wandb_run=wandb_run,
                save_dir=args.save_dir,
                save_weight=args.save_weight,
                scaler=scaler,
            )
            raw_model.train()

        # 显存释放
        del padded, attn_mask, rewards_tensor, advantages
        if ref_log_probs is not None:
            del ref_log_probs
        torch.cuda.empty_cache()

    # 收尾: 末尾残余梯度
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return last_step


# ============================================================
# 路径 & 初始化辅助
# ============================================================

def setup_save_dir(args):
    """
    解析 save_dir:
      - 续训: 找最近一次 grpo 时间戳目录
      - 首次: 新建 timestamp_grpo_xxx 子目录
    """
    base_dir = resolve_project_path(args.save_dir)
    os.makedirs(base_dir, exist_ok=True)

    if args.from_resume == 1:
        # 在 base_dir 找一个含有 {save_weight}_checkpoint.pt 的子目录
        for d in sorted(os.listdir(base_dir), reverse=True):
            cand = os.path.join(base_dir, d)
            if not os.path.isdir(cand):
                continue
            if os.path.exists(os.path.join(cand, f"{args.save_weight}_checkpoint.pt")):
                Logger(f"Resume: use {cand}")
                return cand
        Logger(f"Resume: no checkpoint subdir found, fallback to {base_dir}")
        return base_dir

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    freeze_tag = "freeze" if args.freeze_vision else "full"
    reward_tag = f"w1{args.w1:.1f}_w2{args.w2:.1f}_w3{args.w3:.1f}"
    sub = f"{ts}_grpo_{freeze_tag}_{reward_tag}"
    save_dir = os.path.join(base_dir, sub)
    os.makedirs(save_dir, exist_ok=True)
    Logger(f"Output directory: {save_dir}")
    return save_dir


def _load_state_smart(model, sd, tag="state"):
    """
    智能加载 state_dict, 自动尝试加 model. 前缀.

    背景:
      - QwenVLM.state_dict() 的键都带 `model.` 前缀 (因为 self.model = ...)
      - HF save_pretrained() 出来的 .bin / .safetensors 键不带 `model.` 前缀
        (因为它存的是 inner HF 模型的 state_dict)
      - 训练主进程用 torch.save 存的 .pt 文件, 一般带 `model.` 前缀

    两种键形式都试, 取 missing 较少的那种.
    """
    candidates = [("raw", sd)]
    if hasattr(model, "model"):
        candidates.append(("with_model_prefix", {f"model.{k}": v for k, v in sd.items()}))

    best_choice = None
    best_missing = None
    for t, sd_try in candidates:
        try:
            missing, unexpected = model.load_state_dict(sd_try, strict=False)
            n_miss = len(missing)
            Logger(f"  [{tag}:{t}] missing={n_miss} unexpected={len(unexpected)}")
            if best_missing is None or n_miss < best_missing:
                best_missing = n_miss
                best_choice = t
            if n_miss == 0:
                return True
        except Exception as e:
            Logger(f"  [{tag}:{t}] failed: {e}")
            continue

    if best_choice is not None:
        if best_missing and best_missing > 0:
            Logger(f"  ⚠ 仍有 {best_missing} 个 key 未匹配, 模型可能加载不完整")
        return True
    return False


def _load_one(model, path):
    """从单个文件加载 state_dict (统一处理 .pt / .bin / .safetensors)"""
    name = os.path.basename(path)

    if path.endswith(".pt"):
        try:
            ckp = torch.load(path, map_location='cpu')
        except Exception as e:
            Logger(f"Warning: failed to read {path}: {e}")
            return False
        sd = ckp['model'] if isinstance(ckp, dict) and 'model' in ckp else ckp
        if sd is None:
            Logger(f"Warning: {path} has no 'model' field (LoRA-only resume?)")
            return False
        if _load_state_smart(model, sd, tag=name):
            Logger(f"Loaded SFT checkpoint: {path}")
            return True
        return False

    if path.endswith(".bin"):
        try:
            sd = torch.load(path, map_location='cpu')
        except Exception as e:
            Logger(f"Warning: failed to read {path}: {e}")
            return False
        if _load_state_smart(model, sd, tag=name):
            Logger(f"Loaded SFT bin: {path}")
            return True
        return False

    if path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
            sd = load_file(path)
        except ImportError:
            Logger("Warning: safetensors not installed")
            return False
        except Exception as e:
            Logger(f"Warning: failed to read {path}: {e}")
            return False
        if _load_state_smart(model, sd, tag=name):
            Logger(f"Loaded SFT safetensors: {path}")
            return True
        return False

    return False


def try_load_sft_checkpoint(model, base_save_dir, args):
    """
    优先级:
      1) --sft_checkpoint 显式路径 (支持 .pt / .bin / .safetensors / 整个目录)
      2) --use_sft_model=1 且 use_base_model=0 时, 找 out/ 下最新 sft_* 目录
         - 优先找 sft_vlm_checkpoint.pt (旧 .pt 格式)
         - 退而找 model.safetensors / model.bin (lora_merge_and_save 输出)
      3) 都不满足则跳过
    """
    # 1) 显式指定
    if args.sft_checkpoint:
        path = args.sft_checkpoint
        if not os.path.exists(path):
            Logger(f"Warning: --sft_checkpoint not found: {path}")
            return
        if os.path.isdir(path):
            for name in ("sft_vlm_checkpoint.pt", "model.safetensors", "pytorch_model.bin", "model.bin"):
                cand = os.path.join(path, name)
                if os.path.exists(cand) and _load_one(model, cand):
                    return
            Logger(f"Warning: no supported checkpoint file in {path}")
            return
        _load_one(model, path)
        return

    # 2) 自动从 out/ 找
    if args.use_sft_model == 1 and args.use_base_model == 0:
        sft_dir, _ = find_latest_checkpoint(base_save_dir, stage_prefix="sft")
        if sft_dir is None:
            Logger("Warning: no sft_* directory found under out/")
            return
        for name in ("sft_vlm_checkpoint.pt", "model.safetensors", "pytorch_model.bin", "model.bin"):
            cand = os.path.join(sft_dir, name)
            if os.path.exists(cand):
                if _load_one(model, cand):
                    return
        Logger(f"Warning: no supported checkpoint file in {sft_dir}")


# ============================================================
# 主流程
# ============================================================

def train_main(args, state):
    """
    state: dict 包含 last_successful_step / error_report_dir, 供错误回调使用
    """
    # ------- 1. 分布式 -------
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))

    # ------- 2. 目录 -------
    base_save_dir = resolve_project_path(args.save_dir)
    args.save_dir = setup_save_dir(args)
    state["error_report_dir"] = args.save_dir

    # ------- 3. 加载 / 恢复 checkpoint -------
    ckp_data = None
    if args.from_resume == 1:
        ckp_data = vlm_checkpoint(save_dir=args.save_dir, save_weight=args.save_weight)

    # ------- 4. 混合精度上下文 -------
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ------- 5. wandb -------
    wandb_run = None
    if args.use_wandb and is_main_process():
        try:
            import wandb
            wandb_id = ckp_data.get('wandb_id') if ckp_data else None
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                id=wandb_id,
                resume='must' if wandb_id else None,
                config=vars(args),
            )
            Logger(f"Wandb: {wandb_run.url}")
        except Exception as e:
            Logger(f"Warning: wandb init failed ({e}), continue without it")
            wandb_run = None

    # ------- 6. 模型 -------
    model, processor = init_vlm_model(
        model_path=resolve_project_path(args.model_path),
        freeze_vision=bool(args.freeze_vision),
        freeze_language=bool(args.freeze_language),
        device=args.device,
    )
    ensure_pad_token(processor, model)

    if args.use_grad_checkpoint == 1:
        try:
            # QwenVLM 是 wrapper, 真正的 HF 模型在 .model 属性
            inner_model = unwrap_model(model).model
            inner_model.gradient_checkpointing_enable()
            # 同步关掉 use_cache, 避免 transformers 刷 'use_cache=True is incompatible' 警告
            try:
                inner_model.config.use_cache = False
                if hasattr(inner_model.generation_config, "use_cache"):
                    inner_model.generation_config.use_cache = False
            except Exception:
                pass
            Logger("Gradient checkpointing enabled (use_cache=False)")
        except Exception as e:
            Logger(f"Warning: grad checkpointing failed: {e}")

    if args.use_enable_input_require_grads:
        try:
            # 同样需要访问 inner model
            inner_model = unwrap_model(model).model
            if hasattr(inner_model, "enable_input_require_grads"):
                inner_model.enable_input_require_grads()
        except Exception:
            pass

    # 加载权重: 续训 > SFT 加载 > 基模
    if ckp_data:
        unwrap_model(model).load_state_dict(ckp_data['model'], strict=False)
        Logger("Loaded GRPO resume checkpoint")
    elif args.use_base_model == 1:
        Logger(f"Using base model from {args.model_path}, skip SFT loading")
    else:
        try_load_sft_checkpoint(unwrap_model(model), base_save_dir, args)

    # ------- 7. 参考模型 (KL 约束) -------
    ref_model = None
    if args.use_ref_model:
        Logger("Loading reference model for KL constraint...")
        ref_model, _ = init_vlm_model(
            model_path=resolve_project_path(args.model_path),
            freeze_vision=True,
            freeze_language=True,
            device=args.device,
        )
        # ref 初始权重 = 当前 model
        unwrap_model(ref_model).load_state_dict(unwrap_model(model).state_dict(), strict=False)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

    # ------- 8. 数据集 -------
    train_ds = GRPODataset(
        args.data_path,
        filter_mode=args.filter_mode,
        llm_model=args.filter_llm_model,
        cache_path=args.filter_cache,
        prefiltered_path=args.prefiltered_path,
    )
    if len(train_ds) == 0:
        raise RuntimeError("No samples after filtering. Check dataset or filter keywords.")

    if args.max_samples and args.max_samples > 0 and args.max_samples < len(train_ds):
        train_ds = Subset(train_ds, list(range(args.max_samples)))
        Logger(f"Truncated dataset to {args.max_samples} samples (dry-run)")

    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None

    # ------- 9. Reward 评分器 -------
    clip_scorer = CLIPScorer(clip_model_name=args.clip_model, device=args.device)
    llm_judge = LLMJudge(
        api_key=args.judge_api_key,
        base_url=args.judge_base_url,
        model=args.judge_model,
        disable_thinking=not args.enable_thinking,
    )
    hallucination_penalizer = None
    if args.w6 > 0:
        from trainer.grpo_utils import HallucinationPenalizer
        hallucination_penalizer = HallucinationPenalizer(
            api_key=args.judge_api_key,
            base_url=args.judge_base_url,
            model=args.judge_model,
            disable_thinking=not args.enable_thinking,
        )
        Logger(f"HallucinationPenalizer enabled (w6={args.w6})")

    # ------- 10. 优化器 -------
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    # ------- 11. compile / DDP -------
    if args.use_compile == 1:
        try:
            model = torch.compile(model)
            Logger("torch.compile enabled")
        except Exception as e:
            Logger(f"Warning: torch.compile failed: {e}")

    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ------- 12. 恢复 optimizer / scaler -------
    start_epoch, start_step = 0, 0
    if ckp_data:
        try:
            optimizer.load_state_dict(ckp_data['optimizer'])
        except Exception as e:
            Logger(f"Warning: optimizer state load failed: {e}")
        if 'scaler' in ckp_data and ckp_data['scaler']:
            try:
                scaler.load_state_dict(ckp_data['scaler'])
            except Exception:
                Logger("Warning: scaler state mismatch, using fresh scaler")
        start_epoch = ckp_data.get('epoch', 0)
        start_step = ckp_data.get('step', 0)
        Logger(f"Resumed: epoch={start_epoch}, step={start_step}")

    # ------- 13. 训练循环 -------
    total_steps = args.epochs * math.ceil(len(train_ds) / max(args.batch_size, 1))

    with TrainingProgressBar(total_steps, desc="MiniQ-VL GRPO", log_interval=args.log_interval) as pbar:
        for epoch in range(start_epoch, args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            setup_seed(args.seed + epoch)

            indices = list(iter(train_sampler)) if train_sampler is not None else list(range(len(train_ds)))
            skip = start_step if (epoch == start_epoch and start_step > 0) else 0
            batch_sampler = SkipBatchSampler(indices, args.batch_size, skip)

            loader = DataLoader(
                train_ds,
                batch_sampler=batch_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
                collate_fn=grpo_collate_fn,
            )

            with autocast_ctx:
                last_step = train_one_epoch(
                    epoch, loader, model, processor, optimizer, scaler,
                    ref_model, clip_scorer, llm_judge, args,
                    start_step if skip > 0 else 0,
                    wandb_run, pbar, hallucination_penalizer,
                )

            state["last_successful_step"] = (epoch + 1) * len(loader)

    # ------- 14. 收尾 -------
    if dist.is_initialized():
        dist.destroy_process_group()
    if wandb_run:
        wandb_run.finish()
    Logger("GRPO training completed!")


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="MiniQ-VL GRPO Training (rewritten)")

    # 模型
    p.add_argument("--model_path", type=str, default="./model/Qwen3-VL-2B-Instruct")
    p.add_argument("--save_dir", type=str, default="./out")
    p.add_argument("--save_weight", type=str, default="grpo_vlm")

    # 训练
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=2, help="每卡 prompt 数")
    p.add_argument("--group_size", type=int, default=4, help="每 prompt 采样数 K (4~8)")
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--accumulation_steps", type=int, default=4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--clip_eps", type=float, default=0.2)

    # 采样
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.9)

    # Reward 权重
    p.add_argument("--w1", type=float, default=0.3, help="CLIPScore")
    p.add_argument("--w2", type=float, default=0.5, help="LLM-Judge (1-10 归一化)")
    p.add_argument("--w3", type=float, default=0.2, help="Length Penalty")
    p.add_argument("--w4", type=float, default=0.0, help="Attribute Coverage")
    p.add_argument("--w5", type=float, default=0.0, help="Diversity Reward")
    p.add_argument("--w6", type=float, default=0.0, help="Hallucination Penalty")
    p.add_argument("--length_min", type=int, default=50)
    p.add_argument("--length_max", type=int, default=300)

    # KL
    p.add_argument("--kl_coef", type=float, default=0.05)
    p.add_argument("--use_ref_model", action="store_true",
                   help="使用 ref_model 计算 KL 约束 (会额外占用显存)")

    # 日志 / 保存
    p.add_argument("--log_interval", type=int, default=5)
    p.add_argument("--save_interval", type=int, default=200)

    # 冻结
    p.add_argument("--freeze_vision", type=int, default=1, choices=[0, 1])
    p.add_argument("--freeze_language", type=int, default=0, choices=[0, 1])

    # 数据
    p.add_argument("--data_path", type=str, default="./dataset/minimind-v_dataset/sft_i2t.parquet")
    p.add_argument("--prefiltered_path", type=str, default="./dataset/minimind-v_dataset/grpo_i2t.parquet",
                   help="预筛选数据集 parquet (由 dataset/prepare_grpo_dataset.py 生成). 存在则直接加载, 跳过筛选")
    p.add_argument("--filter_mode", type=str, default="auto", choices=["auto", "llm", "keyword"])
    p.add_argument("--filter_llm_model", type=str, default=_DEFAULT_MODEL,
                   help=f"LLM 筛选模型 (默认 {_DEFAULT_MODEL}, 来自 utils/api_client.py)")
    p.add_argument("--filter_cache", type=str, default=None)
    p.add_argument("--max_samples", type=int, default=1000,
                   help="最多使用的训练样本数; 0=全部 (默认 1000, 跑全量设 0)")

    # 断点续训 / 加载
    p.add_argument("--from_resume", type=int, default=0, choices=[0, 1])
    p.add_argument("--sft_checkpoint", type=str, default=None)
    p.add_argument("--use_sft_model", type=int, default=1, choices=[0, 1])
    p.add_argument("--use_base_model", type=int, default=0, choices=[0, 1])

    # wandb
    p.add_argument("--use_wandb", type=int, default=1, choices=[0, 1])
    p.add_argument("--wandb_project", type=str, default="MiniQ-VL-GRPO")
    p.add_argument("--wandb_entity", type=str, default=None)

    # LLM-Judge
    p.add_argument("--judge_model", type=str, default=_DEFAULT_MODEL,
                   help=f"LLM-Judge 模型 (默认 {_DEFAULT_MODEL}, 来自 utils/api_client.py)")
    p.add_argument("--judge_api_key", type=str, default=None,
                   help="Judge API key (留空用 utils/api_client.py 默认值)")
    p.add_argument("--judge_base_url", type=str, default=None,
                   help="Judge base URL (留空用 utils/api_client.py 默认值)")
    p.add_argument("--enable_thinking", action="store_true",
                   help="开启 deepseek thinking 模式 (默认关闭, 走非思考模式 extra_body.thinking=disabled)")

    # CLIP
    p.add_argument("--clip_model", type=str, default="./model/clip-vit-base-patch32")

    # 优化
    p.add_argument("--use_compile", type=int, default=0, choices=[0, 1])
    p.add_argument("--use_grad_checkpoint", type=int, default=1, choices=[0, 1])
    p.add_argument("--use_enable_input_require_grads", action="store_true",
                   help="配合 grad checkpoint: 让输入层保留 grad")

    # 其它
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    state = {
        "last_successful_step": 0,
        "error_report_dir": None,
    }

    try:
        train_main(args, state)
    except KeyboardInterrupt:
        print("\n[!] Training interrupted (Ctrl+C)", flush=True)
        sys.exit(1)
    except Exception as e:
        # 终端打印异常 (类型 + message + 完整 traceback)
        print("\n" + "=" * 70, flush=True)
        print(f"  ✗ 训练失败: {type(e).__name__}: {e}", flush=True)
        print("=" * 70, flush=True)
        traceback.print_exc()
        raise
