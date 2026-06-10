"""
MiniQ-VL DPO 损失函数
"""
import torch
import torch.nn.functional as F
from typing import Optional, Dict


def compute_dpo_loss(
    model,
    ref_model,
    chosen_ids: torch.Tensor,
    chosen_mask: torch.Tensor,
    rejected_ids: torch.Tensor,
    rejected_mask: torch.Tensor,
    pixel_values: Optional[torch.Tensor] = None,
    image_grid_thw: Optional[torch.Tensor] = None,
    mm_token_type_ids: Optional[torch.Tensor] = None,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> tuple:
    """
    计算 DPO 损失
    
    Args:
        model: 策略模型
        ref_model: 参考模型 (可选)
        chosen_ids: chosen 序列 [batch, seq_len]
        chosen_mask: chosen mask [batch, seq_len]
        rejected_ids: rejected 序列 [batch, seq_len]
        rejected_mask: rejected mask [batch, seq_len]
        pixel_values: 图像张量
        image_grid_thw: 图像网格尺寸
        mm_token_type_ids: 多模态 token 类型
        beta: DPO temperature
        label_smoothing: 标签平滑
    
    Returns:
        loss, chosen_logps, rejected_logps, kl
    """
    # 计算 ref_logps (如果有 ref_model)
    with torch.no_grad():
        if ref_model is not None:
            ref_chosen_logps = get_sequence_logps(
                ref_model, chosen_ids, chosen_mask, pixel_values, image_grid_thw, mm_token_type_ids
            )
            ref_rejected_logps = get_sequence_logps(
                ref_model, rejected_ids, rejected_mask, pixel_values, image_grid_thw, mm_token_type_ids
            )
        else:
            ref_chosen_logps = 0
            ref_rejected_logps = 0
    
    # 计算策略模型 logps
    chosen_logps = get_sequence_logps(
        model, chosen_ids, chosen_mask, pixel_values, image_grid_thw, mm_token_type_ids
    )
    rejected_logps = get_sequence_logps(
        model, rejected_ids, rejected_mask, pixel_values, image_grid_thw, mm_token_type_ids
    )
    
    # DPO 损失
    # loss = -log(sigmoid(beta * (logps_pi - logps_ref)))
    # 等价于
    # policy_logps = beta * (chosen_logps - rejected_logps)
    # reference_logps = beta * (ref_chosen_logps - ref_rejected_logps)
    # loss = -log sigmoid(policy_logps - reference_logps)
    
    policy_logps = beta * (chosen_logps - rejected_logps)
    if ref_model is not None:
        reference_logps = beta * (ref_chosen_logps - ref_rejected_logps)
        logits = policy_logps - reference_logps
    else:
        logits = policy_logps
    
    # Label smoothing
    if label_smoothing > 0:
        loss = -F.logsigmoid(logits) * (1 - label_smoothing) - F.logsigmoid(-logits) * label_smoothing
    else:
        loss = -F.logsigmoid(logits)
    
    loss = loss.mean()
    
    # KL 散度
    if ref_model is not None:
        kl = (chosen_logps - ref_chosen_logps + rejected_logps - ref_rejected_logps).mean()
    else:
        kl = torch.tensor(0.0)
    
    return loss, chosen_logps, rejected_logps, kl


def get_sequence_logps(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_values: Optional[torch.Tensor] = None,
    image_grid_thw: Optional[torch.Tensor] = None,
    mm_token_type_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    计算序列的 log probabilities
    
    Args:
        model: 模型
        input_ids: [batch, seq_len]
        attention_mask: [batch, seq_len]
        pixel_values: 图像张量 (可选)
        image_grid_thw: 图像网格 (可选)
        mm_token_type_ids: 多模态 token 类型 (可选)
    
    Returns:
        logps: [batch]
    """
    # 移除 pad 部分，只保留有效 token
    # 找到最后一个非 pad token
    seq_len = attention_mask.sum(dim=1)  # [batch]
    
    # 前向传播
    fwd_kwargs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
    }
    if pixel_values is not None:
        fwd_kwargs['pixel_values'] = pixel_values
    if image_grid_thw is not None:
        fwd_kwargs['image_grid_thw'] = image_grid_thw
    if mm_token_type_ids is not None:
        fwd_kwargs['mm_token_type_ids'] = mm_token_type_ids
    # 训练时不要 KV cache
    fwd_kwargs['use_cache'] = False
    
    outputs = model(**fwd_kwargs)
    
    logits = outputs.logits  # [batch, seq_len, vocab_size]
    
    # 计算 log probabilities
    log_probs = F.log_softmax(logits, dim=-1)
    
    # 获取每个位置对应下一个 token 的 log prob
    # shifted: [batch, seq_len-1, vocab_size] -> 目标 token 是 input_ids[:, 1:]
    shifted_log_probs = log_probs[:, :-1, :].contiguous()  # [batch, seq_len-1, vocab_size]
    target_ids = input_ids[:, 1:].contiguous()  # [batch, seq_len-1]
    target_mask = attention_mask[:, 1:].contiguous()  # [batch, seq_len-1]
    
    # gather 目标 token 的 log prob
    gather_indices = target_ids.unsqueeze(-1).expand(-1, -1, logits.size(-1))  # [batch, seq_len-1, vocab_size]
    target_log_probs = torch.gather(shifted_log_probs, 2, gather_indices).squeeze(-1)  # [batch, seq_len-1]
    
    # 只保留有效位置
    target_log_probs = target_log_probs * target_mask
    seq_logps = target_log_probs.sum(dim=1)  # [batch]
    
    return seq_logps